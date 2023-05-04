#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>

#include "sift.hpp"
#include "image.hpp"

using namespace std;

namespace sift {

/*
 * (Function):
 * The goal is to create a series of images with different levels of blur to detect features at different scales.
 * (Params):
 * "img" is the input image.
 * "sigma_min" is the minimum standard deviation of the Gaussian kernel used for blurring.
 * "num_octaves" is the number of octaves in the pyramid. Each octave is a series of images where the size of the images is half of the size of the images in the previous octave.
 * "scales_per_octave" is the number of scales per octave. The number of images per octave is scales_per_octave + 3
 * (Steps):
 * The function first calculates the base image of the pyramid by resizing the input image and then applying Gaussian
 * blur to reach the required base_sigma. The function then calculates the sigma values for blurring the remaining
 * images in each octave using the formula sigma_total = k * sigma_prev, where k is a constant factor that is calculated
 * based on the number of scales per octave.The function then creates the scale space pyramid by iterating through each
 * octave and creating a series of images by applying Gaussian blur to the previous image in the series.
 * Finally, the function returns the scale space pyramid.
*/
ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min, int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = pow(2, 1.0/scales_per_octave);
    vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++)
    {
        float sigma_prev = base_sigma * pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        vector<vector<Image>>(num_octaves)
    };

    for (int i = 0; i < num_octaves; i++)
    {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++)
        {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2, Interpolation::NEAREST);
    }
    return pyramid;
}


/*
 * (Function):
 * Generate pyramid of difference of gaussians (DoG) images.
 * (Params):
 * "img_pyramid" is the input scale space pyramid.
 * (Steps):
 * The function creates an empty DoG pyramid with the same number of octaves as the input pyramid, but with one fewer
 * image per octave. It then iterates through each octave and creates a series of DoG images by subtracting adjacent
 * Gaussian blurred images in each octave. The difference of each pixel value between the two adjacent images is
 * calculated and stored in a new image. The resulting image is added to the DoG pyramid for the current octave.
 * Finally, the function returns the DoG pyramid.
*/
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid =
    {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        vector<vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}


/*
 * (Function):
 * This code defines a function to determine if a given point in a scale space pyramid of images is an extremum
 * (i.e. a potential keypoint).
 * (Params):
 * "octave" is a vector of images in the current octave.
 * "scale" is the index of the image in the octave.
 * "x" and "y" are the coordinates of the pixel in the image.
 * (Steps):
 * The function first retrieves the current image, as well as the adjacent images in the octave. It then sets two
 * boolean flags to true, indicating that the current pixel is a candidate for both a maximum and a minimum value in
 * the 3x3 neighborhood.
 * The function then loops through the 3x3 neighborhood of the pixel in the current, previous, and next images.
 * For each neighbor, it compares the pixel value to the current pixel value and updates the minimum and maximum
 * flags accordingly.
 * If the current pixel is not a minimum or maximum value in any of the 3x3 neighborhoods, the function returns false.
 * Otherwise, it returns true, indicating that the current pixel is an extremum.
*/
bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}


/*
 * (Function):
 * fit a quadratic near the discrete extremum, update the keypoint (interpolated) extremum value and return offsets of
 * the interpolated extremum from the discrete extremum
 * (Params):
 * "kp" A reference to a Keypoint object, which contains the location and other information about the keypoint being processed.
 * "octave" A constant reference to a vector of Image objects representing the octave at the given scale.
 * "scale" An integer representing the index of the scale in the octave that the keypoint belongs to.
 * (Steps):
 * The function takes in a Keypoint kp, a vector of images octave, and an integer scale.
 * It retrieves the images corresponding to the current scale, the previous scale, and the next scale from octave.
 * It computes the gradient of the image at kp.i, kp.j by taking the difference of the pixel values at the corresponding
 * location in the next and previous scale images, divided by 2.
 * It computes the Hessian of the image at kp.i, kp.j. The Hessian is a matrix of second-order partial derivatives of
 * the image. Specifically, it computes the Hessian Matrix.
 * It then calculates the inverse of the Hessian matrix, which is needed to find the offsets of the interpolated extremum
 * from the discrete extremum. The determinant of the Hessian matrix is also computed as it is used to calculate the
 * inverse.
 * Using the inverse Hessian and the gradient, it computes the offsets of the interpolated extremum from the discrete
 * extremum in the scale, x, and y directions.
 * It computes the value of the interpolated extremum by adding the interpolated offsets to the value of the pixel at
 * the discrete extremum location.
 * Finally, it updates the value of kp.extremum_val and returns a tuple containing the offsets in the scale, x, and y
 * directions.
*/
tuple<float, float, float> fit_quadratic(Keypoint& kp, const vector<Image>& octave, int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // Gradient
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // Hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // Invert Hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // Find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0) + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}


/*
 * (Function):
 * This function checks if a given keypoint is on an edge or not, based on the "edgeness" metric calculated from the
 * Hessian matrix of the image at the keypoint location.
 * (Params):
 * "kp" a Keypoint object that represents the keypoint to be tested.
 * "octave" a vector of Image objects representing the octave of the image pyramid at the scale of the keypoint.
 * "edge_thresh" a float value representing the edge threshold to use in the edgeness metric calculation. The default value is C_EDGE, which is a constant defined elsewhere in the code.
*/
bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}


/*
 * (Function):
 * The function find_input_img_coords takes a Keypoint object kp and several offsets (offset_s, offset_x, offset_y).
 *  It then computes the coordinates of the keypoint on the original input image.
 * (Steps):
 * Compute the scale of the keypoint using the octave and scale values stored in kp, as well as the sigma_min value.
 * Compute the x and y coordinates of the keypoint on the scale-space image using the i and j values stored in kp,
 * as well as the min_pix_dist value.
 * Compute the final x and y coordinates of the keypoint on the input image by scaling the coordinates from step 2 using
 * the scale factor computed in step 1 and the n_spo value.
 * Set the sigma, x, and y values of kp to the computed values from steps 1-3.
*/
void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y, float sigma_min=SIGMA_MIN,
                           float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = pow(2, kp.octave) * sigma_min * pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * pow(2, kp.octave) * (offset_y+kp.j);
}


/*
 * (Function):
 * It takes a Keypoint object kp, a vector of Images representing an octave, a contrast threshold contrast_thresh,
 * and an edge threshold edge_thresh as input. It tries to refine the Keypoint coordinates by iteratively fitting a
 * quadratic function to the nearby pixels in the octave, and updates the Keypoint coordinates according to the
 * quadratic fit.
 * (Params):
 * "kp" A reference to a Keypoint object to be refined or discarded.
 * "octave" A vector of Image objects representing the Gaussian pyramid octave where the keypoint was found.
 * "contrast_thresh" A threshold value used to determine if the keypoint has sufficient contrast. A keypoint is discarded if its extremum value is below this threshold.
 * "edge_thresh" A threshold value used to determine if the keypoint is on an edge. A keypoint is discarded if its edgeness score (based on the Hessian matrix) is above this threshold.
 * (Steps):
 * This function takes a keypoint, an octave (a set of images with different scales), a contrast threshold, and an edge
 * threshold as input. The function refines the keypoint's location by fitting a quadratic function to the neighboring
 * pixels, and then iteratively adjusting the keypoint's location based on the extremum of the quadratic function until
 * the maximum iteration number is reached or the keypoint meets certain validity conditions. If the keypoint is valid,
 * it updates the keypoint's scale, coordinates, and sigma values and returns true. Otherwise, it returns false
*/
bool refine_or_discard_keypoint(Keypoint& kp, const vector<Image>& octave, float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS)
    {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = max({abs(offset_s), abs(offset_x), abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += round(offset_s);
        kp.i += round(offset_x);
        kp.j += round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}


/*
 * (Function):
 * It takes a difference-of-Gaussian scale space pyramid, a contrast threshold, and an edge threshold, and returns a
 * vector of keypoints representing detected interest points in the input image.
 * (Params):
 * "dog_pyramid" A reference to a ScaleSpacePyramid object that represents the Difference-of-Gaussian (DoG) scale space pyramid.
 * "contrast_thresh" A threshold value for eliminating low-contrast keypoints. Keypoints with a magnitude below this threshold are discarded.
 * "edge_thresh" A threshold value for eliminating keypoints that are likely to lie on an edge. Keypoints with an edgeness value above this threshold are discarded. The edgeness value is computed based on the determinant and trace of the Hessian matrix of the image at the keypoint location.
 * (Steps):
 * - This function takes a Difference of Gaussians (DoG) scale space pyramid, a contrast threshold, and an edge threshold
 * as inputs. It iterates over each octave, each scale level, and each pixel in the scale space pyramid.
 * - For each pixel, it checks if the absolute value of the pixel value is above 0.8 times the contrast threshold.
 * If not, it continues to the next pixel. If the pixel is above the contrast threshold, it checks if it is an extremum
 * (maximum or minimum) by comparing it to its 26 neighbors in the current and adjacent scales.
 * - If the pixel is an extremum, a new keypoint is created at the pixel location with octave, scale, and image
 * coordinates. The keypoint is then refined using a quadratic fit and checked for validity based on contrast and edge
 * thresholds. If the keypoint is valid, it is added to the list of keypoints.
 * - The function returns a vector of valid keypoints found in the scale space pyramid.
*/
vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh, float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    for (int i = 0; i < dog_pyramid.num_octaves; i++)
    {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++)
        {
            const Image& img = octave[j];
            for (int x = 1; x < img.width-1; x++)
            {
                for (int y = 1; y < img.height-1; y++)
                {
                    if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh)
                    {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y))
                    {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);
                        if (kp_is_valid)
                        {
                            keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}


/*
 * (Function):
 * Calculate x and y derivatives for all images in the input pyramid.
 * (Params):
 * "pyramid" A reference to a ScaleSpacePyramid object that represents the Difference-of-Gaussian (DoG) scale space pyramid.
 * (Steps):
 * This function takes in a ScaleSpacePyramid object and generates a new ScaleSpacePyramid object that contains gradient
 * images computed from the input pyramid. The gradient images are computed using central difference approximation for
 * the x and y derivatives of the input images. The resulting gradient pyramid has the same number of octaves and
 * images per octave as the input pyramid, but the images are two-channel, with the first channel corresponding to
 * the x-gradient and the second channel corresponding to the y-gradient.
*/
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int i = 0; i < pyramid.num_octaves; i++)
    {
        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        for (int j = 0; j < pyramid.imgs_per_octave; j++)
        {
            Image grad(width, height, 2);
            float gx, gy;
            for (int x = 1; x < grad.width-1; x++)
            {
                for (int y = 1; y < grad.height-1; y++)
                {
                    gx = (pyramid.octaves[i][j].get_pixel(x+1, y, 0)
                         -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) * 0.5;
                    grad.set_pixel(x, y, 0, gx);
                    gy = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                         -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) * 0.5;
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            grad_pyramid.octaves[i].push_back(grad);
        }
    }
    return grad_pyramid;
}


/*
 * (Function):
 * Convolve 6x with box filter.
 * (Params):
 * "hist[N_BINS]" float array of size N_BINS as input, which represents a histogram. It does not return anything.
 * (Steps):
 * temporary histogram tmp_hist to store the smoothed values, and iteratively performs a weighted average of each bin's
 * value with its two neighboring bins' values. This is done six times, and the final smoothed histogram is stored in
 * the original hist array.
*/
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < N_BINS; j++)
        {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++)
        {
            hist[j] = tmp_hist[j];
        }
    }
}


/*
 * (Function):
 * The function calculates the gradient orientation histogram of the gradient image at the location of the Keypoint
 * object, and then smooths the histogram using the smooth_histogram function. Finally, the function extracts the
 * reference orientations of the Keypoint object from the smoothed histogram and returns them as a vector<float>
 * object.
 * (Params):
 * "kp" A Keypoint object representing a keypoint detected in an image. The Keypoint object contains information about the location, scale, and orientation of the keypoint, as well as its descriptor.
 * "grad_pyramid" A ScaleSpacePyramid object representing a gradient pyramid of the image that the keypoint was detected in. The gradient pyramid is used to compute the gradient orientation histogram.
 * "lambda_ori" A floating-point value representing the scaling factor for the patch size used to compute the gradient orientation histogram.
 * "lambda_desc" A floating-point value representing the scaling factor for the patch size used to compute the keypoint descriptor.
*/
vector<float> find_keypoint_orientations(Keypoint& kp, const ScaleSpacePyramid& grad_pyramid, float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma)
    {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++)
    {
        for (int y = y_start; y <= y_end; y++)
        {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))/(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++)
    {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++)
    {
        if (hist[j] >= ori_thresh * ori_max)
        {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}


/*
 * (Function):
 * This function takes as input a 3D array of float histograms with dimensions N_HIST x N_HIST x N_ORI and outputs a
 * feature vector represented as a array<uint8_t, 128>.
 * (Steps):
 * The function first flattens the 3D array into a 1D array of size N_HISTN_HISTN_ORI. It then normalizes the array
 * by calculating the L2 norm of all the values and dividing each value by this norm. After normalization, the array
 * is clipped so that each value is less than or equal to 0.2 times the L2 norm of the array. The array is then
 * renormalized again. Finally, each value in the array is scaled by 512 and rounded down to the nearest integer.
 * The resulting value is then clipped so that it is between 0 and 255, and stored in the corresponding element of the
 * output feature vector.
*/
void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y, float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++)
    {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++)
        {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++)
            {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}


/*
 * (Function):
 * This function converts a set of histograms into a compact feature vector that can be used for further processing or
 * analysis, such as matching and recognition of image features.
 * (Steps):
 * The function first calculates the L2-norm of the input histograms by summing the squares of all the values in the
 * histograms, and taking the square root of the result. Then, it normalizes each element in the histograms by dividing
 * it by this L2-norm, and computes the new L2-norm. If the new L2-norm is still larger than a threshold
 * (0.2 times the original L2-norm), the elements in the histograms are further normalized by dividing them by the new
 * L2-norm. Finally, each element in the normalized histograms is multiplied by 512 and rounded down to the nearest
 * integer, and then clipped to the range [0, 255]. The resulting integers are stored in the feature_vec array.
*/
void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++)
    {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++)
    {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++)
    {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}


/*
 * (Function):
 * This function computes the descriptor for a keypoint given a gradient pyramid and other parameters. The descriptor
 * is represented as a 128-dimensional feature vector.
 * (Steps):
 * It computes the pixel distance for the current octave and gets the gradient image for the current scale of the keypoint.
 * It defines a patch size based on the scale of the keypoint and the desired number of histograms.
 * It loops over the pixels within the patch and accumulates samples into orientation histograms.
 * It uses the orientation histograms to build a feature vector.
*/
void compute_keypoint_descriptor(Keypoint& kp, float theta, const ScaleSpacePyramid& grad_pyramid, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++)
    {
        for (int n = y_start; n <= y_end; n++)
        {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}


/*
 * (Function):
 * This code implements a keypoint detection and descriptor computation pipeline based on Difference of Gaussians (DoG)
 * and SIFT (Scale-Invariant Feature Transform) algorithms.
 * (Steps):
 * Convert the input image to grayscale if necessary.
 * Generate a Gaussian scale-space pyramid with num_octaves octaves and scales_per_octave scales per octave, starting with a base scale of sigma_min.
 * Generate a Difference of Gaussians (DoG) scale-space pyramid from the Gaussian pyramid.
 * Detect keypoints in the DoG pyramid by finding maxima or minima across space and scales, and reject keypoints with low contrast or on edges.
 * Generate a gradient magnitude and orientation scale-space pyramid from the Gaussian pyramid.
 * For each detected keypoint, compute the orientation(s) by constructing histograms of gradient orientations in a circular window around the keypoint and selecting the peaks in the histogram(s). If there are multiple peaks above a certain threshold, create new keypoints with the same location and scale as the original keypoint but with different orientations.
 * For each keypoint and orientation, compute a SIFT descriptor by accumulating gradient magnitude and orientation information in a patch around the keypoint and orientation and converting it to a vector of 128 floating-point values.
*/
vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,int num_octaves, int scales_per_octave,
                                           float contrast_thresh, float edge_thresh, float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves, scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    vector<Keypoint> kps;

    for (Keypoint& kp_tmp : tmp_kps)
    {
        vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid, lambda_ori, lambda_desc);
        for (float theta : orientations)
        {
            Keypoint kp = kp_tmp;
            compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
            kps.push_back(kp);
        }
    }
    return kps;
}


/*
 * (Function):
 * The function calculates the cross-correlation between two 128-dimensional feature descriptors, descriptor1 and
 * descriptor2.
 * (Steps):
 * Firstly, The function first calculates the mean, variance, and standard deviation of both descriptors. It then
 * calculates the cross-correlation between the two descriptors by taking the dot product of the normalized descriptors.
 * The cross-correlation value ranges from -1 to 1, where a value of 1 indicates that the two descriptors are identical,
 * 0 indicates that they are uncorrelated, and -1 indicates that they are negatively correlated.
 * This function is commonly used in computer vision tasks such as image matching and object recognition, where features
 * of images are compared to identify matches or similarities between them.
*/
double crossCorrelation(const std::array<uint8_t, 128>& descriptor1, const std::array<uint8_t, 128>& descriptor2) {
    double sum = 0;
    double descriptor1Mean = 0, descriptor2Mean = 0;
    double descriptor1Var = 0, descriptor2Var = 0;
    double descriptor1StdDev = 0, descriptor2StdDev = 0;

    int descriptorLength = descriptor1.size();

    // Calculate mean of descriptor1 and descriptor2
    for (int i = 0; i < descriptorLength; i++)
    {
        descriptor1Mean += descriptor1[i];
        descriptor2Mean += descriptor2[i];
    }

    descriptor1Mean /= descriptorLength;
    descriptor2Mean /= descriptorLength;

    // Calculate variance and standard deviation of descriptor1 and descriptor2
    for (int i = 0; i < descriptorLength; i++)
    {
        double descriptor1Val = descriptor1[i] - descriptor1Mean;
        double descriptor2Val = descriptor2[i] - descriptor2Mean;
        descriptor1Var += descriptor1Val * descriptor1Val;
        descriptor2Var += descriptor2Val * descriptor2Val;
    }

    descriptor1Var /= descriptorLength;
    descriptor2Var /= descriptorLength;
    descriptor1StdDev = sqrt(descriptor1Var);
    descriptor2StdDev = sqrt(descriptor2Var);

    // Calculate cross-correlation
    for (int i = 0; i < descriptorLength; i++)
    {
        double descriptor1Val = descriptor1[i] - descriptor1Mean;
        double descriptor2Val = descriptor2[i] - descriptor2Mean;
        sum += descriptor1Val * descriptor2Val;
    }

    return sum / (descriptor1StdDev * descriptor2StdDev);
}


/*
 * (Function):
 * function takes in two descriptors of type array<uint8_t, 128> and calculates the sum of squared differences
 * (SSD) between them. It returns a double value representing the SSD.
 * (Steps):
 * The function loops through each element of the two descriptors, calculates the difference between the corresponding
 * elements, squares the difference and adds it to a running sum. The final sum is returned as the SSD between the two
 * descriptors.
 */
 double ssd(const std::array<uint8_t, 128>& descriptor1, const std::array<uint8_t, 128>& descriptor2)
 {
     double sum = 0;
     for (int i = 0; i < descriptor1.size(); i++)
     {
         double diff = descriptor1[i] - descriptor2[i];
         sum += diff * diff;
     }
     return sum;
 }


/*
 * (Function):
 * This is a function that finds matches between two sets of keypoints, a and b. The function takes in a relative
 * threshold, an absolute threshold, and a mode (corr or ssd).
 * (Steps):
 * The function takes in two vectors of keypoints a and b, along with two threshold values thresh_relative and
 * thresh_absolute, and a string mode which specifies the matching mode (either "corr" or "ssd").
 * The function checks if the input vectors a and b have at least two keypoints. If not, it throws an assertion error.
 * The function creates an empty vector of pairs matches to store the keypoint matches.
 * For each keypoint in vector a, the function searches for its two nearest neighbors in vector b, based on either
 * cross-correlation or sum of squared differences (depending on the mode).
 * If the nearest neighbor (nn1) has a correlation/SSD value that is greater/smaller than a threshold value
 * thresh_absolute, and greater/smaller than a relative threshold value thresh_relative times the value of the second
 * nearest neighbor (nn2), then the function adds the pair of keypoint indices (i, nn1) to the matches vector.
 * After iterating through all keypoints in vector a, the function returns the matches vector containing the matched
 * keypoints.
*/
vector<pair<int, int>> find_keypoint_matches(const vector<Keypoint>& a,const vector<Keypoint>& b, double thresh_relative,
                                            double thresh_absolute, string mode)
 {
    if(mode=="corr")
    {
    assert(a.size() >= 2 && b.size() >= 2);

    vector<pair<int, int>> matches;

    for (int i = 0; i < a.size(); i++)
    {
        // find two nearest neighbours in b for current keypoint from a
        int nn1_idx = -1;
        double nn1_corr = -1, nn2_corr = -1;
        for (int j = 0; j < b.size(); j++)
        {
             double corr = crossCorrelation(a[i].descriptor, b[j].descriptor);
             if (corr > nn1_corr)
             {
                nn2_corr = nn1_corr;
                nn1_corr = corr;
                nn1_idx = j;
             }
             else if (nn1_corr >= corr && corr > nn2_corr)
             {
                nn2_corr = corr;
             }
        }
        if (nn1_corr > thresh_relative * nn2_corr && nn1_corr > thresh_absolute)
        {
            matches.push_back({i, nn1_idx});
        }
    }
    return matches;
    }
    else
    {
        assert(a.size() >= 2 && b.size() >= 2);
        vector<pair<int, int>> matches;

        for (int i = 0; i < a.size(); i++)
        {
            // find two nearest neighbours in b for current keypoint from a
            int nn1_idx = -1;
            double nn1_ssd = std::numeric_limits<double>::max(), nn2_ssd = std::numeric_limits<double>::max();
            for (int j = 0; j < b.size(); j++)
            {
                double ssd_val = ssd(a[i].descriptor, b[j].descriptor);
                if (ssd_val < nn1_ssd)
                {
                    nn2_ssd = nn1_ssd;
                    nn1_ssd = ssd_val;
                    nn1_idx = j;
                }
                else if (nn1_ssd <= ssd_val && ssd_val < nn2_ssd)
                {
                    nn2_ssd = ssd_val;
                }
            }
            if (nn1_ssd < thresh_relative * nn2_ssd && nn1_ssd < thresh_absolute)
            {
                matches.push_back({i, nn1_idx});
            }
        }
        return matches;
    }
 }


/*
 * (Function):
 * This function takes an input image and a vector of keypoints and returns a new image with the keypoints drawn on it.
 * If the input image is grayscale, it first converts it to RGB.
 * (Steps):
 * Then, for each keypoint in the vector, it calls the draw_point function with the coordinates of the keypoint and a
 * fixed size of 5 pixels. The draw_point function modifies the image in place by drawing a circle or a cross at the
 * specified coordinates.
 * Finally, the function returns the modified image with the keypoints drawn on it.
*/
Image draw_keypoints(const Image& img, const vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1)
    {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps)
    {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}


/*
 * (Function):
 * The function draw_matches takes two images, two vectors of keypoints for each image, and a vector of matching
 * keypoint pairs, and returns an image with lines drawn connecting the matched keypoints.
 * (Steps):
 * This function takes in two images, a and b, along with their respective keypoint vectors kps_a and kps_b, and a
 * vector of matches between the keypoints matches. It creates a new image res with a width equal to the sum of the
 * widths of a and b, and a height equal to the maximum of the heights of a and b.
 * The function then copies the pixels from a and b into res, adjusting the coordinates of the pixels from b to account
 * for the width of a. Finally, it draws lines connecting the keypoints in a and b that have been matched according to
 * matches.
 * The resulting image res displays the two input images side-by-side with lines connecting the matched keypoints.
*/
Image draw_matches(const Image& a, const Image& b, vector<Keypoint>& kps_a, vector<Keypoint>& kps_b,
                   vector<pair<int, int>> matches)
{
    Image res(a.width+b.width, std::max(a.height, b.height), 3);

    for (int i = 0; i < a.width; i++)
    {
        for (int j = 0; j < a.height; j++)
        {
            res.set_pixel(i, j, 0, a.get_pixel(i, j, 0));
            res.set_pixel(i, j, 1, a.get_pixel(i, j, a.channels == 3 ? 1 : 0));
            res.set_pixel(i, j, 2, a.get_pixel(i, j, a.channels == 3 ? 2 : 0));
        }
    }

    for (int i = 0; i < b.width; i++)
    {
        for (int j = 0; j < b.height; j++)
        {
            res.set_pixel(a.width+i, j, 0, b.get_pixel(i, j, 0));
            res.set_pixel(a.width+i, j, 1, b.get_pixel(i, j, b.channels == 3 ? 1 : 0));
            res.set_pixel(a.width+i, j, 2, b.get_pixel(i, j, b.channels == 3 ? 2 : 0));
        }
    }

    for (auto& m : matches)
    {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_line(res, kp_a.x, kp_a.y, a.width+kp_b.x, kp_b.y);
    }
    return res;
}


/*
 * (Function):
 * This function draws a rectangle on an image with the specified coordinates, width, height, and color.
 * (Steps):
 * The rectangle is drawn using a nested loop that iterates over the pixels within the specified width and height.
 * For each pixel, if it is on the edge of the rectangle (determined by its position relative to the rectangle center),
 * its color is set to the specified color using the set_pixel function. The final output image will have the rectangle
 * drawn on it.
*/
void draw_rectangle(Image& img, int x, int y, int width, int height, const std::array<uint8_t, 3>& color) {
    for (int i = -width/2; i <= width/2; i++)
    {
        for (int j = -height/2; j <= height/2; j++)
        {
            if (i == -width/2 || i == width/2 || j == -height/2 || j == height/2)
            {
                img.set_pixel(x + i, y + j, 0, color[0]);
                img.set_pixel(x + i, y + j, 1, color[1]);
                img.set_pixel(x + i, y + j, 2, color[2]);
            }
        }
    }
}


/*
 * (Function):
 * This function takes two input images and two vectors of keypoints for each image, along with a vector of matches
 * between them. It draws a rectangle around each keypoint in both images and then draws lines connecting the matching
 * keypoints between the two images. The resulting image is returned.
 * (Steps):
 * Create a new image res with dimensions to fit both a and b images side by side.
 * Copy the pixel values of the first image a into the left side of res.
 * Copy the pixel values of the second image b into the right side of res.
 * Define the color, width, and height of the rectangles to be drawn around the keypoints.
 * For each pair of matching keypoints, draw a rectangle centered at the x and y coordinates of the keypoint in both a and b images using the draw_rectangle function.
 * For each pair of matching keypoints, draw a line between the keypoints in both a and b images using the draw_line function.
 * Return the final image res.
*/
Image draw_matches_rect(const Image& a, const Image& b, vector<Keypoint>& kps_a,
                        vector<Keypoint>& kps_b, vector<pair<int, int>> matches)
{
    Image res(a.width+b.width, max(a.height, b.height), 3);

    for (int i = 0; i < a.width; i++)
    {
        for (int j = 0; j < a.height; j++)
        {
            res.set_pixel(i, j, 0, a.get_pixel(i, j, 0));
            res.set_pixel(i, j, 1, a.get_pixel(i, j, a.channels == 3 ? 1 : 0));
            res.set_pixel(i, j, 2, a.get_pixel(i, j, a.channels == 3 ? 2 : 0));
        }
    }

    for (int i = 0; i < b.width; i++)
    {
        for (int j = 0; j < b.height; j++)
        {
            res.set_pixel(a.width+i, j, 0, b.get_pixel(i, j, 0));
            res.set_pixel(a.width+i, j, 1, b.get_pixel(i, j, b.channels == 3 ? 1 : 0));
            res.set_pixel(a.width+i, j, 2, b.get_pixel(i, j, b.channels == 3 ? 2 : 0));
        }
    }

    array<uint8_t, 3> rectangle_color = {255, 0, 0};
    int rectangle_width = 10;
    int rectangle_height = 10;

    for (auto& m : matches)
    {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_rectangle(res, kp_a.x, kp_a.y, rectangle_width, rectangle_height, rectangle_color);
        draw_rectangle(res, a.width + kp_b.x, kp_b.y, rectangle_width, rectangle_height, rectangle_color);
    }

    for (auto& m : matches)
    {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_line(res, kp_a.x, kp_a.y, a.width+kp_b.x, kp_b.y);
    }
    return res;
}

} // namespace sift
