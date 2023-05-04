#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

/* (Constructor):
 * This is a constructor function for an Image class.
 * It uses the stb_image library to load the image data from the file and stores it in a new float array called data
 *
 * (params):
 * "file_path":input file path
 *
 * (steps):
 * 1) The stbi_load function from the stb_image library is called to load the image data from the file.
 *    The function takes the file path, as well as pointers to the width, height, channels, and a desired number of channels
 *    (in this case, 0 to keep the original number of channels) as arguments.The image data is loaded into a unsigned char pointer called img_data.
 *
 * 2) The constructor checks if img_data is null. If so, it prints an error message using std::cerr and exits the program with an error code of 1
 *
 * 3) The size of the image (in number of pixels) is calculated by multiplying the width, height, and number of channels
 *
 * 4) A new float array called data is created with size elements using the "new" operator.
 *
 * 5) A nested loop iterates through every pixel in the image, and for each pixel, it iterates through each color channel.
 *    The src_idx variable is used to calculate the index of the current pixel in img_data, and the dst_idx variable is used to calculate the index of the current pixel in the data array.
 *    The value of the pixel in img_data is normalized by dividing it by 255., and the result is stored in the corresponding element of the data array.
 *
 * 6) If the image has 4 channels (RGBA), then the constructor sets the number of channels to 3 by discarding the alpha channel.
 *
 * 7) The stbi_image_free function is called to free the memory used by img_data.
 */
Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3;
    stbi_image_free(img_data);
}




/* (constructor):
 * This is a constructor function for an Image class which Allocate memory for the image data
 *
 * (params):
 * "w": width
 * "h": height
 * "c": no of channels
 *
 * (steps):
 * 1) Initialize the width, height, and channels member variables to the provided arguments.
 * 2) Calculate the size of the image data array based on the provided width, height, and channels.
 * 3) Allocate memory for the image data array using the new operator.
 * 4) Initialize all elements of the data array to zero.
 */
Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}




/* (constructor):
 * This code is a constructor for the Image class that creates a new Image object with default values of zero for
 * its width, height, channels, size, and data array
 */
Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}




/*(Destructor):
 * Its purpose is to release the dynamically allocated memory used by the object's data array
 *  when the object goes out of scope or is explicitly deleted.
 */
Image::~Image()
{
    delete[] this->data;
}




/*(constructor):
 * This code is a copy constructor for the Image classwhich creates a new Image object as a copy of an
 * existing Image object.
 *
 * (parms):
 * "Image"
 */
Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}




/*(function):
 *  provide a way to copy the contents of one Image object to another, in a way that avoids memory leaks
 *  and ensures that the new object has its own separate copy of the image data.
 *(parms):
 * "Image"
 */
Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}




/*(Move Constructor)
 * This code defines a move constructor for the Image class. The move constructor is used to initialize a
 * new Image object using an existing Image object that is about to be destroyed or is no longer needed
 */
Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    other.data = nullptr;
    other.size = 0;
}




/* (move assignment operator)
 * This code implements a move assignment operator for the Image class. The move assignment operator allows an
 * object to be assigned the contents of another object that is about to be destroyed. The purpose of this code is to move the resources
 * (i.e., dynamically allocated memory) owned by the other object to the this object, and then invalidate the resources owned by the other object.
 */
Image& Image::operator=(Image&& other)
{
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}




/*(Method):
 * Saving Img as JPG
 *
 *(Parms):
 * "file_path":Where the img is going to be saved
 *
 *(Steps):
 * 1) Allocate a new array of unsigned chars with a size of width * height * channels.
 * 2) Iterate through each pixel in the image using nested loops, and for each pixel, iterate through each color channel.
 * 3) Calculate the source and destination indices of the current pixel in the out_data array and the data array, respectively.
 * 4) Multiply the pixel value in data by 255 and round it to the nearest integer to get the pixel value in the range of 0 to 255, and store it in the corresponding element of the out_data array.
 * 5) Call the stbi_write_jpg function from the stb_image library to save the image to a file in JPEG format with the specified file path, width, height, channels, quality (100), and image data.
 * 6) Load the saved image using the imread function from the OpenCV library and display it in a window named "My Image" using the imshow function.
 * 7) If the saving operation is not successful, print an error message to the standard error stream.
 * 8) Free the memory allocated for the out_data array.
 */
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    cv::Mat image = cv::imread(file_path);
    cv::imshow("My Image", image);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}




/*(method):
 *   it sets the value of the pixel in the data array.
 *
 *(parms):
 * "x":width
 * "y":height
 * "c":Channels
 * "val":pixel value
 *
 * (steps):
 *  1) Checks if the given coordinates x, y, and c are within the bounds of the image's dimensions and number of channels.
 *  2) If any of the coordinates are out of bounds, an error message is printed and the program is exited with a status of 1.
 *  3) Otherwise, the pixel value at the given coordinates is set to the given value val using a formula that calculates the appropriate index into the data array.
 */
void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}




/*(method):
 *  It first clamps the indices within the bounds of the image. It then returns the value of the pixel in the data array.
 *
 *(params):
 * "x":width
 * "y":height
 * "c":Channels
 *
 *(steps):
 *  1) Checks if the given coordinates x, y, and c are within the bounds of the image's dimensions and number of channels.
 *  2) If x or y is out of bounds, it is clamped to the nearest valid value.
 *  3) The pixel value at the given coordinates is retrieved from the data array using a formula that calculates the appropriate index.
 */
float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}




/*(method)
 * ensure that the image data is within the valid range for display or further processing
 *
 *(steps):
 *1) Computes the total number of pixels in the image as width * height * channels.
 *2) Iterates over each pixel value in the data array.
 *3) Clamps each pixel value to the range [0, 1] using the formula val = (val > 1.0) ? 1.0 : val; val = (val < 0.0) ? 0.0 : val;
 */
void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}




/*(funciton):
 * map coordinate from 0-current_max range to 0-new_max range
 */
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}




/*(function):
 *  Resize the Image
 *
 *(Params):
 *"new_w":New Width of picture
 *"new_h":New Height of picture
 *"method": Interpolation method used
 *
 *(Steps):
 * 1) Create a new Image object called resized with the specified width, height, and number of channels.
 * 2) Initialize a variable called value to 0.
 * 3) Loop through each pixel of the new Image resized using two nested loops, one for the x-coordinate and one for the y-coordinate.
 * 4) For each pixel, loop through each channel of the resized Image.
 * 5) Calculate the corresponding old x and y coordinates in the original Image using the map_coordinate function.
 * 6) Depending on the specified interpolation method, either call the bilinear_interpolate or nn_interpolate function to interpolate the pixel value at the old x and y coordinates for the current channel.
 * 7) Set the pixel value of the current channel in the new Image resized to the interpolated value.
 */
Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}




/*(function):
 * The bilinear_interpolate function calculates the interpolated value of a given pixel coordinate in an image using bilinear interpolation.
 * It first finds the four nearest pixels around the given coordinate and calculates the weighted average of their values
 * based on the distance from the coordinate. It then returns the final interpolated value.
 *
 *(params):
 * "image":Image
 * "x":Width
 * "y":Height
 * "c":no of channels
 *
 *(Steps):
 *  1) The function bilinear_interpolate takes an image (img), a floating-point x-coordinate (x), a floating-point y-coordinate (y), and an integer channel index (c) as inputs.
 *  2) The function first calculates the four pixel values that surround the point (x,y) using get_pixel function. These pixels are assigned to p1, p2, p3, and p4.
 *  3) The function then calculates the weighted average of the two horizontal pixel values p1 and p3 using (y_ceil-y)*p1 + (y-y_floor)*p3, where y_floor and y_ceil are the integer floor and ceiling of y,
 *     respectively. This result is assigned to q1.
 *  4) The function then calculates the weighted average of the two horizontal pixel values p2 and p4 using (y_ceil-y)*p2 + (y-y_floor)*p4. This result is assigned to q2.
 *  5) the function calculates the weighted average of the two vertical pixel values q1 and q2 using (x_ceil-x)*q1 + (x-x_floor)*q2, where x_floor and x_ceil are the integer floor and ceiling of x, respectively. This result is returned as the output of the function
 */
float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}




/*(Function):
 * The nn_interpolate function implements nearest-neighbor interpolation for an image pixel. Given an image img,
 * and floating point coordinates x and y, representing a position in the image, and an integer channel index c,
 * the function retrieves the value of the nearest pixel in the image for that channel using the get_pixel function.
 * The std::round function is used to round the floating point coordinates to the nearest integer values, which represent
 * the indices of the nearest pixel in the image.The function then returns the value of that pixel in the specified channel c
 */
float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}




/*(function):
 * changing RGB To Gray scale image
 */
Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}




/*(function):
 * changing gray scale To RGB image
 */
Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}




/*(Function)
 * Apply Gussian blur on image
 */
Image gaussian_blur(const Image& img, float sigma)
{
    assert(img.channels == 1);

    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++)
        kernel.data[k] /= sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    // convolve vertical
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                sum += img.get_pixel(x, y+dy, 0) * kernel.data[k];
            }
            tmp.set_pixel(x, y, 0, sum);
        }
    }
    // convolve horizontal
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                sum += tmp.get_pixel(x+dx, y, 0) * kernel.data[k];
            }
            filtered.set_pixel(x, y, 0, sum);
        }
    }
    return filtered;
}




/*(function):
 * Draw Points on the Image
 */
void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}




/*(function):
 * Draw lines on the Image
 */
void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}

