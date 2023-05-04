#ifndef SIFT_H
#define SIFT_H

#include <vector>
#include <array>
#include <cstdint>

#include "image.hpp"

using namespace std;

namespace sift {

struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves; 
};

struct Keypoint {
    // discrete coordinates
    int i;
    int j;
    int octave;
    int scale; //index of gaussian image inside the octave

    // continuous coordinates (interpolated)
    float x;
    float y;
    float sigma;
    float extremum_val; //value of interpolated DoG extremum
    
    std::array<uint8_t, 128> descriptor;
};

//*******************************************
// SIFT algorithm parameters, used by default
//*******************************************

// digital scale space configuration and keypoint detection
const int MAX_REFINEMENT_ITERS = 5;
const float SIGMA_MIN = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = 8;
const int N_SPO = 3;
const float C_DOG = 0.015;
const float C_EDGE = 10;

// computation of the SIFT descriptor
const int N_BINS = 36;
const float LAMBDA_ORI = 1.5;
const int N_HIST = 4;
const int N_ORI = 8;
const float LAMBDA_DESC = 6;

// feature matching
const float THRESH_ABSOLUTE = 350;
const float THRESH_RELATIVE = 0.7;

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min=SIGMA_MIN,
                                            int num_octaves=N_OCT, int scales_per_octave=N_SPO);

ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid);

vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid,
                                     float contrast_thresh=C_DOG, float edge_thresh=C_EDGE);

ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid);

vector<float> find_keypoint_orientations(Keypoint& kp, const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori=LAMBDA_ORI, float lambda_desc=LAMBDA_DESC);

void compute_keypoint_descriptor(Keypoint& kp, float theta, const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc=LAMBDA_DESC);

vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min=SIGMA_MIN,
                                                     int num_octaves=N_OCT, 
                                                     int scales_per_octave=N_SPO, 
                                                     float contrast_thresh=C_DOG,
                                                     float edge_thresh=C_EDGE,
                                                     float lambda_ori=LAMBDA_ORI,
                                                     float lambda_desc=LAMBDA_DESC);

vector<pair<int, int>> find_keypoint_matches(const vector<Keypoint>& a,
                                                       const vector<Keypoint>& b,
                                                       double thresh_relative,
                                                       double thresh_absolute, string mode);

Image draw_keypoints(const Image& img, const vector<Keypoint>& kps);

Image draw_matches(const Image& a, const Image& b, vector<Keypoint>& kps_a,
                   vector<Keypoint>& kps_b, vector<pair<int, int>> matches);
double crossCorrelation(const array<uint8_t, 128>& descriptor1, const array<uint8_t, 128>& descriptor2);

void draw_rectangle(Image& img, int x, int y, int width, int height, const array<uint8_t, 3>& color);

Image draw_matches_rect(const Image& a, const Image& b, vector<Keypoint>& kps_a,
                   vector<Keypoint>& kps_b, vector<pair<int, int>> matches);

double ssd(const array<uint8_t, 128>& descriptor1, const array<uint8_t, 128>& descriptor2);


} // namespace sift
#endif
