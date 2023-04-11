#include "harris.h"
#include <cmath>

// Main harris corner detecting function
// Takes original image as input, the k parameter and threshold
// Returns the original image with detected corners

Mat harrisCornerDetector(Mat& original, double k, double thr){
    Mat gray, output;
    output = original;
    cvtColor(original, gray, COLOR_BGR2GRAY);

    // Calculate Image gradients
    Mat Ixx, Iyy, Ixy;
    getGradients(gray, Ixx, Iyy, Ixy);

    // Calculate Harris Response
    Mat det, tr, response;
    det = Ixx.mul(Iyy) -  Ixy.mul(Ixy);
    pow((Ixx + Iyy), 2.0, tr);
    response = det - k * tr;

    // Spot corners on original image
    drawResponse(response, output, thr);

    return output;
}

// Calculates the sobel gradiants and applying Gaussian filter upon them.
// Takes the grayscale image and the matrices to store calculations in.

void getGradients(Mat& gray, Mat& Ixx, Mat& Iyy, Mat& Ixy){
    // Sobel Gradients
    Mat grad_x, grad_y;
    Sobel(gray, grad_x, CV_32FC1, 1, 0, 3);
    Sobel(gray, grad_y, CV_32FC1, 0, 1, 3);

    // Multiply gradients
    Mat grad_x_squared, grad_y_squared, grad_x_y;
    pow(grad_x, 2.0, grad_x_squared);
    pow(grad_y, 2.0, grad_y_squared);
    multiply(grad_x, grad_y, grad_x_y);

    // Apply Gaussian
    GaussianBlur(grad_x_squared, Ixx, Size(3,3), 2.0, 0);
    GaussianBlur(grad_y_squared, Iyy, Size(3,3), 0, 2.0);
    GaussianBlur(grad_x_y, Ixy, Size(3,3), 2.0, 2.0);
}

// Spots (Draws) corners on the image
// Takes the calculated Harris Response Matrix, the output image to draw on and the threshold

void drawResponse(Mat& response, Mat& output, double thr){
    // Normalize the response values from 0 to 255 only
    normalize(response, response, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // check the harris response for each pixel and decide either it's a corner or not
    for (int i = 0; i < output.rows; i++){
        for (int j = 0; j < output.cols; j++){
            if (response.at<float>(i, j) > thr){
                circle(output, Point(j, i), 3, Scalar(255, 0, 0), 3, 8, 0);
            }
        }
    }
}



// Code to add in mainmenu

//Mat original = img1.getOutputImg();

//double k = 0.04;
//double thr = 150;
//Mat result = harrisCornerDetector(original, k, thr);

//showImg(result, ui->outputImg, QImage::Format_RGB888, ui->outputImg->width(), ui->outputImg->height());
