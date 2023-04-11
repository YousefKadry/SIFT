#ifndef HARRIS_H
#define HARRIS_H
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat harrisCornerDetector(Mat& original, double k, double thr);
void getGradients(Mat& gray, Mat& Ixx, Mat& Iyy, Mat& Ixy);
void drawResponse(Mat& response, Mat& output, double thr);

#endif // HARRIS_H
