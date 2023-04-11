# include "opencv2/opencv_modules.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
// # include "opencv2/nonfree/features2d.hpp"
# include "opencv2\imgproc\imgproc.hpp"
# include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat img_1, img_2;

    // Load image in grayscale format
    img_1 = imread("C:/Users/dell/Downloads/imgs/g.png", 0);

    // Rotate the input image without loosing the corners 
    Point center = Point(img_1.cols / 2, img_1.rows / 2);
    double angle = 45, scale = 1;
    Mat rot = getRotationMatrix2D(center, angle, scale);
    Rect bbox = cv::RotatedRect(center, img_1.size(), angle).boundingRect();
    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    warpAffine(img_1, img_2, rot, bbox.size());

    // SIFT feature detector
    SiftFeatureDetector detector;
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect(img_1, keypoints_1);
    detector.detect(img_2, keypoints_2);

    // Calculate descriptors 
    SiftDescriptorExtractor extractor;
    Mat descriptors_1, descriptors_2;

    extractor.compute(img_1, keypoints_1, descriptors_1);
    extractor.compute(img_2, keypoints_2, descriptors_2);

    // Matching descriptors using Brute Force
    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);


    //-- Quick calculation of max and min distances between Keypoints
    double max_dist = 0; double min_dist = 100;

    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    std::vector< DMatch > good_matches;

    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 0.02))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Show detected matches
    imshow("Good Matches", img_matches);

    waitKey(0);
    return 0;
}
