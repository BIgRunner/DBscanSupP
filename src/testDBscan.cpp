#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "DBSCAN.hpp"
using namespace std;

int main(int argc, char **argv)
{
    Mat image = cv::imread(argv[1], 1);
    int w = image.cols;
    int h = image.rows;
    int nr_superpixels = atoi(argv[2]);
    cout << "I have read parameters, prepare for superpixel..." << endl;

    int step = sqrt((w * h) / (double)nr_superpixels);
    cout << "my step is " << step << endl;

    DBscan dbscan;
    dbscan.cluster_stage(image, step);
    dbscan.display_contours(image, CV_RGB(255, 0, 0));
    imshow("image",image);
    cvWaitKey(0);
    return 0;
}
