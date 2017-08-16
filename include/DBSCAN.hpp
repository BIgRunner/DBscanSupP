#ifndef DBSCAN_HPP
#define DBSCAN_HPP

/* DBSCAN.hpp
 * written by BigRunner
 * This hpp file contains class DBSCAN, which is used for imge
 * segmentation,
 */

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct center
{
    int lable;
    double row, col;
    double r, g, b;
    int count;
};

class DBscan
{
    private:
        // labels for each pixel;
        Mat labels;

        // seeds for segmentation
        Mat centers;

        double step;

        // thresthold for segmentation
        double thresthold;

        double compute_dist(Mat &img, CvPoint &p1, CvPoint &p2);

        void init_paras(Mat &img);

        CvPoint find_local_minimum(Mat &img, CvPoint center);

        void add_neighbors(Mat &img, vector<CvPoint> &neighbors,
                CvPoint &center, CvPoint &point, int label);

    public:

        void cluster_stage(Mat &image, int cluster_num);

        void merge_stage(Mat &img);
};

#endif
