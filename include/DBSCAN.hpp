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

struct Center
{
    ushort label;
    double x, y;
    double r, g, b;
    int count;
    vector<CvPoint> pixels;
};

class DBscan
{
    private:
        // labels for each pixel;
        Mat labels;

        // seeds for segmentation
        // Mat centers;
        vector<Center> centers;

        int  step;

        map<ushort, set<ushort> > ngb_matrix;

        // thresthold for segmentation
        double thresthold;

        double cmp_pix_dist(Mat &img, CvPoint &p1, CvPoint &p2);

        double cmp_lb_dist(int label_a, int label_b);

        void init_paras(Mat &img);

        CvPoint find_local_minimum(Mat &img, CvPoint center);

        void add_neighbors(Mat &img, vector<CvPoint> &neighbors,
                CvPoint &center, CvPoint &point, int label);

        void add_unlabeled(Mat &img, vector<CvPoint> &neighbors,
                CvPoint &point, int label);

    public:

        void cluster_stage(Mat &image, int step);

        void refine_stage(Mat &image, int step);

        void merge_stage(Mat &img);

        void display_contours(Mat &img, CvScalar);
};

#endif
