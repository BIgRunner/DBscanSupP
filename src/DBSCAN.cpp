#include "DBSCAN.hpp"

// In this file, x is alone the rows and y the columns;

const int THRESHOLD = 400;
const double ALPHA = 0.62;
const double BETA = 0.38;

void DBscan::init_paras(Mat& img)
{
    labels = Mat(img.size(), CV_16UC1, Scalar(65535));
    centers = Mat(((img.rows-step/2-1)/step+1) * ((img.cols-step/2-1)/step+1),
                6, CV_32FC1, Scalar(0));

    int k = 0;
    for (int r = step/2; r < img.rows; r += step)
        for (int c = step/2; c < img.cols; c += step)
        {
            CvPoint loc_ctr = find_local_minimum(img, cvPoint(r, c));
            centers.ptr<float>(k)[0] = (float)loc_ctr.x;
            centers.ptr<float>(k)[1] = (float)loc_ctr.y;
            centers.ptr<float>(k)[2] =
                (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y];
            centers.ptr<float>(k)[3] =
                (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y + 1];
            centers.ptr<float>(k)[4] =
                (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y + 2];
            centers.ptr<float>(k)[5] = 0.0;
            k++;
        }
    // cout << "initialization finished" << endl;
}

double DBscan::cmp_pix_dist(Mat &img, CvPoint &p1, CvPoint &p2)
{
    double d =
        pow(img.ptr<uchar>(p1.x)[3 * p1.y]
                - img.ptr<uchar>(p2.x)[3 * p2.y], 2)
        + pow(img.ptr<uchar>(p1.x)[3 * p1.y + 1]
                - img.ptr<uchar>(p2.x)[3 * p2.y + 1], 2)
        + pow(img.ptr<uchar>(p1.x)[3 * p1.y + 2]
                - img.ptr<uchar>(p2.x)[3 * p2.y + 2], 2);
    return d;
}

double DBscan::cmp_lb_dist(int label_a, int label_b)
{
    double dist_xy =
        pow(centers.ptr<float>(label_a)[0]
                - centers.ptr<float>(label_b)[0], 2)
        + pow(centers.ptr<float>(label_a)[0]
                - centers.ptr<float>(label_b)[1], 2);
    double dist_rgb =
        pow(centers.ptr<float>(label_a)[2]
                - centers.ptr<float>(label_b)[2], 2)
        + pow(centers.ptr<float>(label_a)[3]
                - centers.ptr<float>(label_b)[3], 2)
        + pow(centers.ptr<float>(label_a)[4]
                - centers.ptr<float>(label_b)[4], 2);
    return dist_rgb + 2*dist_xy;
}

bool withInBound(int x, int y, Mat& img)
{
    return x >= 0 && x < img.rows && y >= 0 && y < img.cols;
}

CvPoint DBscan::find_local_minimum(Mat &img, CvPoint center)
{
    double min_grad = FLT_MAX;
    CvPoint loc_min = cvPoint(center.x, center.y);

    for(int r = center.x-1; r < center.x+2; r++)
        for(int c = center.y-1; c < center.y+2; c++)
        {
            if(withInBound(r, c+1, img) && withInBound(r+1, c, img))
            {
                double l1 = img.ptr<uchar>(r+1)[3*c];
                double l2 = img.ptr<uchar>(r)[3*(c+1)];
                double l3 = img.ptr<uchar>(r)[3*c];

                double grad = pow(l1-l3, 2) + pow(l2-l3, 2);

                if(grad < min_grad)
                {
                    min_grad = grad;
                    loc_min.x = r;
                    loc_min.y = c;
                }
            }
        }

    return loc_min;
}

void DBscan::add_neighbors(Mat &img, vector<CvPoint> &neighbors,
        CvPoint &center, CvPoint &point, int label)
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};
    for(int i = 0; i < 4; i ++)
    {
        CvPoint temp = cvPoint(point.x + dx4[i], point.y + dy4[i]);
        if(withInBound(temp.x, temp.y, img) &&
                labels.ptr<ushort>(temp.x)[temp.y] == 65535)
        {
            double dist = ALPHA * cmp_pix_dist(img, center, temp)
                + BETA * cmp_pix_dist(img, point, temp);
            if(dist < THRESHOLD)
            {
                neighbors.push_back(temp);
                labels.ptr<ushort>(temp.x)[temp.y] = label;
                centers.ptr<float>(label)[0] += temp.x;
                centers.ptr<float>(label)[1] += temp.y;
                centers.ptr<float>(label)[2] +=
                    img.ptr<uchar>(temp.x)[3 * temp.y];
                centers.ptr<float>(label)[3] +=
                    img.ptr<uchar>(temp.x)[3 * temp.y + 1];
                centers.ptr<float>(label)[4] +=
                    img.ptr<uchar>(temp.x)[3 * temp.y + 2];
            }
        }
    }
}

void DBscan::add_unlabeled(Mat &img, vector<CvPoint> &neighbors,
        CvPoint &point, int label)
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};
    for(int i = 0; i < 4; i++)
    {
        CvPoint temp = cvPoint(point.x + dx4[i], point.y + dy4[i]);
        if(withInBound(temp.x, temp.y, img) &&
                labels.ptr<ushort>(temp.x)[temp.y] == 65535)
        {
            neighbors.push_back(temp);
            labels.ptr<ushort>(temp.x)[temp.y] = label;
        }
    }
}

void DBscan::cluster_stage(Mat &img, int step)
{
    this -> step = step;
    init_paras(img);

    // ushort label = 0;

    const int lims = (img.rows * img.cols)/((int)centers.rows);
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    // cout << "here" << endl;

    for(int i=0; i < centers.rows; i ++)
    {
        CvPoint center = cvPoint(centers.ptr<float>(i)[0],
                centers.ptr<float>(i)[1]);

        // cout << "i: " << i << endl;

        if(labels.ptr<ushort>(center.x)[center.y] != 65535)
            continue;

        labels.ptr<ushort>(center.x)[center.y] = i;
        vector<CvPoint> neighbors;
        neighbors.push_back(center);
        int count = 1;

        for(int m =0; m < count && count < 4*lims; m++)
        {
            // cout << neighbors[m].x << endl;
            // cout << neighbors[m].y << endl;
            // // cout << neighbors[m].x << endl;
            add_neighbors(img, neighbors, center, neighbors[m], i);
            // cout<<"m: " << m << endl;
            count = neighbors.size();
            // cout << "count: " << count << endl;
        }
        // cout << "his" <<endl;

        centers.ptr<float>(i)[5] = neighbors.size();
        centers.ptr<float>(i)[0] /= centers.ptr<float>(i)[5];
        centers.ptr<float>(i)[1] /= centers.ptr<float>(i)[5];
        centers.ptr<float>(i)[2] /= centers.ptr<float>(i)[5];
        centers.ptr<float>(i)[3] /= centers.ptr<float>(i)[5];
        centers.ptr<float>(i)[4] /= centers.ptr<float>(i)[5];

        // cout << "label " << i << endl;
        // cout << "pixels: " << centers.ptr<float>(i)[5] << endl;
        // cout << "x: " << centers.ptr<float>(i)[0] << endl;
        // cout << "y: " << centers.ptr<float>(i)[1] << endl;
        // cout << "r: " << centers.ptr<float>(i)[2] << endl;
        // cout << "g: " << centers.ptr<float>(i)[3] << endl;
        // cout << "b: " << centers.ptr<float>(i)[4] << endl;
        cout << i << "\t" << centers.ptr<float>(i)[0] <<
            "\t" << centers.ptr<float>(i)[1] <<
            "\t" << centers.ptr<float>(i)[2] <<
            "\t" << centers.ptr<float>(i)[3] <<
            "\t" << centers.ptr<float>(i)[4] <<
            "\t" << centers.ptr<float>(i)[5] << endl;
        // cout <<" 3"<< endl;
        // cout << neighbors.size() << endl;
    }
    // cout << "cluster finished" << endl;

}

void DBscan::display_contours(Mat &img, CvScalar color)
{
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    vector<CvPoint> contours;
    Mat istaken = Mat(img.size(), CV_8UC1, Scalar(0));

    for (int r=0; r<img.rows; r++)
        for (int c=0; c<img.cols; c++)
        {
            int nr_p = 0;

            for(int k = 0; k < 8; k++)
            {
                int x = r + dx8[k];
                int y = c + dy8[k];

                if ( withInBound(x, y, img) )
                {
                    if (istaken.ptr<uchar>(x)[y] == 0 &&
                            labels.ptr<ushort>(x)[y]!=labels.ptr<ushort>(r)[c])
                        nr_p ++;
                }
            }

            if(nr_p >= 2)
            {
                contours.push_back(cvPoint(r,c));
                istaken.ptr<uchar>(r)[c] = 255;
            }
        }

    for (vector<CvPoint>::iterator iter = contours.begin();
            iter != contours.end(); iter ++)
    {
        // cvGet2D(img, iter->x, iter->y, color);
        img.ptr<uchar>(iter->x)[3 * iter->y] = color.val[0];
        img.ptr<uchar>(iter->x)[3 * iter->y + 1] = color.val[1];
        img.ptr<uchar>(iter->x)[3 * iter->y + 2] = color.val[2];

    }
}

