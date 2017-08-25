#include "DBSCAN.hpp"

// In this file, x is alone the rows and y the columns;

const int THRESHOLD = 400;
const double ALPHA = 0.62;
const double BETA = 0.38;

void DBscan::init_paras(Mat& img)
{
    labels = Mat(img.size(), CV_16UC1, Scalar(65535));
    //centers = Mat(((img.rows-step/2-1)/step+1) * ((img.cols-step/2-1)/step+1),
    //             6, CV_32FC1, Scalar(0));

    Center center;
    int k = 0;
    for (int r = step/2; r < img.rows; r += step)
        for (int c = step/2; c < img.cols; c += step)
        {
            center.label = k;
            CvPoint loc_ctr = find_local_minimum(img, cvPoint(r, c));
            center.x = (float)loc_ctr.x;
            center.y = (float)loc_ctr.y;
            center.b = (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y];
            center.g = (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y + 1];
            center.r = (float)img.ptr<uchar>(loc_ctr.x)[3*loc_ctr.y + 2];
            center.count = 1;
            centers.push_back(center);
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
    double dist_xy = pow(centers[label_a].x - centers[label_b].x, 2)
        + pow(centers[label_a].y - centers[label_b].y, 2);
    double dist_rgb = pow(centers[label_a].b - centers[label_b].b, 2)
        + pow(centers[label_a].g - centers[label_b].g, 2)
        + pow(centers[label_a].r - centers[label_b].r, 2);
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
        if(withInBound(temp.x, temp.y, img))
        {
            if(labels.ptr<ushort>(temp.x)[temp.y] == 65535)
            {
                double dist = ALPHA * cmp_pix_dist(img, center, temp)
                    + BETA * cmp_pix_dist(img, point, temp);
                if(dist < THRESHOLD)
                {
                    neighbors.push_back(temp);
                    labels.ptr<ushort>(temp.x)[temp.y] = label;
                    centers[label].x += temp.x;
                    centers[label].y += temp.y;
                    centers[label].b +=
                        img.ptr<uchar>(temp.x)[3 * temp.y];
                    centers[label].g +=
                        img.ptr<uchar>(temp.x)[3 * temp.y + 1];
                    centers[label].r +=
                        img.ptr<uchar>(temp.x)[3 * temp.y + 2];
                }
            }
            else if(labels.ptr<ushort>(temp.x)[temp.y] != label)
            {
                ushort ngb_label = labels.ptr<ushort>(temp.x)[temp.y];
                ngb_matrix[label].insert(ngb_label);
                ngb_matrix[ngb_label].insert(label);
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
        if(withInBound(temp.x, temp.y, img))
        {
            if(labels.ptr<ushort>(temp.x)[temp.y] == 65535)
            {
                neighbors.push_back(temp);
                labels.ptr<ushort>(temp.x)[temp.y] = label;
                centers[label].x += temp.x;
                centers[label].y += temp.y;
                centers[label].b +=
                    img.ptr<uchar>(temp.x)[3 * temp.y];
                centers[label].g +=
                    img.ptr<uchar>(temp.x)[3 * temp.y + 1];
                centers[label].r +=
                    img.ptr<uchar>(temp.x)[3 * temp.y + 2];

            }
            else if(labels.ptr<ushort>(temp.x)[temp.y] != label)
            {
                ushort ngb_label = labels.ptr<ushort>(temp.x)[temp.y];
                ngb_matrix[label].insert(ngb_label);
                ngb_matrix[ngb_label].insert(label);
            }
        }
    }
}

void DBscan::cluster_stage(Mat &img, int step)
{
    this -> step = step;
    init_paras(img);

    // ushort label = 0;

    const int up_lims = 4.0 * step * step;
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    // cout << "here" << endl;

    for(int i=0; i < centers.size(); i ++)
    {
        CvPoint center = cvPoint(centers[i].x, centers[i].y);

        // cout << "i: " << i << endl;

        if(labels.ptr<ushort>(center.x)[center.y] != 65535)
            continue;

        labels.ptr<ushort>(center.x)[center.y] = i;
        vector<CvPoint> neighbors;
        neighbors.push_back(center);
        int count = 1;

        for(int m =0; m < count && count < up_lims; m++)
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

        centers[i].count = neighbors.size();
        centers[i].x /= centers[i].count;
        centers[i].y /= centers[i].count;
        centers[i].b /= centers[i].count;
        centers[i].g /= centers[i].count;
        centers[i].r /= centers[i].count;
        centers[i].pixels = neighbors;

        // cout << "label " << i << endl;
        // cout << "pixels: " << centers.ptr<float>(i)[5] << endl;
        // cout << "x: " << centers.ptr<float>(i)[0] << endl;
        // cout << "y: " << centers.ptr<float>(i)[1] << endl;
        // cout << "r: " << centers.ptr<float>(i)[2] << endl;
        // cout << "g: " << centers.ptr<float>(i)[3] << endl;
        // cout << "b: " << centers.ptr<float>(i)[4] << endl;
        // cout << neighbors.size() << endl;
        /*
        cout << i << "\t" << centers[i].x <<
            "\t" << centers[i].y <<
            "\t" << centers[i].b <<
            "\t" << centers[i].g <<
            "\t" << centers[i].r <<
            "\t" << centers[i].count << endl;
        */
    }
    // cout << "cluster finished" << endl;

}

void DBscan::refine_stage(Mat &img)
{
    int label = centers.size();
    for(int r = 0; r < img.rows; r++ )
        for(int c = 0; c < img.cols; c++ )
        {
            if(labels.ptr<ushort>(r)[c] != 65535)
                continue;

            CvPoint seed = cvPoint(r, c);
            labels.ptr<ushort>(r)[c] = label;
            Center center;
            center.label = label;
            center.x = r;
            center.y = c;
            center.b = img.ptr<uchar>(r)[3*c];
            center.g = img.ptr<uchar>(r)[3*c + 1];
            center.r = img.ptr<uchar>(r)[3*c + 2];
            centers.push_back(center);
            vector<CvPoint> neighbors;
            neighbors.push_back(seed);
            int count = 1;

            for(int m = 0; m < count; m ++ )
            {
                add_unlabeled(img, neighbors, neighbors[m], label);
                count = neighbors.size();
            }

            centers[label].count = neighbors.size();
            centers[label].x /= centers[label].count;
            centers[label].y /= centers[label].count;
            centers[label].b /= centers[label].count;
            centers[label].g /= centers[label].count;
            centers[label].r /= centers[label].count;
            centers[label].pixels = neighbors;

            // cout << label << "\t" << centers[label].x <<
            //     "\t" << centers[label].y <<
            //     "\t" << centers[label].b <<
            //     "\t" << centers[label].g <<
            //     "\t" << centers[label].r <<
            //     "\t" << centers[label].count << endl;

            label ++;
        }


    // cout <<" 3"<< endl;
    // cout << "lable: " << label << endl;
    /*
    int unlabeled = 0;
    for(int r = 0; r < img.rows; r ++)
        for(int c = 0; c < img.cols; c++ )
        {
            if(labels.ptr<ushort>(r)[c] == 65535)
                unlabeled ++;
        }
    */
    // cout << "unlabeled: " << unlabeled << endl;
    /*
    for(map<ushort, set<ushort> >::const_iterator map_it = ngb_matrix.begin();
            map_it != ngb_matrix.end(); map_it++)
    {
        cout << map_it -> first << " has "
            << map_it -> second.size() << "neighbors." << endl;
        for(set<ushort>::const_iterator set_it = map_it -> second.begin();
                set_it != map_it->second.end(); set_it++)
        {
            if(*set_it == 65535)
            {
                cout << map_it -> first << " has invalid neighbors" <<  endl;
            }
            else
                cout << "all valid neighbors." << endl;
        }
    }
    */
}

void DBscan::merge_stage(Mat &img)
{
    const int under_lims = step * step / 2;
    for(map<ushort, set<ushort> >::const_iterator map_it = ngb_matrix.begin();
            map_it != ngb_matrix.end(); map_it++)
    {
        int label_a = map_it -> first;
        if(centers[label_a].count < under_lims)
        {
            // find nearest neighbors
            double min_dist = FLT_MAX;
            int near_label = 65535;
            for(set<ushort>::const_iterator set_it = map_it -> second.begin();
                    set_it != map_it->second.end(); set_it++)
            {
                int label_b = *set_it;
                double dist = cmp_lb_dist(label_a, label_b);

                if(dist < min_dist)
                {
                    near_label = label_b;
                    min_dist = dist;
                }
            }
            for(vector<CvPoint>::iterator pt_it = centers[near_label].pixels.begin();
                    pt_it != centers[near_label].pixels.end(); pt_it++)
            {
                labels.ptr<ushort>(pt_it->x)[pt_it->y] = centers[label_a].label;
            }
            centers[label_a].x =
                (centers[label_a].x * centers[label_a].count +
                 centers[near_label].x * centers[near_label].count) /
                (centers[label_a].count + centers[near_label].count);
            centers[label_a].y =
                (centers[label_a].y * centers[label_a].count +
                 centers[near_label].y * centers[near_label].count) /
                (centers[label_a].count + centers[near_label].count);
            centers[label_a].b =
                (centers[label_a].b * centers[label_a].count +
                 centers[near_label].b * centers[near_label].count) /
                (centers[label_a].count + centers[near_label].count);
            centers[label_a].g =
                (centers[label_a].g * centers[label_a].count +
                 centers[near_label].g * centers[near_label].count) /
                (centers[label_a].count + centers[near_label].count);
            centers[label_a].r =
                (centers[label_a].r * centers[label_a].count +
                 centers[near_label].r * centers[near_label].count) /
                (centers[label_a].count + centers[near_label].count);
            centers[label_a].count = centers[label_a].count +
                centers[near_label].count;
            centers[label_a].pixels.insert(centers[label_a].pixels.end(),
                    centers[near_label].pixels.begin(),
                    centers[near_label].pixels.end());
            centers[near_label] = centers[label_a];
        }
    }
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

