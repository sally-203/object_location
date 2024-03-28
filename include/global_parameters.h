#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include <cmath>

struct ClusterParameters {
    bool show_flag = false;
    double cluster_tolerance = 0.02;
    int min_cluster_size = 100;
    int max_cluster_size = 25000;
};

struct VfhParameters{
    bool show_flag = false;
};

struct CvfhParameters {
    bool show_flag = false;
    double eps_angle = 5.0 / 180.0 * M_PI;
    double curv_thre = 1.0;
};

struct OurcvfhParameters {
    bool show_flag = false;
    double eps_angle = 5.0 / 180.0 * M_PI;
    double curv_thre = 1.0;
    double axis_ratio = 0.8;
};

struct EsfParameters{
    bool show_flag = false;
};

struct GfpfhParameters {
    bool show_flag = false;
    double octree_leaf_size = 0.01;
    int num_classes = 2;
};

struct GrsdParameters {
    bool show_flag = false;
    double grsd_radius = 0.1;
};

#endif