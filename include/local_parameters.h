#ifndef LOCAL_PARAMETERS_H
#define LOCAL_PARAMETERS_H

struct CommonParameters {
    bool show_flag = true;
    int randomness = 3;
    double inlier_fraction = 0.01;
    int num_samples = 3;
    double similiar_thre = 0.4;
    double corres_distance = 1.0;
    int nr_iterations = 20000;
};

struct IssParameters {
    int salient_radius = 6;
    int nonmax_radius = 4;
    int min_neighbors = 5;
    double threshold21 = 0.975;
    double threshold32 = 0.975;
    int num_threads = 4;
};

// fpfh_radius: FPFH 特征搜索球半径，必须大于normal_radius
// normal_radius: 法向量搜索半径
// distance_thre: 搜索correpondence阈值，不同描述符的阈值不一样
struct PfhParameters {
    double pfh_radius = 0.025;
    double distance_thre = 600;
    CommonParameters common_params;
};

struct FpfhParameters {
    double fpfh_radius = 0.01;
    double distance_thre = 2000;
    CommonParameters common_params;
};

struct RsdParameters {
    double rsd_radius = 0.02;
    double plane_radius = 0.05;
    double distance_thre = 1e-6;
    CommonParameters common_params;
};

struct Dsc3Parameters {
    double dsc_radius = 0.02;
    double minimal_radius = 0.01;
    double point_density_raidus = 0.03;
    double normal_radius = 0.015;
    double distance_thre = 12000;
    CommonParameters common_params;
};

struct UscParameters {
    double usc_radius = 0.02;
    double minimal_radius = 0.01;
    double point_density_raidus = 0.055;
    double local_radius = 0.02;
    double distance_thre = 5000;
    CommonParameters common_params;
};

struct ShotParameters {
    double shot_radius = 0.02;
    double normal_radius = 0.02;
    double distance_thre = 1;
    CommonParameters common_params;
};

struct SpinParameters {
    double si_radius = 0.02;
    int image_width = 1;
    double normal_radius = 0.02;
    double distance_thre = 0.1;
    CommonParameters common_params;
};

struct RopsParameters {
    double rops_radius = 0.02;
    int num_partions_bins = 5;
    int num_rotations = 3;
    double support_radius = 0.04;
    double normal_radius = 0.02;
    double distance_thre = 0.1;
    CommonParameters common_params;
};

#endif