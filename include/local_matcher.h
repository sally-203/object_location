 ///
 ///@file local_matcher.h
 ///@author liheng (xlh18801265253@sina.com)
 ///@brief Local Matching Descriptors of Point Cloud
 ///@version 0.1
 ///@date 2024-01-02
 ///
 ///@copyright Copyright (c) 2024
 ///
 ///
#ifndef LOCAL_MATCHRE_H
#define LOCAL_MATCHER_H

#include <chrono>
#include <iostream>
#include <local_parameters.h>
#include <pcl/common/transforms.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/rsd.h>
#include <pcl/features/shot.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/usc.h>
#include <pcl/features/vfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>

using PXYZ = pcl::PointXYZ;
using PXYZS = pcl::PointCloud<PXYZ>;
using PRGB = pcl::PointXYZRGB;
using PRGBS = pcl::PointCloud<PRGB>;
using PN = pcl::Normal;
using PNS = pcl::PointCloud<PN>;

namespace matcher {

class LocalMatcher {
public:
    LocalMatcher() = default;
    ~LocalMatcher() = default;

    void SetSceneCloud(const PXYZS::Ptr scene);
    void SetModelCloud(const PXYZS::Ptr model);

    void ExtractISSKeypoints(bool flag, const IssParameters& iss_param);

    void ExtractDownSamplingKeypoints(bool flag, double radius = 5);

    void PFHMatch(const PfhParameters& pfh_param);

    void FPFHMatch(const FpfhParameters& fpfh_param);

    void RSDMatch(const RsdParameters& rsd_param);

    void DSC3Match(const Dsc3Parameters& dsc3_param);

    void USCMatch(const UscParameters& usc_param);

    void SHOTMatch(const ShotParameters& shot_param);

    void SIMatch(const SpinParameters& spin_param);

    void ROPSMatch(const RopsParameters& rops_param);

    void AccuracyEstimate();

    void CorresGrouping(double gc_size = 0.01);

private:
    PXYZS::Ptr scene_cloud_;
    PXYZS::Ptr model_cloud_;

    PXYZS::Ptr scene_keypoints_;
    PXYZS::Ptr model_keypoints_;

    pcl::CorrespondencesPtr correspondences_;
    Eigen::Matrix4f transformations_;
    double distance_ = 0;

    void CalculatePfhDescri(
        const PXYZS::Ptr cloud,
        double pfh_radius,
        const pcl::PointCloud<pcl::PFHSignature125>::Ptr& descriptors);

    void CalculateFpfhDescri(
        const PXYZS::Ptr cloud,
        double fpfh_radius,
        const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& descriptors);

    void CalculateRsdDescri(
        const PXYZS::Ptr cloud,
        double rsd_radius, double plane_radius,
        const pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr& descriptors);

    void CalculateDscDescri(
        const PXYZS::Ptr cloud,
        double dsc_radius, double minimal_radius,
        double point_density_raidus,
        const pcl::PointCloud<pcl::ShapeContext1980>::Ptr& descriptors);

    void CalculateUscDescri(
        const PXYZS::Ptr cloud,
        double usc_radius, double minimal_radius,
        double point_density_raidus, double local_radius,
        const pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr& descriptors);

    void CalculateShotDescri(const PXYZS::Ptr cloud,
        double shot_radius,
        const pcl::PointCloud<pcl::SHOT352>::Ptr& descriptors);

    void CalculateSiDescri(const PXYZS::Ptr cloud,
        double si_radius, int image_width,
        double normal_radius,
        const pcl::PointCloud<pcl::Histogram<153>>::Ptr& descriptors);

    void CalculateRopsDescri(
        const PXYZS::Ptr cloud, double rops_radius,
        int num_partions_bins, int num_rotations,
        double support_radius, double normal_radius,
        const pcl::PointCloud<pcl::Histogram<135>>::Ptr& descriptors);

    void EstimateNormals(const PXYZS::Ptr cloud,
        pcl::search::KdTree<PXYZ>::Ptr kdtree,
        const PNS::Ptr normals, double radius);

    void EstimateNormalsByK(const PXYZS::Ptr cloud, const PNS::Ptr normals,
        int k);

    double ComputeCloudResolution(const PXYZS::ConstPtr& cloud);

    void print();

    void ShowKeypoints(PXYZS::Ptr keypoints, PXYZS::Ptr cloud);

    void Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
        const std::string cloudName);

    void VisualizeCorrs();

    void VisualizeNormals(const PXYZS::Ptr cloud, const PNS::Ptr normals);
};

} // namespace matcher
#endif