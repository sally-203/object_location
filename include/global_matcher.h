 ///
 ///@file global_matcher.h
 ///@author liheng (xlh18801265253@sina.com)
 ///@brief Global Recognizing Descriptors of Point Cloud
 ///@version 0.1
 ///@date 2024-01-02
 ///
 ///@copyright Copyright (c) 2024
 ///
 ///
#ifndef GLOBAL_MATCHER_H
#define GLOBAL_MATCHER_H

#include <global_parameters.h>
#include <iostream>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/features/crh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/gfpfh.h>
#include <pcl/features/grsd.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/vfh.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

using PXYZ = pcl::PointXYZ;
using PXYZS = pcl::PointCloud<PXYZ>;
using PRGB = pcl::PointXYZRGB;
using PRGBS = pcl::PointCloud<PRGB>;
using PN = pcl::Normal;
using PNS = pcl::PointCloud<PN>;

typedef pcl::Histogram<90> CRH90;

namespace matcher {

class GlobalMatcher {
public:
    GlobalMatcher() = default;
    ~GlobalMatcher() = default;

    void SetModelCloud(const PXYZS::Ptr scene);

    void SetSceneCloud(const PXYZS::Ptr model);

    void ClusterModelPointCloud(const PXYZS::Ptr cloud,
        const ClusterParameters& cluster_param);

    void ClusterScenePointCloud(const PXYZS::Ptr cloud,
        const ClusterParameters& cluster_param);

    void VFHMatch(const VfhParameters& vfh_params);

    void CVFHMatch(const CvfhParameters& cvfh_params);

    void OUR_CVFH_Match(const OurcvfhParameters& our_cvfh_params);

    void ESFMatch(const EsfParameters& esf_params);

    void GFPFHMatch(const GfpfhParameters& gfpfh_params);

    void GRSDMatch(const GrsdParameters& grsd_params);

    void ICP(const PXYZS::Ptr source_cloud, const PXYZS::Ptr target_cloud);

private:
    PXYZS::Ptr model_cloud_;
    PXYZS::Ptr scene_cloud_;

    std::vector<pcl::PointCloud<PXYZ>::Ptr> model_clusters_;
    std::vector<pcl::PointCloud<PXYZ>::Ptr> scene_clusters_;

    pcl::CorrespondencesPtr correspondences_;
    Eigen::Matrix4f transformations_;

    void CalculateVfhDescri(
        const PXYZS::Ptr cloud,
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors);

    void CalculateCvfhDescri(
        const PXYZS::Ptr cloud,
        double eps_angle, double curv_thre,
        const pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors);

    void CalculateOurcvfhDescri(
        const PXYZS::Ptr cloud,
        double eps_angle, double curv_thre, double axis_ratio,
        const pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptions);

    void CalculateEsfDescri(
        const PXYZS::Ptr cloud,
        const pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptions);

    void CalculateGfpfhDescri(
        const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
        double octree_leaf_size, double num_classes,
        const pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptions);

    void CalculateGrsdDescri(
        const PXYZS::Ptr cloud,
        double grsd_radius,
        const pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors);

    void EstimateNormalsByK(const PXYZS::Ptr cloud, const PNS::Ptr normals,
        int k);

    void CorrespondenceViewer(const bool& show_flag);

    void Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
        const std::string cloudName);

    void VisualizeClusters(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);

    void Resize(pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descriptors,
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptiors);
};

} // namespace matcher
#endif