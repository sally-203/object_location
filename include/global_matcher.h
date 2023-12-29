#ifndef GLOBAL_MATCHER_H
#define GLOBAL_MATCHER_H

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/features/crh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/grsd.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/gfpfh.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

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
  void SetModelCloud(const PXYZS::Ptr scene);
  void SetSceneCloud(const PXYZS::Ptr model);

  void ClusterModelPointCloud(const PXYZS::Ptr cloud,
                              double cluster_tolerance = 0.02,
                              int min_cluster_size = 100,
                              int max_cluster_size = 25000);

  void ClusterScenePointCloud(const PXYZS::Ptr cloud,
                              double cluster_tolerance = 0.02,
                              int min_cluster_size = 100,
                              int max_cluster_size = 25000);

  void GFPFHMatch(double octree_leaf_size = 0.01,
                  int num_classes = 2);

  void GRSDMatch(double grsd_radius = 0.1);

  void CVFHMatch(double eps_angle = 5.0 / 180.0 * M_PI, double curv_thre = 1.0);

  void VFHMatch();

  void OUR_CVFH_Match(double eps_angle = 5.0 / 180.0 * M_PI,
                      double curv_thre = 1.0, double axis_ratio = 0.8);

  void ESFMatch();

  void ICP(const PXYZS::Ptr source_cloud, const PXYZS::Ptr target_cloud);

 private:
  PXYZS::Ptr model_cloud_;
  PXYZS::Ptr scene_cloud_;

  std::vector<pcl::PointCloud<PXYZ>::Ptr> model_clusters_;
  std::vector<pcl::PointCloud<PXYZ>::Ptr> scene_clusters_;

  pcl::CorrespondencesPtr correspondences_;
  Eigen::Matrix4f transformations_;

  void print(Eigen::Matrix4f& transformations);

  void resize(pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descriptors,
              pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptiors);

  void CalculateGfpfhDescri(
      const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
      pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptions,
      double octree_leaf_size, double num_classes);

  void CalculateEsfDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptions);

  void CalculateOurcvfhDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptions, double eps_angle,
      double curv_thre, double axis_ratio);

  void CalculateGrsdDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors,
      double grsd_radius);

  void CalculateCvfhDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors, double eps_angle,
      double curv_thre);

  void CalculateVfhDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors);

  void EstimateNormalsByK(const PXYZS::Ptr cloud, const PNS::Ptr normals,
                          int k);

  void Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
                 const std::string cloudName);

  void VisualizeCorrs();

  void visualizeClusters(
      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);
};

}  // namespace matcher
#endif