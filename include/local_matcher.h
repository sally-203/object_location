#ifndef LOCAL_MATCHER_H
#define LOCAL_MATCHER_H

#include <pcl/common/transforms.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/rsd.h>
#include <pcl/features/shot.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/usc.h>
#include <pcl/features/vfh.h>
#include <pcl/features/pfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
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
#include <chrono>
#include <iostream>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

using PXYZ = pcl::PointXYZ;
using PXYZS = pcl::PointCloud<PXYZ>;
using PRGB = pcl::PointXYZRGB;
using PRGBS = pcl::PointCloud<PRGB>;
using PN = pcl::Normal;
using PNS = pcl::PointCloud<PN>;

namespace matcher {

class LocalMatcher {
 public:
  //   PointCloudMatcher();
  //   ~PointCloudMatcher();

  void SetSceneCloud(const PXYZS::Ptr scene);
  void SetModelCloud(const PXYZS::Ptr model);

  void ExtractISSKeypoints(bool flag, int salient_radius = 6,
                           int nonmax_radius = 4, int min_neighbors = 5,
                           double threshold21 = 0.975,
                           double threshold32 = 0.975, int num_threads = 4);

  void ExtractDownSamplingKeypoints(bool flag, double radius = 5);

  // fpfh_radius: FPFH 特征搜索球半径，必须大于normal_radius
  // normal_radius: 法向量搜索半径
  // distance_thre: 搜索correpondence阈值，不同描述符的阈值不一样
  void FPFHMatch(double fpfh_radius = 0.01, double normal_radius = 0.01,
                 double distance_thre = 2000, int randomness = 3,
                 double inlier_fraction = 0.01, int num_samples = 3,
                 double similiar_thre = 0.4, double corres_distance = 1.0,
                 int nr_iterations = 20000);

  void RSDMatch(double rsd_radius = 0.02, double plane_radius = 0.05,
                double normal_radius = 0.02, double distance_thre = 1e-6,
                int randomness = 3, double inlier_fraction = 0.01,
                int num_samples = 3, double similiar_thre = 0.4, 
                double corres_distance = 1.0, int nr_iterations = 20000);

  void DSC3Match(double dsc_radius = 0.02, double minimal_radius = 0.01,
                 double point_density_raidus = 0.03,
                 double normal_radius = 0.015, double distance_thre = 12000,
                 int randomness = 3, double inlier_fraction = 0.01,
                 int num_samples = 3, double similiar_thre = 0.4,
                 double corres_distance = 1.0, int nr_iterations = 20000);

  void USCMatch(double usc_radius = 0.02, double minimal_radius = 0.01,
                double point_density_raidus = 0.055, double local_radius = 0.02,
                double distance_thre = 5000, int randomness = 3,
                double inlier_fraction = 0.01, int num_samples = 3,
                double similiar_thre = 0.4, double corres_distance = 1.0,
                int nr_iterations = 20000);

  void SHOTMatch(double shot_radius = 0.02, double normal_radius = 0.02,
                 double distance_thre = 1, int randomness = 3,
                 double inlier_fraction = 0.01, int num_samples = 3,
                 double similiar_thre = 0.4, double corres_distance = 1.0,
                 int nr_iterations = 20000);

  void SIMatch(double si_radius = 0.02, int image_width = 2,
               double normal_radius = 0.02, double distance_thre = 0.1,
               int randomness = 3, double inlier_fraction = 0.01,
               int num_samples = 3, double similiar_thre = 0.4,
               double corres_distance = 1.0, int nr_iterations = 20000);

  void ROPSMatch(double rops_radius = 0.02, int num_partions_bins = 5,
                 int num_rotations = 3, double support_radius = 0.04,
                 double normal_radius = 0.02, double distance_thre = 0.1,
                 int randomness = 3, double inlier_fraction = 0.01,
                 int num_samples = 3, double similiar_thre = 0.4,
                 double corres_distance = 1.0, int nr_iterations = 20000);

  void PFHMatch(double pfh_radius = 0.025, double normal_radius = 0.02,
                double distance_thre = 600, int randomness = 3,
                double inlier_fraction = 0.01, int num_samples = 3,
                double similiar_thre = 0.4, double corres_distance = 1.0,
                int nr_iterations = 20000);
  
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
      pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors,
      double pfh_radius, double normal_radius);

  void CalculateRopsDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::Histogram<135>>::Ptr descriptors, double rops_radius,
      int num_partions_bins, int num_rotations, double support_radius,
      double normal_radius);

  void CalculateFpfhDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors,
      double fpfh_radius, double normal_radius);

  void CalculateRsdDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr descriptors,
      double rsd_radius, double plane_radius, double normal_radius);

  void CalculateDscDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::ShapeContext1980>::Ptr descriptors,
      double dsc_radius, double minimal_radius, double point_density_raidus,
      double normal_radius);

  void CalculateUscDescri(
      const PXYZS::Ptr cloud,
      pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr descriptors,
      double usc_radius, double minimal_radius, double point_density_raidus,
      double local_radius);

  void CalculateShotDescri(const PXYZS::Ptr cloud,
                           pcl::PointCloud<pcl::SHOT352>::Ptr descriptors,
                           double shot_radius, double normal_radius);

  void CalculateSiDescri(const PXYZS::Ptr cloud,
                         pcl::PointCloud<pcl::Histogram<153>>::Ptr descriptors,
                         double si_radius, int image_width,
                         double normal_radius);

  void EstimateNormals(const PXYZS::Ptr cloud,
                       pcl::search::KdTree<PXYZ>::Ptr kdtree,
                       const PNS::Ptr normals, double radius);

  void EstimateNormalsByK(const PXYZS::Ptr cloud, const PNS::Ptr normals,
                          int k);

  double ComputeCloudResolution(const PXYZS::ConstPtr& cloud);

  void ShowKeypoints(PXYZS::Ptr keypoints, PXYZS::Ptr cloud);

  void Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
                 const std::string cloudName);

  void VisualizeCorrs();

  void VisualizeNormals(const PXYZS::Ptr cloud, const PNS::Ptr normals);
};

}  // namespace matcher
#endif