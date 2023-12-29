#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Eigen>

using PRGB = pcl::PointXYZRGB;
using PRGBS = pcl::PointCloud<PRGB>;

using EXTRACT_KEYPOINT_PTR = PRGBS::Ptr (*)(PRGBS::Ptr);
using FEATURE_MATCHING_PTR = Eigen::Matrix4f (*)(PRGBS::Ptr, PRGBS::Ptr);

PRGBS::Ptr extract_1(PRGBS::Ptr pc) {
  // extract keypoint
}

PRGBS::Ptr extract_2(PRGBS::Ptr pc) {
  // extract keypoint
}

Eigen::Matrix4f matching_1(PRGBS::Ptr pc1, PRGBS::Ptr pc2) {
  // compute descriptors
  // matching with descriptors
}

Eigen::Matrix4f mathcing_2(PRGBS::Ptr pc1, PRGBS::Ptr pc2) {
  // compute descriptors
  // matching with descriptors
}

class Pipeline {
  EXTRACT_KEYPOINT_PTR extract_func;
  FEATURE_MATCHING_PTR matching_func;

 public:
  Pipeline(EXTRACT_KEYPOINT_PTR func1, FEATURE_MATCHING_PTR func2)
      : extract_func(func1), matching_func(func2) {}
};

Pipeline regis(extract_1, matching_1);
