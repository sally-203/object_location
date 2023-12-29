#include "global_matcher.h"

int main() {
  PXYZS::Ptr scene(new PXYZS);
  PXYZS::Ptr model(new PXYZS);
  // m
  pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/5.ply",
                       *scene);
  // m
  pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/1.ply",
                       *model);

  //==================Global Descriptions Matching==================
  matcher::GlobalMatcher globalMatcher;
  
  globalMatcher.SetSceneCloud(scene);
  globalMatcher.SetModelCloud(model);

  globalMatcher.ClusterScenePointCloud(scene);
  globalMatcher.ClusterModelPointCloud(model);

  // globalMatcher.VFHMatch();
  // globalMatcher.CVFHMatch(); // multi descris
  // globalMatcher.OUR_CVFH_Match(); // multi descris
  // globalMatcher.ESFMatch();
  // globalMatcher.GFPFHMatch();
  globalMatcher.GRSDMatch();

}