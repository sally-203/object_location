#include "local_matcher.h"

int main() {
  PXYZS::Ptr scene(new PXYZS);
  PXYZS::Ptr model(new PXYZS);
  // m
  pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/2.ply",
                       *scene);
  // mm
  pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/S_model.ply",
                       *model);

  //==================Local Descriptions Matching==================
  matcher::LocalMatcher localMatch;

  // 1. set scene cloud and model cloud
  localMatch.SetSceneCloud(scene);
  localMatch.SetModelCloud(model);

  // 2. extract keypoints
  // localMatch.ExtractISSKeypoints(0);
  // localMatch.ExtractISSKeypoints(1);
  localMatch.ExtractDownSamplingKeypoints(1, 0.01);
  localMatch.ExtractDownSamplingKeypoints(0, 0.01);

  // 3. match
  auto start_time = std::chrono::high_resolution_clock::now();
  // localMatch.PFHMatch();  // OK

  localMatch.FPFHMatch();

  // localMatch.RSDMatch(); // OK

  // localMatch.DSC3Match();

  // localMatch.USCMatch(); // OK

  // localMatch.SHOTMatch(); // OK

  // localMatch.SIMatch(); // OK

  // localMatch.ROPSMatch();  // OK
  auto end_time = std::chrono::high_resolution_clock::now();

  // 4. correspondences grouping
  // localMatch.CorresGrouping(0.15);

  // 5. estimate speed and accuracy
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
  std::cout << "Matching Time taken: " << duration.count() << " ms" << std::endl;

  localMatch.AccuracyEstimate();

  return 0;
}