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
  // IssParameters iss_param;
  // localMatch.ExtractISSKeypoints(0, iss_param);
  // localMatch.ExtractISSKeypoints(1, iss_param);
  localMatch.ExtractDownSamplingKeypoints(1, 0.01);
  localMatch.ExtractDownSamplingKeypoints(0, 0.01);

  // 3. match
  auto start_time = std::chrono::high_resolution_clock::now();

  // PfhParameters pfh_param;
  // localMatch.PFHMatch(pfh_param);  // OK

  // FpfhParameters fpfh_param;
  // localMatch.FPFHMatch(fpfh_param);

  // RsdParameters rsd_param;
  // localMatch.RSDMatch(rsd_param); // OK

  // Dsc3Parameters dsc3_param;
  // localMatch.DSC3Match(dsc3_param);

  // UscParameters usc_param;
  // localMatch.USCMatch(usc_param); // OK

  // ShotParameters shot_param;
  // localMatch.SHOTMatch(shot_param); // OK

  // SpinParameters spin_param;
  // localMatch.SIMatch(spin_param); // OK

  RopsParameters rops_param;
  localMatch.ROPSMatch(rops_param);  // OK

  auto end_time = std::chrono::high_resolution_clock::now();

  // 4. correspondences grouping
  // localMatch.CorresGrouping(0.15);

  // 5. estimate speed and accuracy
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
  std::cout << "Matching Time taken: " << duration.count() << " ms" << std::endl;

  localMatch.AccuracyEstimate();

  return 0;
}