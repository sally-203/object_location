#include "local_matcher.h"
#include <pcl/io/pcd_io.h>
#include "read_config.h"

int main()
{
    ivgs_util::LocalConfig localconfig("../config/local_param.cfg");
    
    PXYZS::Ptr scene(new PXYZS);
    PXYZS::Ptr model(new PXYZS);

    // ========================= OUR DATASET (m)===========================
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/2.ply",
    //                      *scene);
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/Standard/S_model.ply",
    //                      *model);
    
    // ========================= YONG'S DATASET(m) =========================
    if (pcl::io::loadPCDFile("/home/xlh/work/others2me/yong2me/regis_dataset/pcd_parts/Hand_part/1.pcd", *scene) == -1) {
        PCL_ERROR("Couldn't read scene file \n");
        return -1;
    }

    if (pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Hand.ply", *model) == -1) {
        PCL_ERROR("Couldn't read model file \n");
        return -1;
    }

    std::cout << "Point Size of Scene: " << scene->points.size() << std::endl;
    std::cout << "Point Size of Model: " << model->points.size() << std::endl;

    //================== Local Descriptions Matching ==================
    matcher::LocalMatcher localMatch;

    std::vector<Eigen::Matrix4f> poses;
    std::string path = "/home/xlh/work/others2me/yong2me/regis_dataset/pcd_parts/Hand_part/pose.txt";
    localMatch.read_pose_txt(path, poses);

    CommonParameters commonparam = localconfig.ReadCommonParam();

    // 1. set scene cloud and model cloud
    localMatch.SetSceneCloud(scene);
    localMatch.SetModelCloud(model);

    // 2. extract keypoints
    // 2.1 ISS keypoints
    // IssParameters issparam = localconfig.ReadIssParam();
    // localMatch.ExtractISSKeypoints(0, iss_param);
    // localMatch.ExtractISSKeypoints(1, iss_param);

    // 2.2 Downsampling keypoints
    localMatch.ExtractDownSamplingKeypoints(1, 0.01);
    localMatch.ExtractDownSamplingKeypoints(0, 0.01);

    // 3. match
    // PfhParameters pfh_param = localconfig.ReadPfhParam();
    // pfh_param.common_params = commonparam;
    // auto start_pfh_time = std::chrono::high_resolution_clock::now();
    // localMatch.PFHMatch(pfh_param);
    // auto pfh_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto pfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pfh_time - start_pfh_time);
    // std::cout << "PFH Matching Time Taken: " << pfh_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // FpfhParameters fpfh_param = localconfig.ReadFpfhParam();
    // fpfh_param.common_params = commonparam;
    // auto start_fpfh_time = std::chrono::high_resolution_clock::now();
    // localMatch.FPFHMatch(fpfh_param);
    // auto fpfh_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto fpfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fpfh_time - start_fpfh_time);
    // std::cout << "FPFH Matching Time Taken: " << fpfh_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // RsdParameters rsd_param = localconfig.ReadRsdParam();
    // rsd_param.common_params = commonparam;
    // auto start_rsd_time = std::chrono::high_resolution_clock::now();
    // localMatch.RSDMatch(rsd_param);
    // auto rsd_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto rsd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rsd_time - start_rsd_time);
    // std::cout << "RSD Matching Time Taken: " << rsd_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // Dsc3Parameters dsc3_param = localconfig.ReadDsc3Param();
    // dsc3_param.common_params = commonparam;
    // auto start_dsc3_time = std::chrono::high_resolution_clock::now();
    // localMatch.DSC3Match(dsc3_param);
    // auto dsc3_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto dsc3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dsc3_time - start_dsc3_time);
    // std::cout << "DSC3 Matching Time Taken: " << dsc3_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // UscParameters usc_param = localconfig.ReadUscParam();
    // usc_param.common_params = commonparam;
    // auto start_usc_time = std::chrono::high_resolution_clock::now();
    // localMatch.USCMatch(usc_param);
    // auto usc_time = std::chrono::high_resolution_clock::now();
    // //estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto usc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(usc_time - start_usc_time);
    // std::cout << "USC Matching Time Taken: " << usc_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // ShotParameters shot_param = localconfig.ReadShotParam();
    // shot_param.common_params = commonparam;
    // auto start_shot_time = std::chrono::high_resolution_clock::now();
    // localMatch.SHOTMatch(shot_param);
    // auto shot_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto shot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shot_time - start_shot_time);
    // std::cout << "SHOT Matching Time Taken: " << shot_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // SpinParameters spin_param = localconfig.ReadSpinParam();
    // spin_param.common_params = commonparam;
    // auto start_spin_time = std::chrono::high_resolution_clock::now();
    // localMatch.SIMatch(spin_param);
    // auto spin_image_time = std::chrono::high_resolution_clock::now();
    // // estimate accuracy
    // localMatch.AccuracyEstimate();
    // localMatch.AbsolueAccuracyEstimate(poses[1]);
    // auto spin_image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spin_image_time - start_spin_time);
    // std::cout << "SPIN IAMGE Matching Time Taken: " << spin_image_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    RopsParameters rops_param = localconfig.ReadRopsParam();
    rops_param.common_params = commonparam;
    auto start_rops_time = std::chrono::high_resolution_clock::now();
    localMatch.ROPSMatch(rops_param);
    auto rops_time = std::chrono::high_resolution_clock::now();
    // estimate accuracy
    localMatch.AccuracyEstimate();
    localMatch.AbsolueAccuracyEstimate(poses[1]);
    auto rops_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rops_time - start_rops_time);
    std::cout << "ROPS Matching Time Taken: " << rops_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}