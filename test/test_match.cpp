#include "dataviewer_pointcloud.h"
#include "local_matcher.h"
#include <pcl/io/pcd_io.h>
#include "read_config.h"

int main()
{
    util::DataViewerPointCloud viewer;
    ivgs_util::LocalConfig localconfig("../config/local_param.cfg");
    
    PXYZS::Ptr scene(new PXYZS);
    PXYZS::Ptr model(new PXYZS);

    // ========================= OUR DATASET (m)===========================
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/2.ply",
    //                      *scene);
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/Standard/S_model.ply",
    //                      *model);
    
    // ========================= YONG'S DATASET(mm) =========================
    if (pcl::io::loadPCDFile("/home/xlh/work/others2me/yong2me/regis_dataset/pcd_parts/Bunny_part/1.pcd", *scene) == -1) {
        PCL_ERROR("Couldn't read scene file \n");
        return -1;
    }

    if (pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Bunny.ply", *model) == -1) {
        PCL_ERROR("Couldn't read model file \n");
        return -1;
    }
    // viewer.VisualizeTwoPointcloud<PXYZ>(scene, model);

    std::cout << "Point Size of Scene: " << scene->points.size() << std::endl;
    std::cout << "Point Size of Model: " << model->points.size() << std::endl;

    std::cout << "Scene Point[1000]: " << scene->points[1000].x << " " << scene->points[1000].y << " " << scene->points[1000].z << std::endl;
    std::cout << "Model Point[100]: " << model->points[100].x << " " << model->points[100].y << " " << model->points[100].z << std::endl;

    //================== Local Descriptions Matching ==================
    matcher::LocalMatcher localMatch;

    std::vector<Eigen::Matrix4f> poses;
    std::string path = "/home/xlh/work/others2me/yong2me/regis_dataset/pcd_parts/Bunny_part/pose.txt";
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
    auto start_time = std::chrono::high_resolution_clock::now();
    PfhParameters pfh_param = localconfig.ReadPfhParam();
    pfh_param.common_params = commonparam;
    localMatch.PFHMatch(pfh_param);
    localMatch.AccuracyEstimate();
    localMatch.AbsolueAccuracyEstimate(poses[0]);
    auto pfh_time = std::chrono::high_resolution_clock::now();
    auto pfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pfh_time - start_time);
    std::cout << "PFH Matching Time Taken: " << pfh_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // FpfhParameters fpfh_param = localconfig.ReadFpfhParam();
    // fpfh_param.common_params = commonparam;
    // localMatch.FPFHMatch(fpfh_param);
    // localMatch.AccuracyEstimate();
    // auto fpfh_time = std::chrono::high_resolution_clock::now();
    // auto fpfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fpfh_time - pfh_time);
    // std::cout << "FPFH Matching Time Taken: " << pfh_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // RsdParameters rsd_param = localconfig.ReadRsdParam();
    // rsd_param.common_params = commonparam;
    // localMatch.RSDMatch(rsd_param);
    // localMatch.AccuracyEstimate();
    // auto rsd_time = std::chrono::high_resolution_clock::now();
    // auto rsd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rsd_time - fpfh_time);
    // std::cout << "RSD Matching Time Taken: " << rsd_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // Dsc3Parameters dsc3_param = localconfig.ReadDsc3Param();
    // dsc3_param.common_params = commonparam;
    // localMatch.DSC3Match(dsc3_param);
    // localMatch.AccuracyEstimate();
    // auto dsc3_time = std::chrono::high_resolution_clock::now();
    // auto dsc3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dsc3_time - rsd_time);
    // std::cout << "DSC3 Matching Time Taken: " << dsc3_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // UscParameters usc_param = localconfig.ReadUscParam();
    // usc_param.common_params = commonparam;
    // localMatch.USCMatch(usc_param);
    // localMatch.AccuracyEstimate();
    // auto usc_time = std::chrono::high_resolution_clock::now();
    // auto usc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(usc_time - dsc3_time);
    // std::cout << "USC Matching Time Taken: " << usc_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // ShotParameters shot_param = localconfig.ReadShotParam();
    // shot_param.common_params = commonparam;
    // localMatch.SHOTMatch(shot_param);
    // localMatch.AccuracyEstimate();
    // auto shot_time = std::chrono::high_resolution_clock::now();
    // auto shot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shot_time - usc_time);
    // std::cout << "SHOT Matching Time Taken: " << shot_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // SpinParameters spin_param = localconfig.ReadSpinParam();
    // spin_param.common_params = commonparam;
    // localMatch.SIMatch(spin_param);
    // localMatch.AccuracyEstimate();
    // auto spin_image_time = std::chrono::high_resolution_clock::now();
    // auto spin_image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spin_image_time - shot_time);
    // std::cout << "SPIN IAMGE Matching Time Taken: " << spin_image_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // RopsParameters rops_param = localconfig.ReadRopsParam();
    // rops_param.common_params = commonparam;
    // localMatch.ROPSMatch(rops_param);
    // localMatch.AccuracyEstimate();
    // auto rops_time = std::chrono::high_resolution_clock::now();
    // auto rops_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rops_time - spin_image_time);
    // std::cout << "ROPS Matching Time Taken: " << rops_duration.count() << " ms" << std::endl;
    // std::cout << "========================================" << std::endl;

    // 4. correspondences grouping
    // localMatch.CorresGrouping(0.15);

    return 0;
}