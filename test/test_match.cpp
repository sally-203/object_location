#include "dataviewer_pointcloud.h"
#include "local_matcher.h"
#include <pcl/io/pcd_io.h>
int main()
{
    util::DataViewerPointCloud viewer;
    PXYZS::Ptr scene(new PXYZS);
    PXYZS::Ptr model(new PXYZS);
    // ========================= OUR DATASET (m)===========================
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/seg/5.ply",
    //                      *scene);
    // pcl::io::loadPLYFile("/home/xlh/work/dataset/feature_matching/Standard/S_model.ply",
    //                      *model);

    // ========================= YONG'S DATASET(m) =========================
    if (pcl::io::loadPCDFile("/home/xlh/work/others2me/yong2me/regis_dataset/pcd_parts/Bunny_part/1.pcd", *scene) == -1) {
        PCL_ERROR("Couldn't read scene file \n");
        return -1;
    }

    if (pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Bunny.ply", *model) == -1) {
        PCL_ERROR("Couldn't read model file \n");
        return -1;
    }
    viewer.VisualizeTwoPointcloud<PXYZ>(scene, model);
    std::cout << "Point Size of Scene: " << scene->points.size() << std::endl;
    std::cout << "Point Size of Model: " << model->points.size() << std::endl;

    std::cout << "Scene Point[1000]: " << scene->points[1000].x << " " << scene->points[1000].y << " " << scene->points[1000].z << std::endl;
    std::cout << "Model Point[100]: " << model->points[100].x << " " << model->points[100].y << " " << model->points[100].z << std::endl;

    // for (auto& p : scene->points) {
    //     p.x /= 1000;
    //     p.y /= 1000;
    //     p.z /= 1000;
    // }

    // for (auto& p : model->points) {
    //     p.x /= 1000;
    //     p.y /= 1000;
    //     p.z /= 1000;
    // }
    //================== Local Descriptions Matching ==================
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

    PfhParameters pfh_param;
    localMatch.PFHMatch(pfh_param); // OK
    auto pfh_time = std::chrono::high_resolution_clock::now();

    FpfhParameters fpfh_param;
    localMatch.FPFHMatch(fpfh_param);
    auto fpfh_time = std::chrono::high_resolution_clock::now();

    RsdParameters rsd_param;
    localMatch.RSDMatch(rsd_param); // OK
    auto rsd_time = std::chrono::high_resolution_clock::now();

    Dsc3Parameters dsc3_param;
    localMatch.DSC3Match(dsc3_param);
    auto dsc3_time = std::chrono::high_resolution_clock::now();

    UscParameters usc_param;
    localMatch.USCMatch(usc_param); // OK
    auto usc_time = std::chrono::high_resolution_clock::now();

    ShotParameters shot_param;
    localMatch.SHOTMatch(shot_param); // OK
    auto shot_time = std::chrono::high_resolution_clock::now();

    SpinParameters spin_param;
    localMatch.SIMatch(spin_param); // OK
    auto spin_image_time = std::chrono::high_resolution_clock::now();

    RopsParameters rops_param;
    localMatch.ROPSMatch(rops_param); // OK
    auto rops_time = std::chrono::high_resolution_clock::now();

    // 4. correspondences grouping
    // localMatch.CorresGrouping(0.15);

    // 5. estimate speed and accuracy
    auto pfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pfh_time - start_time);
    auto fpfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fpfh_time - pfh_time);
    auto rsd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rsd_time - fpfh_time);
    auto dsc3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dsc3_time - rsd_time);
    auto usc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(usc_time - dsc3_time);
    auto shot_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shot_time - usc_time);
    auto spin_image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spin_image_time - shot_time);
    auto rops_duration = std::chrono::duration_cast<std::chrono::milliseconds>(rops_time - spin_image_time);

    std::cout << "PFH Matching Time Taken: " << pfh_duration.count() << " ms" << std::endl;
    std::cout << "FPFH Matching Time Taken: " << pfh_duration.count() << " ms" << std::endl;
    std::cout << "RSD Matching Time Taken: " << rsd_duration.count() << " ms" << std::endl;
    std::cout << "DSC3 Matching Time Taken: " << dsc3_duration.count() << " ms" << std::endl;
    std::cout << "USC Matching Time Taken: " << usc_duration.count() << " ms" << std::endl;
    std::cout << "SHOT Matching Time Taken: " << shot_duration.count() << " ms" << std::endl;
    std::cout << "SPIN IAMGE Matching Time Taken: " << spin_image_duration.count() << " ms" << std::endl;
    std::cout << "ROPS Matching Time Taken: " << rops_duration.count() << " ms" << std::endl;

    localMatch.AccuracyEstimate();

    return 0;
}