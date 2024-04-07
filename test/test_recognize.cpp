#include "global_matcher.h"
#include "read_config.h"

int main(int argc, char** argv)
{
    ivgs_util::GlobalConfig globalconfig("../config/global_param.cfg");
        
    PXYZS::Ptr scene(new PXYZS);
    PXYZS::Ptr model(new PXYZS);

    std::string path1 = argv[1];
    std::string path2 = argv[2];
    std::cout << path1 << "\n" << path2 << std::endl;

    pcl::io::loadPLYFile(path1, *scene);
    pcl::io::loadPLYFile(path2, *model);

    std::cout << "Point Size of Scene: " << scene->points.size() << std::endl;
    std::cout << "Point Size of Model: " << model->points.size() << std::endl;

    //==================Global Descriptions Matching==================
    matcher::GlobalMatcher globalMatcher;

    globalMatcher.SetSceneCloud(scene);
    globalMatcher.SetModelCloud(model);
    
    ClusterParameters cluster_param = globalconfig.ReadClusterParam();
    globalMatcher.ClusterPointCloud(true, scene, cluster_param);
    globalMatcher.ClusterPointCloud(false, model, cluster_param);

    VfhParameters vfh_param = globalconfig.ReadVfhParam();
    auto start_vfh_time = std::chrono::high_resolution_clock::now();
    globalMatcher.VFHMatch(vfh_param);
    auto vfh_time = std::chrono::high_resolution_clock::now();
    auto vfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vfh_time - start_vfh_time);
    std::cout << "VFH Recognizing Time Taken: " << vfh_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    CvfhParameters cvfh_param = globalconfig.ReadCvfhParam();
    auto start_cvfh_time = std::chrono::high_resolution_clock::now();
    globalMatcher.CVFHMatch(cvfh_param); // multi descris
    auto cvfh_time = std::chrono::high_resolution_clock::now();
    auto cvfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cvfh_time - start_cvfh_time);
    std::cout << "CVFH Recognizing Time Taken: " << cvfh_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    OurcvfhParameters our_cvfh_param = globalconfig.ReadOurcvfhParam();
    auto start_our_cvfh_time = std::chrono::high_resolution_clock::now();
    globalMatcher.OUR_CVFH_Match(our_cvfh_param); // multi descris
    auto our_cvfh_time = std::chrono::high_resolution_clock::now();
    auto our_cvfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(our_cvfh_time-start_our_cvfh_time);
    std::cout << "OUR CVFH Recognizing Time Taken: " << our_cvfh_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    EsfParameters esf_param = globalconfig.ReadEsfParam();
    auto start_esf_time = std::chrono::high_resolution_clock::now();
    globalMatcher.ESFMatch(esf_param);
    auto esf_time = std::chrono::high_resolution_clock::now();
    auto esf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(esf_time - start_esf_time);
    std::cout << "ESF Recognizing Time Taken: " << esf_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    GfpfhParameters gfpfh_param = globalconfig.ReadGfpfhParam();
    auto start_gfpfh_time = std::chrono::high_resolution_clock::now();
    globalMatcher.GFPFHMatch(gfpfh_param);
    auto gfpfh_time = std::chrono::high_resolution_clock::now();
    auto gfpfh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gfpfh_time - start_gfpfh_time);
    std::cout << "GFPFH Recognizing Time Taken: " << gfpfh_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    GrsdParameters grsd_param = globalconfig.ReadGrsdParam();
    auto start_grsd_time = std::chrono::high_resolution_clock::now();
    globalMatcher.GRSDMatch(grsd_param);
    auto grsd_time = std::chrono::high_resolution_clock::now();
    auto grsd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(grsd_time-start_grsd_time);
    std::cout << "GRSD Recognizing Time Taken: " << grsd_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
