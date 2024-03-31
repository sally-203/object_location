#include "global_matcher.h"
#include "read_config.h"

int main()
{
    ivgs_util::GlobalConfig globalconfig("../config/global_param.cfg");

    ClusterParameters clusterparam = globalconfig.ReadClusterParam();
    VfhParameters vfhparam = globalconfig.ReadVfhParam();
    CvfhParameters cvfhparam = globalconfig.ReadCvfhParam();
    OurcvfhParameters ourcvfhparam = globalconfig.ReadOurcvfhParam();
    EsfParameters esfparam = globalconfig.ReadEsfParam();
    GfpfhParameters gfpfhparam = globalconfig.ReadGfpfhParam();
    GrsdParameters grsdparam = globalconfig.ReadGrsdParam();
    
    return 0;
    
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
    auto start_time = std::chrono::high_resolution_clock::now();

    globalMatcher.SetSceneCloud(scene);
    globalMatcher.SetModelCloud(model);

    ClusterParameters cluster_param;
    globalMatcher.ClusterPointCloud(true, scene, cluster_param);
    globalMatcher.ClusterPointCloud(false, model, cluster_param);

    VfhParameters vfh_param;
    globalMatcher.VFHMatch(vfh_param);

    CvfhParameters cvfh_param;
    globalMatcher.CVFHMatch(cvfh_param); // multi descris

    OurcvfhParameters our_cvfh_param;
    globalMatcher.OUR_CVFH_Match(our_cvfh_param); // multi descris

    EsfParameters esf_param;
    globalMatcher.ESFMatch(esf_param);

    GfpfhParameters gfpfh_param;
    globalMatcher.GFPFHMatch(gfpfh_param);

    GrsdParameters grsd_param;
    globalMatcher.GRSDMatch(grsd_param);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
    std::cout << "Recognition Time taken: " << duration.count() << " ms" << std::endl;

    return 0;
}
