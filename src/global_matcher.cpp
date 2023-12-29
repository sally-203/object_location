#include "global_matcher.h"

typedef pcl::Histogram<90> CRH90;

namespace matcher {

void GlobalMatcher::SetModelCloud(const PXYZS::Ptr model)
{
    model_cloud_ = model;
    return;
}

void GlobalMatcher::SetSceneCloud(const PXYZS::Ptr scene)
{
    scene_cloud_ = scene;
    return;
}

void GlobalMatcher::Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
    const std::string cloudName)
{
    pcl::visualization::PCLVisualizer viewer("Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<PXYZ> pc_1_color(cloud1, 0,
        0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<PXYZ> pc_2_color(cloud2, 255,
        0, 0);
    viewer.setBackgroundColor(255, 255, 255);
    viewer.setWindowName(cloudName);
    viewer.addPointCloud(cloud1, pc_1_color, "cloud1");
    viewer.addPointCloud(cloud2, pc_2_color, "cloud2");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void GlobalMatcher::ClusterScenePointCloud(const PXYZS::Ptr cloud,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size)
{
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>);
    kdtree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<PXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        PXYZS::Ptr cluster(new PXYZS);
        for (const auto& index : indices.indices) {
            cluster->points.push_back(cloud->points[index]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        scene_clusters_.push_back(cluster);
    }
    std::cout << "scene cluster size: " << scene_clusters_.size() << std::endl;
    // visualizeClusters(scene_clusters_);
}

void GlobalMatcher::ClusterModelPointCloud(const PXYZS::Ptr cloud,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size)
{
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>);
    kdtree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<PXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        PXYZS::Ptr cluster(new PXYZS);
        for (const auto& index : indices.indices) {
            cluster->points.push_back(cloud->points[index]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        model_clusters_.push_back(cluster);
    }
    std::cout << "model cluster size: " << model_clusters_.size() << std::endl;
    // visualizeClusters(model_clusters_);
}

void GlobalMatcher::ICP(const PXYZS::Ptr source_cloud,
    const PXYZS::Ptr target_cloud)
{
    // 定义 ICP 对象
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    // 设置源点云和目标点云
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    // 设置其他参数，根据需要进行调整
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setMaximumIterations(50);

    // 定义输出点云，存储配准结果
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // 运行 ICP 算法
    icp.align(*aligned_cloud);

    // 输出配准的变换矩阵
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    std::cout << "Final Transformation Matrix:" << std::endl
              << transformation << std::endl;

    // 可视化结果
    Visualize(aligned_cloud, target_cloud, "registration clouds");
}

void GlobalMatcher::GRSDMatch(double grsd_radius)
{
    pcl::PointCloud<pcl::GRSDSignature21>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::GRSDSignature21>());
    for (const auto cluster : model_clusters_) {
        pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors(
            new pcl::PointCloud<pcl::GRSDSignature21>());
        CalculateGrsdDescri(cluster, descriptors, grsd_radius);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "The Grsd size in cluster cloud is not 1" << std::endl;
            return;
        }
    }

    pcl::PointCloud<pcl::GRSDSignature21>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::GRSDSignature21>());
    for (const auto cluster : scene_clusters_) {
        pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors(
            new pcl::PointCloud<pcl::GRSDSignature21>());
        CalculateGrsdDescri(cluster, descriptors, grsd_radius);
        if (descriptors->size() == 1) {
            scene_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "The Grsd size in cluster cloud is not 1" << std::endl;
            return;
        }
    }

    pcl::KdTreeFLANN<pcl::GRSDSignature21> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        // std::cout << squaredDistances[0] << std::endl;
        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;

    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
        // ICP(scene_clusters_[index_scene], model_clusters_[index_model]);
    }

    return;
}

void GlobalMatcher::CalculateGrsdDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors,
    double grsd_radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    EstimateNormalsByK(cloud, normals, 10);

    pcl::GRSDEstimation<PXYZ, PN, pcl::GRSDSignature21> grsd;
    grsd.setInputCloud(cloud);
    grsd.setInputNormals(normals);
    grsd.setSearchMethod(kdtree);
    grsd.setRadiusSearch(grsd_radius);

    grsd.compute(*descriptors);
}

void GlobalMatcher::VFHMatch()
{
    pcl::PointCloud<pcl::VFHSignature308>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : model_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateVfhDescri(cluster, descriptors);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "The VFH size in cluster cloud is not 1" << std::endl;
            return;
        }
    }

    pcl::PointCloud<pcl::VFHSignature308>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : scene_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateVfhDescri(cluster, descriptors);
        if (descriptors->size() == 1) {
            scene_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "The VFH size in cluster cloud is not 1" << std::endl;
            return;
        }
    }
    // std::cout << "model descriptions[0]: \n"
    //           << model_descriptors->points[0] << std::endl;
    // std::cout << "model descriptions[1]: \n"
    //           << model_descriptors->points[1] << std::endl;
    // std::cout << "scene descriptions[0]: \n"
    //           << scene_descriptors->points[0] << std::endl;
    // std::cout << "scene descriptions[1]: \n"
    //           << scene_descriptors->points[1] << std::endl;

    pcl::KdTreeFLANN<pcl::VFHSignature308> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        // std::cout << "scene index, distance: " << neighbors[0] << " "
        //           << squaredDistances[0] << std::endl;

        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;
    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        std::cout << "scene, model: " << index_scene << " " << index_model
                  << std::endl;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
        // ICP(scene_clusters_[index_scene], model_clusters_[index_model]);
    }

    return;
}

void GlobalMatcher::CalculateVfhDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    EstimateNormalsByK(cloud, normals, 10);

    pcl::VFHEstimation<PXYZ, PN, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);
    vfh.setSearchMethod(kdtree);
    vfh.setNormalizeBins(true);
    vfh.setNormalizeDistance(false);

    vfh.compute(*descriptors);
}

void GlobalMatcher::GFPFHMatch(double octree_leaf_size,
    int num_classes)
{
    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr model_descriptors(new pcl::PointCloud<pcl::GFPFHSignature16>);
    for (int i = 0; i < model_clusters_.size(); ++i) {
        auto cluster = model_clusters_[i];
        pcl::PointCloud<pcl::PointXYZL>::Ptr object(new pcl::PointCloud<pcl::PointXYZL>());
        for (int j = 0; j < cluster->points.size(); ++j) {
            pcl::PointXYZL point;
            point.x = cluster->points[j].x;
            point.y = cluster->points[j].y;
            point.z = cluster->points[j].z;
            point.label = 1 + j % scene_clusters_.size();
            object->push_back(point);
        }
        pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptors(new pcl::PointCloud<pcl::GFPFHSignature16>);
        CalculateGfpfhDescri(object, descriptors, octree_leaf_size, num_classes);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "descriptors'size is not 1" << descriptors->size() << std::endl;
            return;
        }
    }
    std::cout << "size of model descriptors: " << model_descriptors->size() << std::endl;

    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr scene_descriptors(new pcl::PointCloud<pcl::GFPFHSignature16>);
    for (int i = 0; i < scene_clusters_.size(); ++i) {
        auto cluster = scene_clusters_[i];
        pcl::PointCloud<pcl::PointXYZL>::Ptr object(new pcl::PointCloud<pcl::PointXYZL>());
        for (int j = 0; j < cluster->points.size(); ++j) {
            pcl::PointXYZL point;
            point.x = cluster->points[j].x;
            point.y = cluster->points[j].y;
            point.z = cluster->points[j].z;
            point.label = 1 + j % scene_clusters_.size();
            object->push_back(point);
        }
        pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptiors(new pcl::PointCloud<pcl::GFPFHSignature16>);
        CalculateGfpfhDescri(object, descriptiors, octree_leaf_size, num_classes);
        if (descriptiors->size() == 1) {
            scene_descriptors->push_back(descriptiors->at(0));
        } else {
            std::cout << "descriptors'size is not 1" << descriptiors->size() << std::endl;
            return;
        }
    }
    std::cout << "size of scene descriptors: " << scene_descriptors->size() << std::endl;

    pcl::KdTreeFLANN<pcl::GFPFHSignature16> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;

    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
    }

    return;
}

void GlobalMatcher::CalculateGfpfhDescri(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr descriptions,
    double octree_leaf_size, double num_classes)
{
    pcl::GFPFHEstimation<pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;
    gfpfh.setInputCloud(cloud);
    gfpfh.setInputLabels(cloud);
    gfpfh.setOctreeLeafSize(octree_leaf_size);
    gfpfh.setNumberOfClasses(num_classes);

    gfpfh.compute(*descriptions);
}

void GlobalMatcher::ESFMatch()
{
    pcl::PointCloud<pcl::ESFSignature640>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::ESFSignature640>());
    for (const auto cluster : model_clusters_) {
        pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptors(
            new pcl::PointCloud<pcl::ESFSignature640>());
        CalculateEsfDescri(cluster, descriptors);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "descriptors'size is not 1" << descriptors->size()
                      << std::endl;
            return;
        }
    }
    std::cout << "size of model descriptors: " << model_descriptors->size()
              << std::endl;

    pcl::PointCloud<pcl::ESFSignature640>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::ESFSignature640>());
    for (const auto cluster : scene_clusters_) {
        pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptors(
            new pcl::PointCloud<pcl::ESFSignature640>());
        CalculateEsfDescri(cluster, descriptors);
        if (descriptors->size() == 1) {
            scene_descriptors->push_back(descriptors->at(0));
        } else {
            std::cout << "descriptors'size is not 1" << descriptors->size()
                      << std::endl;
            return;
        }
    }
    std::cout << "size of scene descriptors: " << scene_descriptors->size()
              << std::endl;

    pcl::KdTreeFLANN<pcl::ESFSignature640> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        // std::cout << squaredDistances[0] << std::endl;
        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;

    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
        // ICP(scene_clusters_[index_scene], model_clusters_[index_model]);
    }

    return;
}

void GlobalMatcher::CalculateEsfDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptions)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    EstimateNormalsByK(cloud, normals, 10);

    pcl::ESFEstimation<PXYZ, pcl::ESFSignature640> esf;
    esf.setInputCloud(cloud);
    esf.compute(*descriptions);
}

void GlobalMatcher::OUR_CVFH_Match(double eps_angle, double curv_thre,
    double axis_ratio)
{
    pcl::PointCloud<pcl::VFHSignature308>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : model_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateOurcvfhDescri(cluster, descriptors, eps_angle, curv_thre,
            axis_ratio);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descri(
                new pcl::PointCloud<pcl::VFHSignature308>());
            resize(new_descri, descriptors);
            model_descriptors->push_back(new_descri->at(0));

            std::cout << "size of new descriptors: " << new_descri->size()
                      << std::endl;
        }
    }
    std::cout << "size of model descriptors: " << model_descriptors->size()
              << std::endl;

    pcl::PointCloud<pcl::VFHSignature308>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : scene_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateOurcvfhDescri(cluster, descriptors, eps_angle, curv_thre,
            axis_ratio);
        if (descriptors->size() == 1) {
            scene_descriptors->push_back(descriptors->at(0));
        } else {
            pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descri(
                new pcl::PointCloud<pcl::VFHSignature308>());
            resize(new_descri, descriptors);
            scene_descriptors->push_back(new_descri->at(0));

            std::cout << "size of new descriptors: " << new_descri->size()
                      << std::endl;
        }
    }
    std::cout << "size of scene descriptors: " << scene_descriptors->size()
              << std::endl;

    pcl::KdTreeFLANN<pcl::VFHSignature308> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        // std::cout << squaredDistances[0] << std::endl;
        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;

    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
        // ICP(scene_clusters_[index_scene], model_clusters_[index_model]);
    }

    return;
}

void GlobalMatcher::CalculateOurcvfhDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptions, double eps_angle,
    double curv_thre, double axis_ratio)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    EstimateNormalsByK(cloud, normals, 10);

    pcl::OURCVFHEstimation<PXYZ, PN, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(cloud);
    ourcvfh.setInputNormals(normals);
    ourcvfh.setSearchMethod(kdtree);
    ourcvfh.setEPSAngleThreshold(eps_angle);
    ourcvfh.setCurvatureThreshold(curv_thre);
    ourcvfh.setNormalizeBins(false);
    ourcvfh.setAxisRatio(axis_ratio);

    ourcvfh.compute(*descriptions);
}

void GlobalMatcher::CVFHMatch(double eps_angle, double curv_thre)
{
    pcl::PointCloud<pcl::VFHSignature308>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : model_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateCvfhDescri(cluster, descriptors, eps_angle, curv_thre);
        if (descriptors->size() == 1) {
            model_descriptors->push_back(descriptors->at(0));
        } else {
            pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descri(
                new pcl::PointCloud<pcl::VFHSignature308>());
            resize(new_descri, descriptors);
            model_descriptors->push_back(new_descri->at(0));

            std::cout << "size of new descriptors: " << new_descri->size()
                      << std::endl;
        }
    }
    std::cout << "size of model descriptors: " << model_descriptors->size()
              << std::endl;

    pcl::PointCloud<pcl::VFHSignature308>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::VFHSignature308>());
    for (const auto cluster : scene_clusters_) {
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(
            new pcl::PointCloud<pcl::VFHSignature308>());
        CalculateCvfhDescri(cluster, descriptors, eps_angle, curv_thre);
        if (descriptors->size() == 1) {
            scene_descriptors->push_back(descriptors->at(0));
        } else {
            pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descri(
                new pcl::PointCloud<pcl::VFHSignature308>());
            resize(new_descri, descriptors);
            scene_descriptors->push_back(new_descri->at(0));

            std::cout << "size of new descriptors: " << new_descri->size()
                      << std::endl;
        }
    }
    std::cout << "size of scene descriptors: " << scene_descriptors->size()
              << std::endl;

    pcl::KdTreeFLANN<pcl::VFHSignature308> kdtree;
    kdtree.setInputCloud(scene_descriptors);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < model_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(model_descriptors->at(i), 1,
            neighbors, squaredDistances);

        // std::cout << squaredDistances[0] << std::endl;
        if (neighborCount == 1) {
            pcl::Correspondence correspondence(
                neighbors[0], static_cast<int>(i),
                squaredDistances[0]); // [scene_index, model_index, distance]
            correspondences->push_back(correspondence);
        }
    }

    correspondences_ = correspondences;

    for (int i = 0; i < correspondences_->size(); ++i) {
        int index_scene = correspondences_->at(i).index_query;
        int index_model = correspondences_->at(i).index_match;
        Visualize(scene_clusters_[index_scene], model_clusters_[index_model],
            "corresponding cluster");
        // ICP(scene_clusters_[index_scene], model_clusters_[index_model]);
    }
    return;
}

void GlobalMatcher::CalculateCvfhDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors, double eps_angle,
    double curv_thre)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    EstimateNormalsByK(cloud, normals, 10);

    pcl::CVFHEstimation<PXYZ, PN, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(cloud);
    cvfh.setInputNormals(normals);
    cvfh.setEPSAngleThreshold(eps_angle);
    cvfh.setCurvatureThreshold(curv_thre);
    cvfh.setNormalizeBins(false);

    cvfh.compute(*descriptors);
}

void GlobalMatcher::EstimateNormalsByK(const PXYZS::Ptr cloud,
    const PNS::Ptr normals, int k)
{
    pcl::NormalEstimationOMP<PXYZ, pcl::Normal> norm_est;
    norm_est.setNumberOfThreads(4);
    norm_est.setKSearch(k);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);
}

void GlobalMatcher::visualizeClusters(const std::vector<PXYZS::Ptr>& clusters)
{
    pcl::visualization::PCLVisualizer viewer("Cluster Viewer");

    int cluster_id = 0;
    for (const auto& cluster : clusters) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            color_handler(cluster, rand() % 256, rand() % 256, rand() % 256);

        viewer.addPointCloud<pcl::PointXYZ>(
            cluster, color_handler, "cluster_" + std::to_string(cluster_id));
        cluster_id++;
    }

    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.spin();
}

void GlobalMatcher::VisualizeCorrs()
{
    // 添加关键点
    pcl::visualization::PCLVisualizer viewer("corrs Viewer");
    viewer.setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color(
        model_cloud_, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(
        scene_cloud_, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        model_keypoint_color(model_cloud_, 0, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        scene_keypoint_color(scene_cloud_, 0, 0, 255);

    viewer.addPointCloud(model_cloud_, model_keypoint_color, "model_keypoints");
    viewer.addPointCloud(scene_cloud_, scene_keypoint_color, "scene_keypoints");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");

    viewer.addCorrespondences<pcl::PointXYZ>(model_cloud_, scene_cloud_,
        *correspondences_);
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void GlobalMatcher::print(Eigen::Matrix4f& transformations)
{
    Eigen::Matrix3f rotation = transformations_.block<3, 3>(0, 0);
    Eigen::Vector3f translation = transformations_.block<3, 1>(0, 3);

    std::cout << "Transformation matrix:" << std::endl
              << std::endl;
    printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1),
        rotation(0, 2));
    printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1),
        rotation(1, 2));
    printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1),
        rotation(2, 2));
    std::cout << std::endl;
    printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1),
        translation(2));
}

// 如果全局描述符>1个，则对所有描述符进行平均，get new_descriptors
void GlobalMatcher::resize(
    pcl::PointCloud<pcl::VFHSignature308>::Ptr new_descriptors,
    pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptiors)
{
    for (size_t i = 1; i < descriptiors->size(); ++i) {
        for (size_t j = 0; j < descriptiors->at(0).descriptorSize(); ++j) {
            descriptiors->at(0).histogram[j] += descriptiors->at(i).histogram[j];
        }
    }

    for (size_t i = 0; i < descriptiors->at(0).descriptorSize(); ++i) {
        descriptiors->at(0).histogram[i] /= descriptiors->size();
    }

    new_descriptors->push_back(descriptiors->at(0));
}

} // namespace matcher