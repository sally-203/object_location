#include "local_matcher.h"

namespace matcher {

void LocalMatcher::SetSceneCloud(const PXYZS::Ptr scene)
{
    scene_cloud_ = scene;
    return;
}

void LocalMatcher::SetModelCloud(const PXYZS::Ptr model)
{
    model_cloud_ = model;
    return;
}

void LocalMatcher::ExtractISSKeypoints(bool flag, int salient_radius,
    int nonmax_radius, int min_neighbors,
    double threshold21, double threshold32,
    int num_threads)
{
    PXYZS::Ptr keypoints(new PXYZS);
    PXYZS::Ptr cloud(new PXYZS);
    pcl::ISSKeypoint3D<PXYZ, PXYZ> detector;

    if (flag) {
        cloud = scene_cloud_;
    } else {
        cloud = model_cloud_;
    }

    detector.setInputCloud(cloud);
    pcl::search::KdTree<PXYZ>::Ptr KdTree(new pcl::search::KdTree<PXYZ>);
    detector.setSearchMethod(KdTree);
    double resolution = ComputeCloudResolution(cloud);

    // Set the radius of the spherical neighborhood used to compute the scatter
    // matrix.
    detector.setSalientRadius(salient_radius * resolution);
    // Set the radius for the application of the non maxima supression algorithm.
    detector.setNonMaxRadius(nonmax_radius * resolution);
    // Set the minimum number of neighbors that has to be found while applying the
    // non maxima suppression algorithm.
    detector.setMinNeighbors(min_neighbors);
    // Set the upper bound on the ratio between the second and the first
    // eigenvalue.
    detector.setThreshold21(threshold21);
    // Set the upper bound on the ratio between the third and the second
    // eigenvalue.
    detector.setThreshold32(threshold32);
    // Set the number of prpcessing threads to use. 0 sets it to automatic.
    detector.setNumberOfThreads(num_threads);
    detector.compute(*keypoints);

    // flag==1, scene keypoints
    if (flag) {
        scene_keypoints_ = keypoints;
    }
    // flag==0, model keypoints
    else {
        model_keypoints_ = keypoints;
    }

    // Show_Keypoints(keypoints, cloud);
}

void LocalMatcher::ExtractDownSamplingKeypoints(bool flag, double radius)
{
    PXYZS::Ptr keypoints(new PXYZS);
    PXYZS::Ptr cloud(new PXYZS);

    if (flag) {
        cloud = scene_cloud_;
    } else {
        cloud = model_cloud_;
    }

    pcl::VoxelGrid<PXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(radius, radius, radius);
    sor.filter(*keypoints);

    if (flag) {
        scene_keypoints_ = keypoints;
    } else {
        model_keypoints_ = keypoints;
    }

    // Show_Keypoints(keypoints, cloud);
}

void LocalMatcher::AccuracyEstimate()
{
    PXYZS::Ptr aligned_model(new PXYZS);
    pcl::transformPointCloud(*model_cloud_, *aligned_model, transformations_);

    pcl::KdTreeFLANN<PXYZ> kdtree;

    kdtree.setInputCloud(aligned_model);
    for (size_t i = 0; i < scene_cloud_->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = kdtree.nearestKSearch(scene_cloud_->points[i], 1,
            neighbors, squaredDistances);
        if (neighborCount == 1) {
            distance_ += squaredDistances[0];
        }
    }

    distance_ = std::sqrt(distance_ / aligned_model->size());
    std::cout << "RMSE: " << distance_ << std::endl;
}

void LocalMatcher::CorresGrouping(double gc_size)
{
    std::vector<pcl::Correspondences> clusteredCorrespondences;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
        transformations;
    pcl::GeometricConsistencyGrouping<PXYZ, PXYZ> grouping;
    grouping.setSceneCloud(scene_keypoints_);
    grouping.setInputCloud(model_keypoints_);
    grouping.setModelSceneCorrespondences(correspondences_);

    grouping.setGCThreshold(2);
    grouping.setGCSize(gc_size);
    grouping.recognize(transformations, clusteredCorrespondences);

    std::cout << "clustered Correspondences size: "
              << clusteredCorrespondences.size() << std::endl;
    std::cout << "Model instances found: " << transformations.size() << std::endl
              << std::endl;
    for (size_t i = 0; i < transformations.size(); i++) {
        std::cout << "Instance " << (i + 1) << ":" << std::endl;
        std::cout << "\tHas " << clusteredCorrespondences[i].size()
                  << " correspondences." << std::endl
                  << std::endl;

        Eigen::Matrix3f rotation = transformations[i].block<3, 3>(0, 0);
        Eigen::Vector3f translation = transformations[i].block<3, 1>(0, 3);
        printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1),
            rotation(0, 2));
        printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1),
            rotation(1, 2));
        printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1),
            rotation(2, 2));
        std::cout << std::endl;
        printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1),
            translation(2));
        pcl::transformPointCloud(*model_cloud_, *model_cloud_, transformations[i]);
        Visualize(scene_cloud_, model_cloud_, "aligned model");
    }
}

void LocalMatcher::PFHMatch(double pfh_radius, double normal_radius,
    double distance_thre, int randomness,
    double inlier_fraction, int num_samples,
    double similiar_thre, double corres_distance,
    int nr_iterations)
{
    pcl::PointCloud<pcl::PFHSignature125>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::PFHSignature125>());
    pcl::PointCloud<pcl::PFHSignature125>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::PFHSignature125>());

    CalculatePfhDescri(scene_keypoints_, scene_descriptors, pfh_radius,
        normal_radius);
    CalculatePfhDescri(model_keypoints_, model_descriptors, pfh_radius,
        normal_radius);

    pcl::KdTreeFLANN<pcl::PFHSignature125> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).histogram[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    // VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::PFHSignature125> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    // Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculatePfhDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors, double pfh_radius,
    double normal_radius)
{
    pcl::PointCloud<PN>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>);

    EstimateNormalsByK(cloud, normals, 10);
    pcl::PFHEstimation<PXYZ, PN, pcl::PFHSignature125> pfh;
    pfh.setInputCloud(cloud);
    pfh.setInputNormals(normals);
    pfh.setSearchMethod(kdtree);
    pfh.setRadiusSearch(pfh_radius);

    pfh.compute(*descriptors);
}

void LocalMatcher::ROPSMatch(double rops_radius, int num_partions_bins,
    int num_rotations, double support_radius,
    double normal_radius, double distance_thre,
    int randomness, double inlier_fraction,
    int num_samples, double similiar_thre,
    double corres_distance, int nr_iterations)
{
    pcl::PointCloud<pcl::Histogram<135>>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::Histogram<135>>());
    pcl::PointCloud<pcl::Histogram<135>>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::Histogram<135>>());

    CalculateRopsDescri(scene_keypoints_, scene_descriptors, rops_radius,
        num_partions_bins, num_rotations, support_radius,
        normal_radius);
    CalculateRopsDescri(model_keypoints_, model_descriptors, rops_radius,
        num_partions_bins, num_rotations, support_radius,
        normal_radius);

    pcl::KdTreeFLANN<pcl::Histogram<135>> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).histogram[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::Histogram<135>> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateRopsDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::Histogram<135>>::Ptr descriptors, double rops_radius,
    int num_partions_bins, int num_rotations, double support_radius,
    double normal_radius)
{
    pcl::PointCloud<PN>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(
        new pcl::PointCloud<pcl::PointNormal>);

    EstimateNormalsByK(cloud, normals, 10);
    // perform triangulation
    pcl::concatenateFields(*cloud, *normals, *cloudNormals);
    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree2(
        new pcl::search::KdTree<pcl::PointNormal>);
    kdtree2->setInputCloud(cloudNormals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> triangulation;
    pcl::PolygonMesh triangles;
    triangulation.setSearchRadius(0.1);
    triangulation.setMu(2.5);
    triangulation.setMaximumNearestNeighbors(10);
    triangulation.setMaximumSurfaceAngle(M_PI / 4);
    triangulation.setNormalConsistency(false);
    triangulation.setMinimumAngle(M_PI / 18);
    triangulation.setMaximumAngle(2 * M_PI / 3);
    triangulation.setInputCloud(cloudNormals);
    triangulation.setSearchMethod(kdtree2);
    triangulation.reconstruct(triangles);

    // rops estimation object
    pcl::ROPSEstimation<PXYZ, pcl::Histogram<135>> rops;
    rops.setInputCloud(cloud);
    rops.setSearchMethod(kdtree);
    rops.setRadiusSearch(rops_radius);
    rops.setTriangles(triangles.polygons);
    rops.setNumberOfPartitionBins(num_partions_bins);
    rops.setNumberOfRotations(num_rotations);
    rops.setSupportRadius(support_radius);

    rops.compute(*descriptors);
}

void LocalMatcher::SIMatch(double si_radius, int image_width,
    double normal_radius, double distance_thre,
    int randomness, double inlier_fraction,
    int num_samples, double similiar_thre,
    double corres_distance, int nr_iterations)
{
    pcl::PointCloud<pcl::Histogram<153>>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::Histogram<153>>());
    pcl::PointCloud<pcl::Histogram<153>>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::Histogram<153>>());

    CalculateSiDescri(scene_keypoints_, scene_descriptors, si_radius, image_width,
        normal_radius);
    CalculateSiDescri(model_keypoints_, model_descriptors, si_radius, image_width,
        normal_radius);

    pcl::KdTreeFLANN<pcl::Histogram<153>> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).histogram[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::Histogram<153>> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateSiDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::Histogram<153>>::Ptr descriptors, double si_radius,
    int image_width, double normal_radius)
{
    pcl::PointCloud<PN>::Ptr normals(new pcl::PointCloud<PN>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    // EstimateNormals(cloud, kdtree, normals, normal_radius);
    EstimateNormalsByK(cloud, normals, 10);

    pcl::SpinImageEstimation<PXYZ, PN, pcl::Histogram<153>> si;
    si.setInputCloud(cloud);
    si.setInputNormals(normals);
    si.setRadiusSearch(si_radius);
    si.setImageWidth(image_width);

    si.compute(*descriptors);
}

void LocalMatcher::SHOTMatch(double shot_radius, double normal_radius,
    double distance_thre, int randomness,
    double inlier_fraction, int num_samples,
    double similiar_thre, double corres_distance,
    int nr_iterations)
{
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::SHOT352>());
    pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::SHOT352>());

    CalculateShotDescri(scene_keypoints_, scene_descriptors, shot_radius,
        normal_radius);
    CalculateShotDescri(model_keypoints_, model_descriptors, shot_radius,
        normal_radius);

    pcl::KdTreeFLANN<pcl::SHOT352> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).descriptor[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::SHOT352> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateShotDescri(
    const PXYZS::Ptr cloud, pcl::PointCloud<pcl::SHOT352>::Ptr descriptors,
    double shot_radius, double normal_radius)
{
    pcl::PointCloud<PN>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    EstimateNormalsByK(cloud, normals, 10);
    // EstimateNormals(cloud, kdtree, normals, normal_radius);

    pcl::SHOTEstimation<PXYZ, PN, pcl::SHOT352> shot;
    shot.setInputCloud(cloud);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(shot_radius);

    shot.compute(*descriptors);
}

void LocalMatcher::USCMatch(double usc_radius, double minimal_radius,
    double point_density_raidus, double local_radius,
    double distance_thre, int randomness,
    double inlier_fraction, int num_samples,
    double similiar_thre, double corres_distance,
    int nr_iterations)
{
    pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::UniqueShapeContext1960>());
    pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::UniqueShapeContext1960>());

    CalculateUscDescri(scene_keypoints_, scene_descriptors, usc_radius,
        minimal_radius, point_density_raidus, local_radius);
    CalculateUscDescri(model_keypoints_, model_descriptors, usc_radius,
        minimal_radius, point_density_raidus, local_radius);

    pcl::KdTreeFLANN<pcl::UniqueShapeContext1960> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).descriptor[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::UniqueShapeContext1960>
        pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateUscDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr descriptors,
    double usc_radius, double minimal_radius, double point_density_raidus,
    double local_radius)
{
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    pcl::UniqueShapeContext<PXYZ, pcl::UniqueShapeContext1960,
        pcl::ReferenceFrame>
        usc;
    usc.setInputCloud(cloud);
    usc.setRadiusSearch(usc_radius);
    usc.setMinimalRadius(minimal_radius);
    usc.setPointDensityRadius(point_density_raidus);
    usc.setLocalRadius(local_radius);

    usc.compute(*descriptors);
}

void LocalMatcher::DSC3Match(double dsc_radius, double minimal_radius,
    double point_density_raidus, double normal_radius,
    double distance_thre, int randomness,
    double inlier_fraction, int num_samples,
    double similiar_thre, double corres_distance,
    int nr_iterations)
{
    pcl::PointCloud<pcl::ShapeContext1980>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::ShapeContext1980>());
    pcl::PointCloud<pcl::ShapeContext1980>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::ShapeContext1980>());

    CalculateDscDescri(scene_keypoints_, scene_descriptors, dsc_radius,
        minimal_radius, point_density_raidus, normal_radius);
    CalculateDscDescri(model_keypoints_, model_descriptors, dsc_radius,
        minimal_radius, point_density_raidus, normal_radius);

    pcl::KdTreeFLANN<pcl::ShapeContext1980> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).descriptor[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::ShapeContext1980> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateDscDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::ShapeContext1980>::Ptr descriptors, double dsc_radius,
    double minimal_radius, double point_density_raidus, double normal_radius)
{
    pcl::PointCloud<PN>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    // EstimateNormals(cloud, kdtree, normals, normal_radius);
    EstimateNormalsByK(cloud, normals, 10);
    pcl::ShapeContext3DEstimation<PXYZ, PN, pcl::ShapeContext1980> sc3d;
    sc3d.setInputCloud(cloud);
    sc3d.setInputNormals(normals);
    sc3d.setSearchMethod(kdtree);
    sc3d.setRadiusSearch(dsc_radius);
    sc3d.setMinimalRadius(minimal_radius);
    sc3d.setPointDensityRadius(point_density_raidus);

    sc3d.compute(*descriptors);
}

void LocalMatcher::RSDMatch(double rsd_radius, double plane_radius,
    double normal_radius, double distance_thre,
    int randomness, double inlier_fraction,
    int num_samples, double similiar_thre,
    double corres_distance, int nr_iterations)
{
    pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::PrincipalRadiiRSD>());
    pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::PrincipalRadiiRSD>());

    CalculateRsdDescri(scene_keypoints_, scene_descriptors, rsd_radius,
        plane_radius, normal_radius);
    CalculateRsdDescri(model_keypoints_, model_descriptors, rsd_radius,
        plane_radius, normal_radius);

    pcl::KdTreeFLANN<pcl::PrincipalRadiiRSD> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
            neighbors, squaredDistances);
        // std::cout << squaredDistances[0] << std::endl;
        if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
            pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                squaredDistances[0]);
            correspondences->push_back(correspondence);
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::PrincipalRadiiRSD> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateRsdDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr descriptors, double rsd_radius,
    double plane_radius, double normal_radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    EstimateNormals(cloud, kdtree, normals, normal_radius);
    pcl::RSDEstimation<PXYZ, PN, pcl::PrincipalRadiiRSD> rsd;
    rsd.setInputCloud(cloud);
    rsd.setInputNormals(normals);
    rsd.setSearchMethod(kdtree);
    rsd.setRadiusSearch(rsd_radius);
    rsd.setPlaneRadius(plane_radius);
    rsd.setSaveHistograms(false);

    rsd.compute(*descriptors);
}

void LocalMatcher::FPFHMatch(double fpfh_radius, double normal_radius,
    double distance_thre, int randomness,
    double inlier_fraction, int num_samples,
    double similiar_thre, double corres_distance,
    int nr_iterations)
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors(
        new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors(
        new pcl::PointCloud<pcl::FPFHSignature33>());
    CalculateFpfhDescri(scene_keypoints_, scene_descriptors, fpfh_radius,
        normal_radius);
    CalculateFpfhDescri(model_keypoints_, model_descriptors, fpfh_radius,
        normal_radius);

    pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
    matching.setInputCloud(model_descriptors);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    for (size_t i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);

        if (std::isfinite(scene_descriptors->at(i).histogram[0])) {
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1,
                neighbors, squaredDistances);
            // std::cout << squaredDistances[0] << std::endl;
            if (neighborCount == 1 && squaredDistances[0] < distance_thre) {
                // model_index, scene_index, distance
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i),
                    squaredDistances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }
    correspondences_ = correspondences;
    std::cout << "Found " << correspondences->size() << " correspondences."
              << std::endl;
    // VisualizeCorrs();

    pcl::SampleConsensusPrerejective<PXYZ, PXYZ, pcl::FPFHSignature33> pose;
    PXYZS::Ptr alignedModel(new PXYZS);

    pose.setInputSource(model_keypoints_);
    pose.setInputTarget(scene_keypoints_);
    pose.setSourceFeatures(model_descriptors);
    pose.setTargetFeatures(scene_descriptors);

    pose.setCorrespondenceRandomness(randomness);
    pose.setInlierFraction(inlier_fraction);
    pose.setNumberOfSamples(num_samples);
    pose.setSimilarityThreshold(similiar_thre);
    pose.setMaxCorrespondenceDistance(corres_distance);
    pose.setMaximumIterations(nr_iterations);

    pose.align(*alignedModel);
    if (pose.hasConverged()) {
        transformations_ = pose.getFinalTransformation();
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
    } else {
        std::cout << "Did not converge." << std::endl;
    }

    // Visualize(scene_cloud_, alignedModel, "aligned model");
}

void LocalMatcher::CalculateFpfhDescri(
    const PXYZS::Ptr cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors, double fpfh_radius,
    double normal_radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<PXYZ>::Ptr kdtree(new pcl::search::KdTree<PXYZ>());

    // EstimateNormals(cloud, kdtree, normals, normal_radius);
    EstimateNormalsByK(cloud, normals, 10);
    pcl::FPFHEstimation<PXYZ, PN, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(kdtree);
    fpfh.setRadiusSearch(fpfh_radius);
    fpfh.compute(*descriptors);
}

void LocalMatcher::EstimateNormals(const PXYZS::Ptr cloud,
    pcl::search::KdTree<PXYZ>::Ptr kdtree,
    const PNS::Ptr normals, double radius)
{
    pcl::NormalEstimation<PXYZ, PN> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(radius);
    pcl::search::KdTree<PXYZ>::Ptr KdTree(new pcl::search::KdTree<PXYZ>);
    normalEstimation.setSearchMethod(KdTree);
    normalEstimation.compute(*normals);

    // visualize_normals(cloud, normals);
}

void LocalMatcher::EstimateNormalsByK(const PXYZS::Ptr cloud,
    const PNS::Ptr normals, int k)
{
    pcl::NormalEstimationOMP<PXYZ, pcl::Normal> norm_est;
    norm_est.setNumberOfThreads(4);
    norm_est.setKSearch(k);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);

    // visualize_normals(cloud, normals);
}

double LocalMatcher::ComputeCloudResolution(const PXYZS::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (!std::isfinite((*cloud)[i].x))
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2) {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

void LocalMatcher::ShowKeypoints(PXYZS::Ptr keypoints, PXYZS::Ptr cloud)
{
    std::cout << "keypoints: " << keypoints->points.size() << std::endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("Keypoints"));
    // RGB values for white
    //   viewer->setBackgroundColor(1.0, 1.0, 1.0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(
        cloud, 0, 225, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "cloud");
    viewer->addPointCloud<pcl::PointXYZ>(keypoints, "keypoints");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0.0, 0.0, "keypoints");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void LocalMatcher::Visualize(PXYZS::Ptr cloud1, PXYZS::Ptr cloud2,
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

void LocalMatcher::VisualizeCorrs()
{
    // 添加关键点
    pcl::visualization::PCLVisualizer viewer("corrs Viewer");
    viewer.setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color(
        model_cloud_, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(
        scene_cloud_, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        model_keypoint_color(model_keypoints_, 0, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        scene_keypoint_color(scene_keypoints_, 0, 0, 255);

    viewer.addPointCloud(model_keypoints_, model_keypoint_color,
        "model_keypoints");
    viewer.addPointCloud(scene_keypoints_, scene_keypoint_color,
        "scene_keypoints");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");

    viewer.addCorrespondences<pcl::PointXYZ>(model_keypoints_, scene_keypoints_,
        *correspondences_);
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void LocalMatcher::VisualizeNormals(const PXYZS::Ptr cloud,
    const PNS::Ptr normals)
{
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addPointCloud<PXYZ>(cloud, "cloud");
    viewer.addPointCloudNormals<PXYZ, PN>(cloud, normals, 10, 0.02, "normals");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 1.0, "normals");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "normals");

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

} // namespace matcher