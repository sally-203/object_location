#include "global_matcher.h"

int main(){
    PXYZS::Ptr bunny(new PXYZS);
    PXYZS::Ptr duck(new PXYZS);
    PXYZS::Ptr hand(new PXYZS);
    PXYZS::Ptr cloud1(new PXYZS);
    PXYZS::Ptr cloud2(new PXYZS);
    PXYZS::Ptr cloud3(new PXYZS);

    pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Bunny.ply", *bunny);
    pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Duck.ply", *duck);
    pcl::io::loadPLYFile("/home/xlh/work/others2me/yong2me/regis_dataset/Hand.ply", *hand);
    
    const double theta1 = M_PI / 3;
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    T1(0, 0) = std::sin(theta1);
    T1(0, 1) = std::cos(theta1);
    T1(1, 0) = -std::cos(theta1);
    T1(1, 1) = std::sin(theta1);
    T1(0, 3) = 1.0;
    T1(1, 3) = 0.2;
    T1(2, 3) = 0.3;

    const double theta2 = M_PI / 5;
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2(0, 0) = std::sin(theta2);
    T2(0, 1) = std::cos(theta2);
    T2(1, 0) = -std::cos(theta2);
    T2(1, 1) = std::sin(theta2);
    T2(0, 3) = -0.8;
    T2(1, 3) = 0.4;
    T2(2, 3) = -0.6;

    const double theta3 = M_PI / 6;
    Eigen::Matrix4f T3 = Eigen::Matrix4f::Identity();
    T3(0, 0) = std::sin(theta3);
    T3(0, 1) = std::cos(theta3);
    T3(1, 0) = -std::cos(theta3);
    T3(1, 1) = std::sin(theta3);
    T3(0, 3) = -0.7;
    T3(1, 3) = 0.3;
    T3(2, 3) = -0.45;

    pcl::transformPointCloud(*duck, *duck, T1);
    pcl::transformPointCloud(*hand, *hand, T2);
    *cloud1 = *bunny + *duck + *hand;

    pcl::transformPointCloud(*bunny, *bunny, T1);
    pcl::transformPointCloud(*duck, *duck, T2);
    pcl::transformPointCloud(*hand, *hand, T3);
    *cloud2 = *bunny + *duck + *hand;

    pcl::io::savePLYFile("../data/3.1.ply", *cloud1);
    pcl::io::savePLYFile("../data/3.2.ply", *cloud2);

    return 0;
}