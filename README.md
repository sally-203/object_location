# Object Location

## Descriptors

This is a project for point cloud matching and recognition based on traditional descriptors. The descriptors used for point cloud matching include PFH/FPFH/RSD/3DSC/USC/SHOT/SPIN IAMGE/ROPS, etc. The descriptors used for point cloud recognition include VFH/CVFH/OUR CVFH/ESF/GFPFH/GRSD.

## Requirements

- OpenCV
- PCL
- Eigen3
- libconfig++

## Usage

After install all requirements:

```C++
    mkdir build && cmake ..
    make -j
```

- Point Cloud Matching: ./test_match "path1 of 1st point cloud" "path2 of 2nd point cloud".
- Point Cloud Recognition: ./test_recognize "path1 of 1st point cloud" "path2 of 2nd point cloud".
- Modify the config file ("global_param.cfg" and "local_param.cfg") for better perfomance.
