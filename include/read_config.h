#include <iostream>
#include <iterator>
#include <libconfig.h++>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "global_parameters.h"
#include "local_parameters.h"

namespace ivgs_util {

class Config {
 public:
  template <typename T>
  T GetValue(const std::string& group, const std::string& key) const {
    T value;
    try {
      libconfig::Setting& root = this->cfg_.getRoot();
      libconfig::Setting& s = root;

      libconfig::Setting& setting = cfg_.lookup(group);
      setting.lookupValue(key, value);
    } catch (const libconfig::SettingNotFoundException& nfex) {
      throw std::runtime_error("Key not found: " + group + "." + key);
    } catch (const libconfig::SettingTypeException& tex) {
      throw std::runtime_error("Type mismatch for key: " + group + "." + key);
    }
    return value;
  }

 public:
  Config(const std::string& cfg_file_path) : cfg_path_(cfg_file_path) {
    try {
      cfg_.readFile(cfg_path_.c_str());
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
  }

 private:
  libconfig::Config cfg_;
  std::string cfg_path_;
};  // namespace ivgs_util

class LocalConfig : public Config {
 public:
  LocalConfig(const std::string& cfg_file_path) : Config(cfg_file_path) {}

  CommonParameters ReadCommonParam() const {
    CommonParameters commonParams;
    try {
      commonParams.show_flag = GetValue<bool>("CommonParameters", "show_flag");
      commonParams.randomness = GetValue<int>("CommonParameters", "randomness");
      commonParams.inlier_fraction =
          GetValue<double>("CommonParameters", "inlier_fraction");
      commonParams.num_samples =
          GetValue<int>("CommonParameters", "num_samples");
      commonParams.similar_thre =
          GetValue<double>("CommonParameters", "similar_thre");
      commonParams.corres_distance =
          GetValue<double>("CommonParameters", "corres_distance");
      commonParams.nr_iterations =
          GetValue<int>("CommonParameters", "nr_iterations");

      std::cout << "CommonParameters.show_flag: " << commonParams.show_flag
                << std::endl;
      std::cout << "CommonParameters.randomness: " << commonParams.randomness
                << std::endl;
      std::cout << "CommonParameters.inlier_fraction: "
                << commonParams.inlier_fraction << std::endl;
      std::cout << "CommonParameters.num_samples: " << commonParams.num_samples
                << std::endl;
      std::cout << "CommonParameters.similar_thre: "
                << commonParams.similar_thre << std::endl;
      std::cout << "CommonParameters.corres_distance: "
                << commonParams.corres_distance << std::endl;
      std::cout << "CommonParameters.nr_iterations: "
                << commonParams.nr_iterations << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return commonParams;
  }

  IssParameters ReadIssParam() const {
    IssParameters issParams;
    try {
      issParams.salient_radius =
          GetValue<int>("IssParameters", "salient_radius");
      issParams.nonmax_radius = GetValue<int>("IssParameters", "nonmax_radius");
      issParams.min_neighbors = GetValue<int>("IssParameters", "min_neighbors");
      issParams.threshold21 = GetValue<double>("IssParameters", "threshold21");
      issParams.threshold32 = GetValue<double>("IssParameters", "threshold32");
      issParams.num_threads = GetValue<int>("IssParameters", "num_threads");

      std::cout << "IssParameters.salient_radius: " << issParams.salient_radius
                << std::endl;
      std::cout << "IssParameters.nonmax_radius: " << issParams.nonmax_radius
                << std::endl;
      std::cout << "IssParameters.min_neighbors: " << issParams.min_neighbors
                << std::endl;
      std::cout << "IssParameters.threshold21: " << issParams.threshold21
                << std::endl;
      std::cout << "IssParameters.threshold32: " << issParams.threshold32
                << std::endl;
      std::cout << "IssParameters.num_threads: " << issParams.num_threads
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return issParams;
  }

  PfhParameters ReadPfhParam() const {
    PfhParameters pfhParams;
    try {
      pfhParams.pfh_radius = GetValue<double>("PfhParameters", "pfh_radius");
      pfhParams.distance_thre =
          GetValue<double>("PfhParameters", "distance_thre");

      std::cout << "PfhParameters.pfh_radius: " << pfhParams.pfh_radius
                << std::endl;
      std::cout << "PfhParameters.distance_thre: " << pfhParams.distance_thre
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return pfhParams;
  }

  FpfhParameters ReadFpfhParam() const {
    FpfhParameters fpfhParams;
    try {
      fpfhParams.fpfh_radius =
          GetValue<double>("FpfhParameters", "fpfh_radius");
      fpfhParams.distance_thre =
          GetValue<double>("FpfhParameters", "distance_thre");

      std::cout << "FpfhParameters.pfh_radius: " << fpfhParams.fpfh_radius
                << std::endl;
      std::cout << "FpfhParameters.distance_thre: " << fpfhParams.distance_thre
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return fpfhParams;
  }

  RsdParameters ReadRsdParam() const {
    RsdParameters rsdParams;
    try {
      rsdParams.rsd_radius = GetValue<double>("RsdParameters", "rsd_radius");
      rsdParams.plane_radius =
          GetValue<double>("RsdParameters", "plane_radius");
      rsdParams.distance_thre =
          GetValue<double>("RsdParameters", "distance_thre");

      std::cout << "RsdParameters.rsd_radius: " << rsdParams.rsd_radius
                << std::endl;
      std::cout << "RsdParameters.plane_radius: " << rsdParams.plane_radius
                << std::endl;
      std::cout << "RsdParameters.distance_thre: " << rsdParams.distance_thre
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return rsdParams;
  }

  Dsc3Parameters ReadDsc3Param() const {
    Dsc3Parameters dsc3Params;
    try {
      dsc3Params.dsc_radius = GetValue<double>("Dsc3Parameters", "dsc_radius");
      dsc3Params.minimal_radius =
          GetValue<double>("Dsc3Parameters", "minimal_radius");
      dsc3Params.point_density_radius =
          GetValue<double>("Dsc3Parameters", "point_density_radius");
      dsc3Params.normal_radius =
          GetValue<double>("Dsc3Parameters", "normal_radius");
      dsc3Params.distance_thre =
          GetValue<double>("Dsc3Parameters", "distance_thre");

      std::cout << "Dsc3Parameters.dsc_radius: " << dsc3Params.dsc_radius
                << std::endl;
      std::cout << "Dsc3Parameters.minimal_radius: "
                << dsc3Params.minimal_radius << std::endl;
      std::cout << "Dsc3Parameters.point_density_radius: "
                << dsc3Params.point_density_radius << std::endl;
      std::cout << "Dsc3Parameters.normal_radius: " << dsc3Params.normal_radius
                << std::endl;
      std::cout << "Dsc3Parameters.distance_thre: " << dsc3Params.distance_thre
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return dsc3Params;
  }

  UscParameters ReadUscParam() const {
    UscParameters uscParams;
    try {
      uscParams.usc_radius = GetValue<double>("UscParameters", "usc_radius");
      uscParams.minimal_radius =
          GetValue<double>("UscParameters", "minimal_radius");
      uscParams.point_density_radius =
          GetValue<double>("UscParameters", "point_density_radius");
      uscParams.local_radius =
          GetValue<double>("UscParameters", "local_radius");
      uscParams.distance_thre =
          GetValue<double>("UscParameters", "distance_thre");

      std::cout << "UscParameters.usc_radius: " << uscParams.usc_radius
                << std::endl;
      std::cout << "UscParameters.minimal_radius: " << uscParams.minimal_radius
                << std::endl;
      std::cout << "UscParameters.point_density_radius: "
                << uscParams.point_density_radius << std::endl;
      std::cout << "UscParameters.local_radius: " << uscParams.local_radius
                << std::endl;
      std::cout << "UscParameters.distance_thre: " << uscParams.distance_thre
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return uscParams;
  }

  ShotParameters ReadShotParam() const {
    ShotParameters shotParams;
    try {
      shotParams.shot_radius =
          GetValue<double>("ShotParameters", "shot_radius");
      shotParams.normal_radius =
          GetValue<double>("ShotParameters", "normal_radius");
      shotParams.distance_thre =
          GetValue<double>("ShotParameters", "distance_thre");

      std::cout << "ShotParameters.shot_radius: " << shotParams.shot_radius
                << std::endl;
      std::cout << "ShotParameters.normal_radius: " << shotParams.normal_radius
                << std::endl;
      std::cout << "ShotParameters.distance_thre: " << shotParams.distance_thre
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return shotParams;
  }

  SpinParameters ReadSpinParam() const {
    SpinParameters spinParams;
    try {
      spinParams.si_radius = GetValue<double>("SpinParameters", "si_radius");
      spinParams.image_width = GetValue<int>("SpinParameters", "image_width");
      spinParams.normal_radius =
          GetValue<double>("SpinParameters", "normal_radius");
      spinParams.distance_thre =
          GetValue<double>("SpinParameters", "distance_thre");

      std::cout << "SpinParameters.si_radius: " << spinParams.si_radius
                << std::endl;
      std::cout << "SpinParameters.image_width: " << spinParams.image_width
                << std::endl;
      std::cout << "SpinParameters.normal_radius: " << spinParams.normal_radius
                << std::endl;
      std::cout << "SpinParameters.distance_thre: " << spinParams.distance_thre
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return spinParams;
  }

  RopsParameters ReadRopsParam() const {
    RopsParameters ropsParams;
    try {
      ropsParams.rops_radius =
          GetValue<double>("RopsParameters", "rops_radius");
      ropsParams.num_partions_bins =
          GetValue<int>("RopsParameters", "num_partions_bins");
      ropsParams.num_rotations =
          GetValue<int>("RopsParameters", "num_rotations");
      ropsParams.support_radius =
          GetValue<double>("RopsParameters", "support_radius");
      ropsParams.normal_radius =
          GetValue<double>("RopsParameters", "normal_radius");
      ropsParams.distance_thre =
          GetValue<double>("RopsParameters", "distance_thre");

      std::cout << "RopsParameters.rops_radius: " << ropsParams.rops_radius
                << std::endl;
      std::cout << "RopsParameters.num_partions_bins: "
                << ropsParams.num_partions_bins << std::endl;
      std::cout << "RopsParameters.num_rotations: " << ropsParams.num_rotations
                << std::endl;
      std::cout << "RopsParameters.support_radius: "
                << ropsParams.support_radius << std::endl;
      std::cout << "RopsParameters.normal_radius: " << ropsParams.normal_radius
                << std::endl;
      std::cout << "RopsParameters.distance_thre: " << ropsParams.distance_thre
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return ropsParams;
  }
};

class GlobalConfig : public Config {
 public:
  GlobalConfig(const std::string& cfg_file_path) : Config(cfg_file_path) {}

  ClusterParameters ReadClusterParam() const {
    ClusterParameters clusterParams;
    try {
      clusterParams.show_flag =
          GetValue<bool>("ClusterParameters", "show_flag");
      clusterParams.cluster_tolerance =
          GetValue<double>("ClusterParameters", "cluster_tolerance");
      clusterParams.min_cluster_size =
          GetValue<int>("ClusterParameters", "min_cluster_size");
      clusterParams.max_cluster_size =
          GetValue<int>("ClusterParameters", "max_cluster_size");

      std::cout << "ClusterParameters.show_flag: " << clusterParams.show_flag
                << std::endl;
      std::cout << "ClusterParameters.cluster_tolerance: "
                << clusterParams.cluster_tolerance << std::endl;
      std::cout << "ClusterParameters.min_cluster_size: "
                << clusterParams.min_cluster_size << std::endl;
      std::cout << "ClusterParameters.max_cluster_size: "
                << clusterParams.max_cluster_size << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return clusterParams;
  }

  VfhParameters ReadVfhParam() const {
    VfhParameters vfhParams;
    try {
      vfhParams.show_flag = GetValue<bool>("VfhParameters", "show_flag");
      std::cout << "VfhParameters.show_flag: " << vfhParams.show_flag
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return vfhParams;
  }

  CvfhParameters ReadCvfhParam() const {
    CvfhParameters cvfhParams;
    try {
      cvfhParams.show_flag = GetValue<bool>("CvfhParameters", "show_flag");
      cvfhParams.eps_angle = GetValue<double>("CvfhParameters", "eps_angle");
      cvfhParams.curv_thre = GetValue<double>("CvfhParameters", "curv_thre");

      std::cout << "CvfhParameters.show_flag: " << cvfhParams.show_flag
                << std::endl;
      std::cout << "CvfhParameters.eps_angle: " << cvfhParams.eps_angle
                << std::endl;
      std::cout << "CvfhParameters.curv_thre: " << cvfhParams.curv_thre
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return cvfhParams;
  }

  OurcvfhParameters ReadOurcvfhParam() const {
    OurcvfhParameters ourcvfhParams;
    try {
      ourcvfhParams.show_flag =
          GetValue<bool>("OurcvfhParameters", "show_flag");
      ourcvfhParams.eps_angle =
          GetValue<double>("OurcvfhParameters", "eps_angle");
      ourcvfhParams.curv_thre =
          GetValue<double>("OurcvfhParameters", "curv_thre");
      ourcvfhParams.axis_ratio =
          GetValue<double>("OurcvfhParameters", "axis_ratio");

      std::cout << "OurcvfhParameters.show_flag: " << ourcvfhParams.show_flag
                << std::endl;
      std::cout << "OurcvfhParameters.eps_angle: " << ourcvfhParams.eps_angle
                << std::endl;
      std::cout << "OurcvfhParameters.curv_thre: " << ourcvfhParams.curv_thre
                << std::endl;
      std::cout << "OurcvfhParameters.axis_ratio: " << ourcvfhParams.axis_ratio
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return ourcvfhParams;
  }

  EsfParameters ReadEsfParam() const {
    EsfParameters esfParams;
    try {
      esfParams.show_flag = GetValue<bool>("EsfParameters", "show_flag");
      std::cout << "EsfParameters.show_flag: " << esfParams.show_flag
                << std::endl;
    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return esfParams;
  }

  GfpfhParameters ReadGfpfhParam() const {
    GfpfhParameters gfpfhParams;
    try {
      gfpfhParams.show_flag = GetValue<bool>("GfpfhParameters", "show_flag");
      gfpfhParams.octree_leaf_size =
          GetValue<double>("GfpfhParameters", "octree_leaf_size");
      gfpfhParams.num_classes =
          GetValue<int>("GfpfhParameters", "num_classes");

      std::cout << "GfpfhParameters.show_flag: " << gfpfhParams.show_flag
                << std::endl;
      std::cout << "GfpfhParameters.octree_leaf_size: "
                << gfpfhParams.octree_leaf_size << std::endl;
      std::cout << "GfpfhParameters.num_classes: " << gfpfhParams.num_classes
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return gfpfhParams;
  }

  GrsdParameters ReadGrsdParam() const {
    GrsdParameters grsdParams;
    try {
      grsdParams.show_flag = GetValue<bool>("GrsdParameters", "show_flag");
      grsdParams.grsd_radius = GetValue<double>("GrsdParameters", "grsd_radius");

      std::cout << "GrsdParameters.show_flag: " << grsdParams.show_flag
                << std::endl;
      std::cout << "GrsdParameters.grsd_radius: " << grsdParams.grsd_radius
                << std::endl;

    } catch (const libconfig::FileIOException& fioex) {
      throw std::runtime_error("I/O error while reading configuration file.");
    } catch (const libconfig::ParseException& pex) {
      throw std::runtime_error("Parse error at " + std::string(pex.getFile()) +
                               ":" + std::to_string(pex.getLine()) + " - " +
                               pex.getError());
    }
    return grsdParams;
  }
};
}  // namespace ivgs_util