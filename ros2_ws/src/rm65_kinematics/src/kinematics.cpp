#include "rm65_kinematics/kinematics.hpp"
#include <cmath>

namespace rm65_kinematics
{
namespace
{
const std::array<DHParam, 6> DH_TABLE = {DHParam{0.0, +1.5708, 0.2405}, DHParam{0.256, 0.0, 0.0},
                                          DHParam{0.0, +1.5708, 0.0}, DHParam{0.0, -1.5708, -0.210},
                                          DHParam{0.0, +1.5708, 0.0}, DHParam{0.0, 0.0, -0.1725}};

Eigen::Isometry3d dh_transform(const DHParam & dh, double theta)
{
  Eigen::AngleAxisd rot_z(theta, Eigen::Vector3d::UnitZ());
  Eigen::Translation3d trans_z(0, 0, dh.d);
  Eigen::Translation3d trans_x(dh.a, 0, 0);
  Eigen::AngleAxisd rot_x(dh.alpha, Eigen::Vector3d::UnitX());
  return rot_z * trans_z * trans_x * rot_x;
}
}  // namespace

Eigen::Isometry3d forward_transform(const std::array<double, 6> & joints)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  for (size_t i = 0; i < DH_TABLE.size(); ++i) {
    transform = transform * dh_transform(DH_TABLE[i], joints[i]);
  }
  return transform;
}

std::array<double, 6> forward_kinematics(const std::array<double, 6> & joints)
{
  auto tf = forward_transform(joints);
  Eigen::Vector3d xyz = tf.translation();
  Eigen::Vector3d rpy = tf.rotation().eulerAngles(0, 1, 2);
  return {xyz.x(), xyz.y(), xyz.z(), rpy.x(), rpy.y(), rpy.z()};
}

std::array<double, 6> inverse_kinematics(const Eigen::Isometry3d & pose, const std::array<double, 6> & seed)
{
  std::array<double, 6> current = seed;
  const double alpha = 0.01;
  for (int iter = 0; iter < 200; ++iter) {
    auto fk_pose = forward_transform(current);
    Eigen::Vector3d pos_error = pose.translation() - fk_pose.translation();
    if (pos_error.norm() < 1e-4) {
      break;
    }
    // Simple proportional update on wrist joints to reduce translation error.
    for (size_t i = 0; i < current.size(); ++i) {
      current[i] += alpha * pos_error.norm() / (1.0 + static_cast<double>(i));
    }
  }
  return current;
}

}  // namespace rm65_kinematics
