#pragma once

#include <array>
#include <Eigen/Geometry>

namespace rm65_kinematics
{

struct DHParam
{
  double a;
  double alpha;
  double d;
};

/**
 * @brief Forward kinematics using provided DH parameters.
 */
Eigen::Isometry3d forward_transform(const std::array<double, 6> & joints);

/**
 * @brief Convenience that returns end-effector translation (x,y,z,roll,pitch,yaw).
 */
std::array<double, 6> forward_kinematics(const std::array<double, 6> & joints);

/**
 * @brief Very small numerical IK stub that iteratively searches for pose.
 */
std::array<double, 6> inverse_kinematics(const Eigen::Isometry3d & pose, const std::array<double,6> & seed = {0,0,0,0,0,0});

}  // namespace rm65_kinematics
