#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace rm65_kinematics {

Eigen::Isometry3d forward_transform(const Eigen::Matrix<double, 6, 1>& q);

Eigen::Matrix<double, 6, 1> forward_kinematics(const Eigen::Matrix<double, 6, 1>& q);

bool inverse_kinematics(const Eigen::Matrix<double, 6, 1>& target,
                        Eigen::Matrix<double, 6, 1>& q_out,
                        const Eigen::Matrix<double, 6, 1>* seed = nullptr);

}  // namespace rm65_kinematics
