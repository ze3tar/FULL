#include "rm65_kinematics/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace rm65_kinematics {
namespace {
Eigen::Isometry3d makeTransform(const Eigen::Vector3d& t, const Eigen::Vector3d& rpy) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translate(t);
  Eigen::AngleAxisd roll(rpy.x(), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch(rpy.y(), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yaw(rpy.z(), Eigen::Vector3d::UnitZ());
  T.rotate(yaw * pitch * roll);
  return T;
}

constexpr std::array<double, 6> kMinLimits = {-3.107, -2.269, -2.356, -3.107, -2.234, -6.283};
constexpr std::array<double, 6> kMaxLimits = {3.107, 2.269, 2.356, 3.107, 2.234, 6.283};

}  // namespace

Eigen::Isometry3d forward_transform(const Eigen::Matrix<double, 6, 1>& q) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

  T = T * makeTransform({0.0, 0.0, 0.2405}, {0.0, 0.0, 0.0});
  T.rotate(Eigen::AngleAxisd(q(0), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, 0.0, 0.0}, {1.5708, -1.5708, 0.0});
  T.rotate(Eigen::AngleAxisd(q(1), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.256, 0.0, 0.0}, {0.0, 0.0, 1.5708});
  T.rotate(Eigen::AngleAxisd(q(2), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, -0.21, 0.0}, {1.5708, 0.0, 0.0});
  T.rotate(Eigen::AngleAxisd(q(3), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, 0.0, 0.0}, {-1.5708, 0.0, 0.0});
  T.rotate(Eigen::AngleAxisd(q(4), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, -0.1725, 0.0}, {1.5708, 0.0, 0.0});
  T.rotate(Eigen::AngleAxisd(q(5), Eigen::Vector3d::UnitZ()));

  return T;
}

Eigen::Matrix<double, 6, 1> forward_kinematics(const Eigen::Matrix<double, 6, 1>& q) {
  Eigen::Isometry3d T = forward_transform(q);
  Eigen::Matrix<double, 6, 1> result;
  result.template head<3>() = T.translation();
  Eigen::Vector3d rpy = T.rotation().eulerAngles(2, 1, 0);
  result(3) = rpy.z();
  result(4) = rpy.y();
  result(5) = rpy.x();
  return result;
}

bool inverse_kinematics(const Eigen::Matrix<double, 6, 1>& target,
                        Eigen::Matrix<double, 6, 1>& q_out,
                        const Eigen::Matrix<double, 6, 1>* seed) {
  Eigen::Matrix<double, 6, 1> q = seed ? *seed : Eigen::Matrix<double, 6, 1>::Zero();
  const int max_iters = 200;
  const double tol = 1e-3;
  const double lambda = 1e-3;
  const double step = 1e-2;

  for (int iter = 0; iter < max_iters; ++iter) {
    Eigen::Matrix<double, 6, 1> fk = forward_kinematics(q);
    Eigen::Matrix<double, 6, 1> error = target - fk;

    for (int i = 3; i < 6; ++i) {
      if (error(i) > M_PI)
        error(i) -= 2 * M_PI;
      else if (error(i) < -M_PI)
        error(i) += 2 * M_PI;
    }

    if (error.norm() < tol) {
      q_out = q;
      return true;
    }

    Eigen::Matrix<double, 6, 6> J;
    const double h = 1e-6;
    for (int i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> q_plus = q;
      q_plus(i) += h;
      Eigen::Matrix<double, 6, 1> f_plus = forward_kinematics(q_plus);
      J.col(i) = (f_plus - fk) / h;
    }

    Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
    Eigen::Matrix<double, 6, 6> damping = lambda * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 1> delta = J.transpose() * (JJt + damping).inverse() * error * step;

    q += delta;
    for (int i = 0; i < 6; ++i) {
      q(i) = std::min(std::max(q(i), kMinLimits[i]), kMaxLimits[i]);
    }
  }

  return false;
}

}  // namespace rm65_kinematics
