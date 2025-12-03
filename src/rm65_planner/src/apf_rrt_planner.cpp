#include "rm65_planner/apf_rrt_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

namespace rm65_planner {
namespace {

struct LinkGeom {
  double radius;
  double length;
};

constexpr std::array<LinkGeom, 6> kLinkGeometries = {{{0.04, 0.25}, {0.04, 0.30}, {0.04, 0.25},
                                                      {0.035, 0.20}, {0.03, 0.15}, {0.03, 0.12}}};

Eigen::Isometry3d makeTransform(const Eigen::Vector3d& t, const Eigen::Vector3d& rpy) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translate(t);
  Eigen::AngleAxisd roll(rpy.x(), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch(rpy.y(), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yaw(rpy.z(), Eigen::Vector3d::UnitZ());
  T.rotate(yaw * pitch * roll);
  return T;
}

std::vector<Eigen::Vector3d> computeFramePositions(const Eigen::Matrix<double, 6, 1>& q) {
  std::vector<Eigen::Vector3d> frames;
  frames.reserve(7);
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  frames.push_back(T.translation());

  T = T * makeTransform({0.0, 0.0, 0.2405}, {0.0, 0.0, 0.0});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(0), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, 0.0, 0.0}, {1.5708, -1.5708, 0.0});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(1), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.256, 0.0, 0.0}, {0.0, 0.0, 1.5708});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(2), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, -0.21, 0.0}, {1.5708, 0.0, 0.0});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(3), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, 0.0, 0.0}, {-1.5708, 0.0, 0.0});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(4), Eigen::Vector3d::UnitZ()));

  T = T * makeTransform({0.0, -0.1725, 0.0}, {1.5708, 0.0, 0.0});
  frames.push_back(T.translation());
  T.rotate(Eigen::AngleAxisd(q(5), Eigen::Vector3d::UnitZ()));

  return frames;
}

double pointSegmentDistance(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const Eigen::Vector3d ab = b - a;
  const double denom = ab.squaredNorm();
  if (denom < 1e-9) {
    return (p - a).norm();
  }
  const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / denom));
  const Eigen::Vector3d proj = a + t * ab;
  return (p - proj).norm();
}

double segmentDistance(const Eigen::Vector3d& a0, const Eigen::Vector3d& a1, const Eigen::Vector3d& b0,
                       const Eigen::Vector3d& b1) {
  const Eigen::Vector3d u = a1 - a0;
  const Eigen::Vector3d v = b1 - b0;
  const Eigen::Vector3d w0 = a0 - b0;
  const double a = u.dot(u);
  const double b = u.dot(v);
  const double c = v.dot(v);
  const double d = u.dot(w0);
  const double e = v.dot(w0);
  const double denom = a * c - b * b;
  double sc, sN, sD = denom;
  double tc, tN, tD = denom;

  const double kEpsilon = 1e-9;
  if (denom < kEpsilon) {
    sN = 0.0;
    sD = 1.0;
    tN = e;
    tD = c;
  } else {
    sN = (b * e - c * d);
    tN = (a * e - b * d);
    if (sN < 0.0) {
      sN = 0.0;
      tN = e;
      tD = c;
    } else if (sN > sD) {
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0.0) {
    tN = 0.0;
    if (-d < 0.0) {
      sN = 0.0;
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {
    tN = tD;
    if (-d + b < 0.0) {
      sN = 0;
    } else if (-d + b > a) {
      sN = sD;
    } else {
      sN = (-d + b);
      sD = a;
    }
  }

  sc = (std::abs(sN) < kEpsilon ? 0.0 : sN / sD);
  tc = (std::abs(tN) < kEpsilon ? 0.0 : tN / tD);

  Eigen::Vector3d dp = w0 + sc * u - tc * v;
  return dp.norm();
}

}  // namespace

APFRRTPlanner::APFRRTPlanner() : step_size_(0.05), goal_bias_(0.2), max_iterations_(2000) {
  q_min_ << -3.107, -2.269, -2.356, -3.107, -2.234, -6.283;
  q_max_ << 3.107, 2.269, 2.356, 3.107, 2.234, 6.283;
}

void APFRRTPlanner::setJointLimits(const Eigen::Matrix<double, 6, 1>& q_min,
                                   const Eigen::Matrix<double, 6, 1>& q_max) {
  q_min_ = q_min;
  q_max_ = q_max;
}

void APFRRTPlanner::setObstacles(const std::vector<Sphere>& spheres,
                                 const std::vector<Cylinder>& cylinders) {
  spheres_ = spheres;
  cylinders_ = cylinders;
}

Eigen::Matrix<double, 6, 1> APFRRTPlanner::sampleConfig(const Eigen::Matrix<double, 6, 1>& q_goal) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  if (unit(gen) < goal_bias_) {
    return q_goal;
  }

  Eigen::Matrix<double, 6, 1> q;
  for (int i = 0; i < 6; ++i) {
    std::uniform_real_distribution<double> dist(q_min_(i), q_max_(i));
    q(i) = dist(gen);
  }

  // Potential-based attraction toward the goal
  if (unit(gen) < 0.3) {
    double blend = 0.35 + 0.25 * unit(gen);
    q = q + blend * (q_goal - q);
    for (int i = 0; i < 6; ++i) {
      q(i) = std::min(std::max(q(i), q_min_(i)), q_max_(i));
    }
  }
  return q;
}

int APFRRTPlanner::nearestNeighbor(const Eigen::Matrix<double, 6, 1>& q_rand) {
  int best_index = -1;
  double best_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < tree_.size(); ++i) {
    double d = (tree_[i].q - q_rand).norm();
    if (d < best_dist) {
      best_dist = d;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

Eigen::Matrix<double, 6, 1> APFRRTPlanner::steer(const Eigen::Matrix<double, 6, 1>& q_near,
                                                 const Eigen::Matrix<double, 6, 1>& q_rand) {
  Eigen::Matrix<double, 6, 1> dir = q_rand - q_near;
  double d = dir.norm();
  if (d <= step_size_) {
    Eigen::Matrix<double, 6, 1> q = q_rand;
    for (int i = 0; i < 6; ++i) {
      q(i) = std::min(std::max(q(i), q_min_(i)), q_max_(i));
    }
    return q;
  }
  Eigen::Matrix<double, 6, 1> q = q_near + step_size_ * dir / d;
  for (int i = 0; i < 6; ++i) {
    q(i) = std::min(std::max(q(i), q_min_(i)), q_max_(i));
  }
  return q;
}

double APFRRTPlanner::potentialCost(const Eigen::Matrix<double, 6, 1>& q,
                                    const Eigen::Matrix<double, 6, 1>& q_goal) {
  const double k_att = 1.0;
  const double k_rep = 0.05;
  const double d0 = 0.2;

  Eigen::Vector3d p = rm65_kinematics::forward_transform(q).translation();
  Eigen::Vector3d p_goal = rm65_kinematics::forward_transform(q_goal).translation();
  double U_att = 0.5 * k_att * (p - p_goal).squaredNorm();

  double U_rep = 0.0;
  for (const auto& s : spheres_) {
    double d = (p - s.center).norm();
    if (d < d0) {
      U_rep += 0.5 * k_rep * std::pow((1.0 / d) - (1.0 / d0), 2);
    }
  }
  for (const auto& c : cylinders_) {
    Eigen::Vector3d rel = p - c.base;
    double proj = rel.dot(c.axis);
    proj = std::min(std::max(proj, 0.0), c.height);
    Eigen::Vector3d closest = c.base + proj * c.axis;
    double d = (p - closest).norm() - c.radius;
    if (d < d0) {
      U_rep += 0.5 * k_rep * std::pow((1.0 / d) - (1.0 / d0), 2);
    }
  }

  return U_att + U_rep;
}

bool APFRRTPlanner::inCollision(const Eigen::Matrix<double, 6, 1>& q) {
  const auto frames = computeFramePositions(q);
  const size_t link_count = std::min(kLinkGeometries.size(), frames.size() - 1);
  for (size_t i = 0; i < link_count; ++i) {
    const Eigen::Vector3d& a = frames[i];
    const Eigen::Vector3d& b = frames[i + 1];
    const double link_radius = kLinkGeometries[i].radius;

    for (const auto& s : spheres_) {
      double dist = pointSegmentDistance(s.center, a, b);
      if (dist < s.radius + link_radius) {
        return true;
      }
    }

    for (const auto& c : cylinders_) {
      Eigen::Vector3d c0 = c.base;
      Eigen::Vector3d c1 = c.base + c.axis * c.height;
      double dist = segmentDistance(a, b, c0, c1);
      if (dist < c.radius + link_radius) {
        return true;
      }
    }
  }

  // Self-collision: check non-adjacent link segments against each other as capsules.
  for (size_t i = 0; i < link_count; ++i) {
    for (size_t j = i + 2; j < link_count; ++j) {  // skip same and adjacent links
      double dist = segmentDistance(frames[i], frames[i + 1], frames[j], frames[j + 1]);
      double clearance = kLinkGeometries[i].radius + kLinkGeometries[j].radius;
      if (dist < clearance) {
        return true;
      }
    }
  }
  return false;
}

bool APFRRTPlanner::reachedGoal(const Eigen::Matrix<double, 6, 1>& q,
                                const Eigen::Matrix<double, 6, 1>& q_goal) {
  const double joint_tol = 0.05;
  const double cart_tol = 0.01;
  if ((q - q_goal).norm() < joint_tol) {
    return true;
  }
  Eigen::Vector3d p = rm65_kinematics::forward_transform(q).translation();
  Eigen::Vector3d p_goal = rm65_kinematics::forward_transform(q_goal).translation();
  return (p - p_goal).norm() < cart_tol;
}

void APFRRTPlanner::backtrackPath(int goal_index, std::vector<Eigen::Matrix<double, 6, 1>>& path_out) {
  std::vector<Eigen::Matrix<double, 6, 1>> tmp;
  int idx = goal_index;
  while (idx >= 0) {
    tmp.push_back(tree_[idx].q);
    idx = tree_[idx].parent_index;
  }
  std::reverse(tmp.begin(), tmp.end());
  path_out = std::move(tmp);
}

bool APFRRTPlanner::plan(const Eigen::Matrix<double, 6, 1>& q_start,
                         const Eigen::Matrix<double, 6, 1>& q_goal,
                         std::vector<Eigen::Matrix<double, 6, 1>>& path_out) {
  tree_.clear();
  tree_.push_back({q_start, -1, 0.0});

  int goal_index = -1;
  for (int iter = 0; iter < max_iterations_; ++iter) {
    Eigen::Matrix<double, 6, 1> q_rand = sampleConfig(q_goal);
    int nn = nearestNeighbor(q_rand);
    if (nn < 0) {
      continue;
    }
    Eigen::Matrix<double, 6, 1> q_new = steer(tree_[nn].q, q_rand);
    if (inCollision(q_new)) {
      continue;
    }
    double cost = tree_[nn].cost + (q_new - tree_[nn].q).norm() +
                  0.1 * potentialCost(q_new, q_goal);
    tree_.push_back({q_new, nn, cost});
    int new_index = static_cast<int>(tree_.size()) - 1;
    if (reachedGoal(q_new, q_goal)) {
      goal_index = new_index;
      break;
    }
  }

  if (goal_index < 0) {
    return false;
  }

  backtrackPath(goal_index, path_out);
  return true;
}

}  // namespace rm65_planner
