#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "rm65_kinematics/kinematics.hpp"

namespace rm65_planner {

struct Sphere {
  Eigen::Vector3d center;
  double radius;
};

struct Cylinder {
  Eigen::Vector3d base;
  Eigen::Vector3d axis;  // normalized
  double radius;
  double height;
};

class APFRRTPlanner {
 public:
  APFRRTPlanner();

  struct Node {
    Eigen::Matrix<double, 6, 1> q;
    int parent_index;
    double cost;
  };

  void setJointLimits(const Eigen::Matrix<double, 6, 1>& q_min,
                      const Eigen::Matrix<double, 6, 1>& q_max);
  void setObstacles(const std::vector<Sphere>& spheres,
                    const std::vector<Cylinder>& cylinders);

  bool plan(const Eigen::Matrix<double, 6, 1>& q_start,
            const Eigen::Matrix<double, 6, 1>& q_goal,
            std::vector<Eigen::Matrix<double, 6, 1>>& path_out);

  void setStepSize(double step) { step_size_ = step; }
  void setGoalBias(double bias) { goal_bias_ = bias; }
  void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }

 private:
  Eigen::Matrix<double, 6, 1> q_min_;
  Eigen::Matrix<double, 6, 1> q_max_;

  std::vector<Node> tree_;
  std::vector<Sphere> spheres_;
  std::vector<Cylinder> cylinders_;

  double step_size_;
  double goal_bias_;
  int max_iterations_;

  Eigen::Matrix<double, 6, 1> sampleConfig(const Eigen::Matrix<double, 6, 1>& q_goal);
  int nearestNeighbor(const Eigen::Matrix<double, 6, 1>& q_rand);
  Eigen::Matrix<double, 6, 1> steer(const Eigen::Matrix<double, 6, 1>& q_near,
                                    const Eigen::Matrix<double, 6, 1>& q_rand);
  double potentialCost(const Eigen::Matrix<double, 6, 1>& q,
                       const Eigen::Matrix<double, 6, 1>& q_goal);
  bool inCollision(const Eigen::Matrix<double, 6, 1>& q);
  bool reachedGoal(const Eigen::Matrix<double, 6, 1>& q,
                   const Eigen::Matrix<double, 6, 1>& q_goal);
  void backtrackPath(int goal_index, std::vector<Eigen::Matrix<double, 6, 1>>& path_out);
};

}  // namespace rm65_planner
