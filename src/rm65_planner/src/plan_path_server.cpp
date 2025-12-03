#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "full_interfaces/srv/plan_path.hpp"
#include "rm65_planner/apf_rrt_planner.hpp"

namespace rm65_planner {
namespace {
constexpr std::array<const char*, 6> kJointNames = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

Eigen::Matrix<double, 6, 1> extractState(const sensor_msgs::msg::JointState& msg) {
  Eigen::Matrix<double, 6, 1> q;
  if (msg.position.size() >= 6) {
    for (size_t i = 0; i < 6; ++i) {
      q(i) = msg.position[i];
    }
    return q;
  }
  // fallback by names
  for (size_t i = 0; i < 6; ++i) {
    auto it = std::find(msg.name.begin(), msg.name.end(), kJointNames[i]);
    if (it != msg.name.end()) {
      size_t idx = std::distance(msg.name.begin(), it);
      q(i) = msg.position.at(idx);
    } else {
      q(i) = 0.0;
    }
  }
  return q;
}

trajectory_msgs::msg::JointTrajectory buildTrajectory(const std::vector<Eigen::Matrix<double, 6, 1>>& path) {
  trajectory_msgs::msg::JointTrajectory traj;
  for (const auto* name : kJointNames) {
    traj.joint_names.push_back(name);
  }
  rclcpp::Duration step = rclcpp::Duration::from_seconds(0.1);
  for (const auto& q : path) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions.resize(6);
    pt.velocities.assign(6, 0.0);
    for (size_t i = 0; i < 6; ++i) {
      pt.positions[i] = q(i);
    }
    pt.time_from_start = step;
    traj.points.push_back(pt);
  }
  // ensure strictly increasing times
  rclcpp::Duration accum = rclcpp::Duration(0, 0);
  for (auto& pt : traj.points) {
    accum += step;
    pt.time_from_start = accum;
  }
  return traj;
}

}  // namespace

class PlanPathServer : public rclcpp::Node {
 public:
  PlanPathServer() : Node("rm65_plan_path_server") {
    using std::placeholders::_1;
    using std::placeholders::_2;
    planner_.setStepSize(0.05);
    planner_.setGoalBias(0.2);
    planner_.setMaxIterations(2000);

    Eigen::Matrix<double, 6, 1> q_min;
    Eigen::Matrix<double, 6, 1> q_max;
    q_min << -3.107, -2.269, -2.356, -3.107, -2.234, -6.283;
    q_max << 3.107, 2.269, 2.356, 3.107, 2.234, 6.283;
    planner_.setJointLimits(q_min, q_max);

    service_ = create_service<full_interfaces::srv::PlanPath>(
        "/rm65/plan_path", std::bind(&PlanPathServer::handleRequest, this, _1, _2));
  }

 private:
  void handleRequest(const std::shared_ptr<full_interfaces::srv::PlanPath::Request> request,
                     std::shared_ptr<full_interfaces::srv::PlanPath::Response> response) {
    if (request->group_name != "arm") {
      response->success = false;
      response->message = "Unsupported group_name";
      return;
    }

    Eigen::Matrix<double, 6, 1> q_start = extractState(request->start);
    Eigen::Matrix<double, 6, 1> q_goal = extractState(request->goal);

    std::vector<Eigen::Matrix<double, 6, 1>> path;
    if (!planner_.plan(q_start, q_goal, path)) {
      response->success = false;
      response->message = "Planning failed";
      return;
    }

    response->trajectory = buildTrajectory(path);
    response->success = true;
    response->message = "OK";
  }

  APFRRTPlanner planner_;
  rclcpp::Service<full_interfaces::srv::PlanPath>::SharedPtr service_;
};

}  // namespace rm65_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<rm65_planner::PlanPathServer>());
  rclcpp::shutdown();
  return 0;
}
