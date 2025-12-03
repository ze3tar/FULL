#include "rm65_moveit_plugin/apf_rrt_planner_plugin.hpp"

#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <pluginlib/class_list_macros.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <array>
#include <chrono>

namespace rm65_moveit_plugin {

APFRRTPlanningContext::APFRRTPlanningContext(const std::string& name, const std::string& group,
                                             const moveit::core::RobotModelConstPtr& model,
                                             const rclcpp::Node::SharedPtr& node, bool use_ppo,
                                             const std::string& ppo_service)
    : planning_interface::PlanningContext(name, group), robot_model_(model), node_(node),
      use_ppo_refinement_(use_ppo), ppo_service_name_(ppo_service) {
  jmg_ = robot_model_->getJointModelGroup(group);

  if (node_ && use_ppo_refinement_) {
    ppo_client_ = node_->create_client<full_integration::srv::RefineTrajectory>(ppo_service_name_);
  }

  Eigen::Matrix<double, 6, 1> q_min;
  Eigen::Matrix<double, 6, 1> q_max;
  const auto& bounds = jmg_->getActiveJointModelsBounds();
  for (size_t i = 0; i < 6; ++i) {
    q_min(i) = bounds[i][0].min_position_;
    q_max(i) = bounds[i][0].max_position_;
  }
  planner_.setJointLimits(q_min, q_max);
}

bool APFRRTPlanningContext::solve(planning_interface::MotionPlanResponse& res) {
  planning_interface::MotionPlanDetailedResponse detailed;
  if (!solve(detailed)) {
    res.error_code_ = detailed.error_code_;
    return false;
  }
  res.trajectory_ = detailed.trajectory_[0];
  res.planning_time_ = detailed.processing_time_[0];
  res.error_code_ = detailed.error_code_;
  return res.error_code_.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
}

bool APFRRTPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res) {
  if (!jmg_) {
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return false;
  }
  if (!planning_scene_) {
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return false;
  }

  const auto& req = getMotionPlanRequest();
  robot_state::RobotState start_state(robot_model_);
  if (!planning_scene_->getCurrentStateNonConst().satisfiesBounds(jmg_)) {
    planning_scene_->getCurrentStateNonConst().enforceBounds(jmg_);
  }
  start_state = planning_scene_->getCurrentState();
  if (!req.start_state.joint_state.name.empty()) {
    moveit::core::robotStateMsgToRobotState(req.start_state, start_state);
  }
  start_state.enforceBounds(jmg_);

  Eigen::Matrix<double, 6, 1> q_start;
  start_state.copyJointGroupPositions(jmg_, q_start.data());

  Eigen::Matrix<double, 6, 1> q_goal = q_start;
  if (req.goal_constraints.empty() || req.goal_constraints.front().joint_constraints.empty()) {
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::GOAL_CONSTRAINTS_VIOLATED;
    return false;
  }
  const auto& constraint = req.goal_constraints.front();
  const auto& joint_names = jmg_->getVariableNames();
  for (const auto& jc : constraint.joint_constraints) {
    auto it = std::find(joint_names.begin(), joint_names.end(), jc.joint_name);
    if (it != joint_names.end()) {
      size_t idx = std::distance(joint_names.begin(), it);
      if (idx < 6) {
        q_goal(idx) = jc.position;
      }
    }
  }

  std::vector<Eigen::Matrix<double, 6, 1>> path;
  if (!planner_.plan(q_start, q_goal, path)) {
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    return false;
  }

  trajectory_msgs::msg::JointTrajectory jt;
  static const std::array<const char*, 6> kJointNames = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
  jt.joint_names.assign(kJointNames.begin(), kJointNames.end());
  const double dt = 0.1;
  rclcpp::Duration step = rclcpp::Duration::from_seconds(dt);
  rclcpp::Duration accum = rclcpp::Duration::from_seconds(0.0);
  for (const auto& q : path) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions.resize(6);
    pt.velocities.assign(6, 0.0);
    for (size_t i = 0; i < 6; ++i) {
      pt.positions[i] = q(i);
    }
    accum += step;
    pt.time_from_start = accum.to_msg();
    jt.points.push_back(pt);
  }

  if (use_ppo_refinement_ && ppo_client_) {
    if (!ppo_client_->wait_for_service(std::chrono::seconds(2))) {
      if (node_) {
        RCLCPP_WARN(node_->get_logger(), "PPO service %s not available, skipping refinement", ppo_service_name_.c_str());
      }
    } else {
      auto request = std::make_shared<full_integration::srv::RefineTrajectory::Request>();
      request->input = jt;
      auto future = ppo_client_->async_send_request(request);
      if (rclcpp::spin_until_future_complete(node_, future, std::chrono::seconds(5)) ==
          rclcpp::FutureReturnCode::SUCCESS) {
        auto response = future.get();
        if (response->success) {
          jt = response->output;
        }
      }
    }
  }

  robot_trajectory::RobotTrajectoryPtr traj(new robot_trajectory::RobotTrajectory(robot_model_, jmg_->getName()));
  robot_state::RobotState waypoint_state(start_state);
  double time = 0.0;
  double last_time = 0.0;
  for (const auto& pt : jt.points) {
    if (pt.positions.size() >= 6) {
      waypoint_state.setJointGroupPositions(jmg_, pt.positions.data());
    }
    rclcpp::Duration dt_segment(pt.time_from_start);
    double segment_time = dt_segment.seconds();
    double delta = segment_time - last_time;
    if (delta <= 0.0) {
      delta = dt;
    }
    traj->addSuffixWayPoint(waypoint_state, delta);
    time += delta;
    last_time = segment_time;
  }

  res.trajectory_.push_back(traj);
  res.description_.push_back(use_ppo_refinement_ ? "apf_rrt ppo-refined path" : "apf_rrt path");
  res.processing_time_.push_back(time);
  res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return true;
}

APFRRTPlannerPlugin::APFRRTPlannerPlugin() : planning_interface::PlannerManager() {}

bool APFRRTPlannerPlugin::initialize(const moveit::core::RobotModelConstPtr& model, const std::string& ns) {
  robot_model_ = model;
  ns_ = ns;
  node_ = rclcpp::Node::make_shared("rm65_apf_rrt_plugin");

  try {
    const std::string share = ament_index_cpp::get_package_share_directory("rm65_moveit_plugin");
    const std::string config_path = share + "/config/plugin_settings.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    use_ppo_refinement_ = config["use_ppo_refinement"].as<bool>(false);
    ppo_service_name_ = config["ppo_service"].as<std::string>("/refine_trajectory");
  } catch (const std::exception& ex) {
    RCLCPP_WARN(node_->get_logger(), "Failed to load plugin settings: %s", ex.what());
    use_ppo_refinement_ = false;
    ppo_service_name_ = "/refine_trajectory";
  }

  if (use_ppo_refinement_) {
    ppo_client_ = node_->create_client<full_integration::srv::RefineTrajectory>(ppo_service_name_);
  }

  return true;
}

planning_interface::PlanningContextPtr APFRRTPlannerPlugin::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene, const planning_interface::MotionPlanRequest& req,
    moveit_msgs::msg::MoveItErrorCodes& error_code) const {
  auto context = std::make_shared<APFRRTPlanningContext>("apf_rrt", req.group_name, robot_model_, node_,
                                                         use_ppo_refinement_, ppo_service_name_);
  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(req);
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return context;
}

}  // namespace rm65_moveit_plugin

PLUGINLIB_EXPORT_CLASS(rm65_moveit_plugin::APFRRTPlannerPlugin, planning_interface::PlannerManager)
