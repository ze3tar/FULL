#pragma once

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "full_integration/srv/refine_trajectory.hpp"

#include "rm65_planner/apf_rrt_planner.hpp"

namespace rm65_moveit_plugin {

class APFRRTPlanningContext : public planning_interface::PlanningContext {
 public:
  APFRRTPlanningContext(const std::string& name, const std::string& group,
                        const moveit::core::RobotModelConstPtr& model,
                        const rclcpp::Node::SharedPtr& node, bool use_ppo, const std::string& ppo_service);

  bool solve(planning_interface::MotionPlanResponse& res) override;
  bool solve(planning_interface::MotionPlanDetailedResponse& res) override;
  void clear() override {}

 private:
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* jmg_;
  rm65_planner::APFRRTPlanner planner_;
  rclcpp::Node::SharedPtr node_;
  bool use_ppo_refinement_;
  std::string ppo_service_name_;
  rclcpp::Client<full_integration::srv::RefineTrajectory>::SharedPtr ppo_client_;
};

class APFRRTPlannerPlugin : public planning_interface::PlannerManager {
 public:
  APFRRTPlannerPlugin();
  bool initialize(const moveit::core::RobotModelConstPtr& model, const std::string& ns) override;
  planning_interface::PlanningContextPtr getPlanningContext(
      const planning_scene::PlanningSceneConstPtr& planning_scene,
      const planning_interface::MotionPlanRequest& req,
      moveit_msgs::msg::MoveItErrorCodes& error_code) const override;
  std::string getDescription() const override { return "APF-RRT"; }
  void getPlanningAlgorithms(std::vector<std::string>& algs) const override { algs.push_back("apf_rrt"); }

 private:
  moveit::core::RobotModelConstPtr robot_model_;
  std::string ns_;
  rclcpp::Node::SharedPtr node_;
  bool use_ppo_refinement_{false};
  std::string ppo_service_name_{"/refine_trajectory"};
  rclcpp::Client<full_integration::srv::RefineTrajectory>::SharedPtr ppo_client_;
};

}  // namespace rm65_moveit_plugin
