#include "rm65_moveit_plugin/apf_rrt_planner_plugin.hpp"

#include <moveit/planning_interface/planning_request_adapter.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <pluginlib/class_list_macros.hpp>

namespace rm65_moveit_plugin
{

class APFRRTContext : public planning_interface::PlanningContext
{
public:
  APFRRTContext(
    const std::string & name, const std::string & group,
    const planning_scene::PlanningSceneConstPtr & scene,
    std::shared_ptr<rm65_planner::APFRRTPlanner> planner)
  : planning_interface::PlanningContext(name, group), scene_(scene), planner_(std::move(planner)) {}

  bool solve(planning_interface::MotionPlanResponse & res) override
  {
    if (!scene_) {
      return false;
    }
    std::array<double, 6> start{};
    std::array<double, 6> goal{};
    for (size_t i = 0; i < 6 && i < request_.start_state.joint_state.position.size(); ++i) {
      start[i] = request_.start_state.joint_state.position[i];
    }
    if (!request_.goal_constraints.empty() && !request_.goal_constraints[0].joint_constraints.empty()) {
      for (size_t i = 0; i < 6 && i < request_.goal_constraints[0].joint_constraints.size(); ++i) {
        goal[i] = request_.goal_constraints[0].joint_constraints[i].position;
      }
    }

    std::vector<rm65_planner::Sphere> obstacles;  // placeholder empty
    auto path = planner_->plan(start, goal, obstacles);
    robot_trajectory::RobotTrajectory trajectory(scene_->getRobotModel(), request_.group_name);
    robot_state::RobotState state(scene_->getCurrentState());
    const auto & joint_model_group = *state.getJointModelGroup(request_.group_name);
    for (const auto & q : path) {
      std::vector<double> positions(q.begin(), q.end());
      state.setJointGroupPositions(&joint_model_group, positions);
      trajectory.addSuffixWayPoint(state, 0.1);
    }
    res.trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(trajectory);
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
  }

  bool solve(planning_interface::MotionPlanDetailedResponse & res) override
  {
    planning_interface::MotionPlanResponse response;
    bool ok = solve(response);
    if (ok) {
      res.trajectory_.push_back(response.trajectory);
    }
    res.error_code_ = response.error_code;
    return ok;
  }

private:
  planning_scene::PlanningSceneConstPtr scene_;
  std::shared_ptr<rm65_planner::APFRRTPlanner> planner_;
};

APFRRTPlannerPlugin::APFRRTPlannerPlugin() = default;

bool APFRRTPlannerPlugin::initialize(
  const robot_model::RobotModelConstPtr & model, const std::string & ns)
{
  robot_model_ = model;
  planner_ = std::make_shared<rm65_planner::APFRRTPlanner>();
  return static_cast<bool>(model);
}

planning_interface::PlanningContextPtr APFRRTPlannerPlugin::getPlanningContext(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & req, moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  auto context = std::make_shared<APFRRTContext>("apf_rrt", req.group_name, planning_scene, planner_);
  context->setMotionPlanRequest(req);
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return context;
}

}  // namespace rm65_moveit_plugin

PLUGINLIB_EXPORT_CLASS(rm65_moveit_plugin::APFRRTPlannerPlugin, planning_interface::PlannerManager)
