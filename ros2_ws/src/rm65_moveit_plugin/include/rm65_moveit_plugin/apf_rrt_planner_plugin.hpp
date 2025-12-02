#pragma once

#include <memory>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <rm65_planner/apf_rrt_planner.hpp>

namespace rm65_moveit_plugin
{

class APFRRTPlannerPlugin : public planning_interface::PlannerManager
{
public:
  APFRRTPlannerPlugin();

  bool initialize(
    const robot_model::RobotModelConstPtr & model, const std::string & ns) override;

  planning_interface::PlanningContextPtr getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req, moveit_msgs::msg::MoveItErrorCodes & error_code) const override;

private:
  std::shared_ptr<rm65_planner::APFRRTPlanner> planner_;
};

}  // namespace rm65_moveit_plugin
