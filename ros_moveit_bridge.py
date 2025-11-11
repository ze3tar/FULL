#!/usr/bin/env python3
"""
Bridge between Cartesian APF-RRT planner and ROS/MoveIt
This converts Cartesian waypoints to joint trajectories for the RM65-6F manipulator
"""

import numpy as np
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class CartesianToMoveItBridge:
    def __init__(self):
        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        rospy.init_node('apf_rrt_moveit_bridge', anonymous=True)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        
        # Set planner parameters
        self.group.set_planning_time(5.0)
        self.group.set_num_planning_attempts(10)
        
    def cartesian_path_to_joints(self, cartesian_waypoints):
        """
        Convert Cartesian waypoints from APF-RRT to joint trajectories
        
        Args:
            cartesian_waypoints: List of (x, y, z) tuples from your APF-RRT
        
        Returns:
            joint_trajectory: MoveIt joint trajectory
        """
        joint_trajectory = []
        
        for i, waypoint in enumerate(cartesian_waypoints):
            # Create pose for this waypoint
            target_pose = Pose()
            target_pose.position.x = waypoint[0] / 1000.0  # Convert mm to m
            target_pose.position.y = waypoint[1] / 1000.0
            target_pose.position.z = waypoint[2] / 1000.0
            
            # Set orientation (you may need to adjust this)
            target_pose.orientation.w = 1.0
            
            # Compute IK for this pose
            self.group.set_pose_target(target_pose)
            
            # Get joint values
            joint_values = self.group.get_current_joint_values()
            
            # Try to plan to this pose
            plan = self.group.plan()
            
            if isinstance(plan, tuple):  # MoveIt 1.0+
                success, trajectory, planning_time, error_code = plan
            else:  # Older MoveIt
                success = (plan.joint_trajectory.points != [])
                trajectory = plan
            
            if success:
                # Extract joint positions from trajectory
                if trajectory.joint_trajectory.points:
                    joint_values = trajectory.joint_trajectory.points[-1].positions
                    joint_trajectory.append(joint_values)
                else:
                    rospy.logwarn(f"Could not solve IK for waypoint {i}: {waypoint}")
                    return None
            else:
                rospy.logwarn(f"Planning failed for waypoint {i}")
                return None
        
        return joint_trajectory
    
    def execute_apf_rrt_path(self, csv_file_path):
        """
        Load path from your APF-RRT CSV and execute in MoveIt
        
        Args:
            csv_file_path: Path to path_points_improved.csv
        """
        # Load waypoints
        waypoints = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)
        
        rospy.loginfo(f"Loaded {len(waypoints)} waypoints from APF-RRT planner")
        
        # Convert to joint space
        joint_trajectory = self.cartesian_path_to_joints(waypoints)
        
        if joint_trajectory is None:
            rospy.logerr("Failed to convert Cartesian path to joint trajectory")
            return False
        
        # Execute the trajectory
        rospy.loginfo("Executing trajectory...")
        
        # Move through waypoints
        for i, joint_values in enumerate(joint_trajectory):
            rospy.loginfo(f"Moving to waypoint {i+1}/{len(joint_trajectory)}")
            self.group.set_joint_value_target(joint_values)
            success = self.group.go(wait=True)
            
            if not success:
                rospy.logerr(f"Failed to reach waypoint {i}")
                return False
            
            self.group.stop()
            rospy.sleep(0.5)
        
        rospy.loginfo("Path execution complete!")
        return True
    
    def visualize_path(self, csv_file_path):
        """
        Visualize the APF-RRT path in RViz
        """
        waypoints = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)
        
        # Convert to Pose array for visualization
        pose_array = []
        for wp in waypoints:
            pose = Pose()
            pose.position.x = wp[0] / 1000.0
            pose.position.y = wp[1] / 1000.0
            pose.position.z = wp[2] / 1000.0
            pose.orientation.w = 1.0
            pose_array.append(pose)
        
        # Use MoveIt's compute_cartesian_path for visualization
        (plan, fraction) = self.group.compute_cartesian_path(
            pose_array,
            0.01,  # 1cm step size
            0.0    # jump threshold
        )
        
        rospy.loginfo(f"Visualizing path (achieved {fraction*100}% of path)")
        
        # Display in RViz
        self.group.execute(plan, wait=False)


def main():
    """
    Example usage of the bridge
    """
    bridge = CartesianToMoveItBridge()
    
    # Add obstacles to planning scene (from your simulation)
    # This should match the obstacles in your APF-RRT environment
    # Example:
    # bridge.scene.add_sphere("obstacle1", pose, radius=0.08)
    
    # Execute the path from your APF-RRT planner
    success = bridge.execute_apf_rrt_path("path_points_improved.csv")
    
    if success:
        rospy.loginfo("APF-RRT path executed successfully!")
    else:
        rospy.logerr("Failed to execute APF-RRT path")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
