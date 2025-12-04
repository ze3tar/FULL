#!/usr/bin/env python3
"""
Bridge between Cartesian APF-RRT planner and ROS/MoveIt
This converts Cartesian waypoints to joint trajectories for the RM65-6F manipulator
"""

import numpy as np
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, Point, PoseStamped
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Path
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class CartesianToMoveItBridge:
    def __init__(self):
                           
        moveit_commander.roscpp_initialize([])
        rospy.init_node('apf_rrt_moveit_bridge', anonymous=True)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        
                                
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
                                           
            target_pose = Pose()
            target_pose.position.x = waypoint[0] / 1000.0                   
            target_pose.position.y = waypoint[1] / 1000.0
            target_pose.position.z = waypoint[2] / 1000.0
            
                                                           
            target_pose.orientation.w = 1.0
            
                                      
            self.group.set_pose_target(target_pose)
            
                              
            joint_values = self.group.get_current_joint_values()
            
                                      
            plan = self.group.plan()
            
            if isinstance(plan, tuple):               
                success, trajectory, planning_time, error_code = plan
            else:                
                success = (plan.joint_trajectory.points != [])
                trajectory = plan
            
            if success:
                                                         
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
                        
        waypoints = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)
        
        rospy.loginfo(f"Loaded {len(waypoints)} waypoints from APF-RRT planner")
        
                                
        joint_trajectory = self.cartesian_path_to_joints(waypoints)
        
        if joint_trajectory is None:
            rospy.logerr("Failed to convert Cartesian path to joint trajectory")
            return False
        
                                
        rospy.loginfo("Executing trajectory...")
        
                                
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
        
                                                 
        pose_array = []
        for wp in waypoints:
            pose = Pose()
            pose.position.x = wp[0] / 1000.0
            pose.position.y = wp[1] / 1000.0
            pose.position.z = wp[2] / 1000.0
            pose.orientation.w = 1.0
            pose_array.append(pose)
        
                                                               
        (plan, fraction) = self.group.compute_cartesian_path(
            pose_array,
            0.01,                 
            0.0                    
        )
        
        rospy.loginfo(f"Visualizing path (achieved {fraction*100}% of path)")
        
                         
        self.group.execute(plan, wait=False)


class APFRRT_ROSBridge:
    """Publish APF-RRT paths to ROS topics and MoveIt."""

    def __init__(self, move_group: str = "manipulator") -> None:
        moveit_commander.roscpp_initialize([])
        if not rospy.core.is_initialized():
            rospy.init_node("apf_rrt_ros_bridge", anonymous=True)

        self.group = moveit_commander.MoveGroupCommander(move_group)
        self.path_pub = rospy.Publisher("/apf_rrt/path", Path, queue_size=10)

    def publish_path(self, path_points):
        import rospy as _rospy

        path_msg = Path()
        path_msg.header.stamp = _rospy.Time.now()
        path_msg.header.frame_id = "map"

        poses = []
        for x, y in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = _rospy.Time.now()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        path_msg.poses = poses
        self.path_pub.publish(path_msg)
        _rospy.loginfo(f"Published path with {len(poses)} poses to /apf_rrt/path")

    def send_to_moveit(self, path_points):
        joint_traj = JointTrajectory()
        joint_traj.joint_names = self.group.get_active_joints()

        base_positions = self.group.get_current_joint_values()
        for idx, (x, y) in enumerate(path_points):
            point = JointTrajectoryPoint()
            positions = list(base_positions)
            if positions:
                positions[0] = float(x)
                if len(positions) > 1:
                    positions[1] = float(y)
            point.positions = positions
            point.time_from_start = rospy.Duration.from_sec(0.5 * (idx + 1))
            joint_traj.points.append(point)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj
        self.group.execute(robot_traj, wait=True)


def main():
    """
    Example usage of the bridge
    """
    bridge = CartesianToMoveItBridge()
    
                                                            
                                                                 
              
                                                             
    
                                                
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
