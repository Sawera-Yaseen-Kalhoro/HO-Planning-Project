#!/usr/bin/python3

import numpy as np
import rospy
import tf

from geometry_msgs.msg import Point, PoseStamped, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

# from utils import State_Validity_Checker
from utils.State_Validity_Checker import StateValidityChecker
from utils.Path_planner import compute_path
from ho_planning_project.srv import PlanPath, PlanPathResponse  

class OnlinePlanner:

    def __init__(self, gridmap_topic, odom_topic, bounds, distance_threshold):


        # ATTRIBUTES
        # List of points which define the plan. None if there is no plan
        self.path = []

        # State Validity Checker object                                                 
        self.svc = StateValidityChecker(distance_threshold)

        # Current robot SE2 pose [x, y, yaw], None if unknown            
        self.current_pose = None

        # Goal where the robot has to move, None if it is not set                                                                   
        self.goal = None

        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = None

        # Dominion [min_x_y, max_x_y] in which the path planner will sample configurations                           
        self.bounds = bounds


        # PUBLISHERS
        self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)

        #SUBSCRIBERS
        # Subscribe to Grid-map topic
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap)

        # Subscribe to /odom topic
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom)
        
        #SERVICES
        # Create a ROS service for receiving path planning requests
        self.plan_path_srv = rospy.Service('/plan_path', PlanPath, self.plan_path_callback)


    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])

        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])

        # Save robot time
        self.current_time = odom.header.stamp

    
    # Map callback: Gets the latest occupancy map published by Octomap server and update the state validity checker
    def get_gridmap(self, gridmap):
      
        # To avoid map update too often (change value '1' if necessary)
        if (not self.svc.there_is_map) or (gridmap.header.stamp - self.last_map_time).to_sec() > 1:            
            self.last_map_time = gridmap.header.stamp

            # Update State Validity Checker
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(env, gridmap.info.resolution, origin)


   # Callback function for receiving path planning requests from main node
    def plan_path_callback(self, req):
        response = PlanPathResponse()

        if self.current_pose is not None and req.goal:
            rospy.loginfo("Received path planning request. Current pose: %s, Goal: %s", self.current_pose, req.goal)
            start_position = (self.current_pose[0], self.current_pose[1])
            goal_position = (req.goal[0], req.goal[1])  # Compute path expects the goal position as (x,y) tuple

            # Invalidate previous plan if available
            #self.path = []

            max_time = 10.0

            try:
                path = compute_path(start_position, goal_position, self.svc, self.bounds, max_time)
                print("Path= ",path)
            except Exception as e:
                rospy.logerr("Exception during path computation: %s", str(e))
                path = None

            if path:
                rospy.loginfo("Path successfully computed: %s", path)
                # Remove initial waypoint in the path (current pose is already reached)
                # del path[0]

                # Set response
                response.success = True
                response.planned_path.poses = [PoseStamped(pose=Pose(position=Point(x=x, y=y))) for x, y in path]
                response.planned_path.header.stamp = self.current_time
                response.planned_path.header.frame_id = 'world_ned'  # Path co-ordinates are expressed in world frame.

                # Store the path for further reference
                self.path = path

                # Publish plan marker to visualize in rviz
                self.publish_path()

            else:
                rospy.logwarn("Failed to plan path. Path computation returned None.")
                response.success = False
        else:
            rospy.logwarn("Invalid path planning request. Current pose: %s, Goal: %s", self.current_pose, req.goal)
            response.success = False

        return response


    # PUBLISHER HELPERS


    # Publish a path as a series of line markers
    def publish_path(self):
        if len(self.path) > 1:
            print("Publish path!")
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = self.current_time
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 1
            color_red.g = 0
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in self.path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_red)
            
            self.marker_pub.publish(m)
            
            
# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('turtlebot_path_planner_node')   
    node = OnlinePlanner('/projected_map', '/odom', np.array([-10.0, 10.0, -10.0, 10.0]), 0.2)
    
    # Run forever
    rospy.spin()