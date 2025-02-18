#!/usr/bin/env python3

import actionlib
import math
import numpy as np
import rospy
import tf

from geometry_msgs.msg import Twist, PointStamped, Point
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA 
from sensor_msgs.msg import JointState

from ho_planning_project.msg import DwaAction, DwaFeedback, DwaResult, DwaGoal
from utils.Dynamic_window_aproach import *
from utils.State_Validity_Checker import StateValidityChecker

class ControllerNode():
    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic):
        # State Validity Checker object                                                 
        self.svc = StateValidityChecker()
        # Current robot SE2 pose [x, y, yaw], None if unknown            
        self.current_pose = None
        # Current velocities v and W
        self.current_velocities = None
        # Distance from robot to close obstacles for DWA, 2 [m] according to second meeting
        self.w_dist = 2
        # Adimissible distance between robot and goal [m]
        self.dist_threshold = 0.1 
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        # linear and angular velocities
        self.angular_velocity = 0
        self.linear_velocity = 0

        self.robot = Robot()
        self.obs = None
        self.goal = None
        self.x = np.array([0.0, 0.0, math.pi*0 , 0.0, 0.0])

        self.left_wheel_flag = False
        self.right_wheel_flag = False

        # for PID
        self.prev_error_dist = 0
        self.prev_error_angle = 0


        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        # Publisher for visualizing the path to with rviz
        # self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)
        self.traject_pub = rospy.Publisher('~current_trajectory', Marker, queue_size=1)
        self.pre_traj_pub = rospy.Publisher('~predicted_trajectory', Marker, queue_size=1)
        self.close_obs_pub = rospy.Publisher('~close_obstacles', Marker, queue_size=1)
        self.point_pub = rospy.Publisher('~goal_point', PointStamped, queue_size=1)
        self.list_pos_paths = rospy.Publisher('~list_pos_paths', Marker, queue_size=1)

        # SUBSCRIBERS
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, callback= self.get_gridmap)  
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry ,callback= self.get_odom)
        self.encoder_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.encoder_callback)
        
        # Action Server
        self.a_server = actionlib.SimpleActionServer(
            "dwa_act_server", DwaAction, execute_cb=self.execute_cb, auto_start=False)
        self.a_server.start()

    ############
    # Display functions for rviz
    ############

    
    def publish_point(self, point):
        p = PointStamped()
        p.header.frame_id = 'world_ned'# uncomment to use in stonefish
        #p.header.frame_id = 'odom' # uncomment to use in gazebo
        p.header.stamp = rospy.Time.now()

        p.point.x = point[0]
        p.point.y = point[1]
        p.point.z = 0.0
        
        self.point_pub.publish(p)


    def publish_pred_traj(self, predicted_trajectory):
        m = Marker()
        m.header.frame_id = 'world_ned' # uncomment to use in stonefish
        #m.header.frame_id = 'odom' # uncomment to use in gazebo
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.ns = 'predicted_trajectory'
        m.action = Marker.DELETE
        m.lifetime = rospy.Duration(0)
        self.pre_traj_pub.publish(m)

        m.action = Marker.ADD
        m.scale.x = 0.07
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
            
        for n in predicted_trajectory:
            p = Point()
            p.x = n[0]
            p.y = n[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_red)
            
        self.pre_traj_pub.publish(m)


   
    def publish_trajectory(self, trajectory):
        m = Marker()
        m.header.frame_id = 'world_ned' # uncomment to use in stonefish
        #m.header.frame_id = 'odom' # uncomment to use in gazebo
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.ns = 'current_trajectory'
        m.action = Marker.DELETE
        m.lifetime = rospy.Duration(0)
        self.traject_pub.publish(m)

        m.action = Marker.ADD
        m.scale.x = 0.1
        m.scale.y = 0.0
        m.scale.z = 0.0
            
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
            
        color_white = ColorRGBA()
        color_white.r = 1
        color_white.g = 1
        color_white.b = 1
        color_white.a = 1
            
        for n in trajectory:
            p = Point()
            p.x = n[0]
            p.y = n[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_white)
            
        self.traject_pub.publish(m)
 
 
    def view_close_obs(self, obs):
        m = Marker()
        m.header.frame_id = 'world_ned' # uncomment to use in stonefish
        #m.header.frame_id = 'odom' # uncomment to use in gazebo
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.POINTS
        m.ns = 'view_close_obs'
        m.action = Marker.DELETE
        m.lifetime = rospy.Duration(0)
        self.close_obs_pub.publish(m)

        m.action = Marker.ADD
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
            
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
            
        color_green = ColorRGBA()
        color_green.r = 0
        color_green.g = 1
        color_green.b = 0
        color_green.a = 1
            
        for n in obs:
            p = Point()
            p.x = n[0]
            p.y = n[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_green)
            
        self.close_obs_pub.publish(m)


    def publish_pos_paths(self, list_possible_paths):
        m = Marker()
        m.header.frame_id = 'world_ned' # uncomment to use in stonefish
        #m.header.frame_id = 'odom' # uncomment to use in gazebo
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.POINTS
        m.ns = 'list_possible_paths'
        m.action = Marker.DELETE
        m.lifetime = rospy.Duration(0)
        self.list_pos_paths.publish(m)

        m.action = Marker.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0.05
            
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
            
        color_green = ColorRGBA()
        color_green.r = 0
        color_green.g = 1
        color_green.b = 1
        color_green.a = 1
            
        for n in list_possible_paths:
            p = Point()
            p.x = n[0,0]
            p.y = n[0,1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_green)
            
        self.list_pos_paths.publish(m)



    def execute_cb(self,goal):
        ("Goal received.")
        v, w = 0, 0
        self.goal = np.array(goal.goal_pos)
        self.publish_point(self.goal)
        initial_time = rospy.Time.now()
        success = True
        # robot moving feedback message
        robot_moving_fdm = ''
        feedback = DwaFeedback()
        result = DwaResult()
        
        moving_robot = True
        trajectory = np.array(self.x)
        # Loop until the robot reach the goal position
        while moving_robot:
            # current_time = rospy.Time.now()
            # time_passed = current_time - initial_time
            # elapsed_time = time_passed.to_sec()  # Convert current time to seconds
            # Checking the obstacles close to the robot
            self.obs = np.array(self.create_map_section())

            if self.a_server.is_preempt_requested():
                self.__send_commnd__(0, 0) 
                self.a_server.set_preempted()
                success = False
                break
            
            # Time passed since the the goal was received
            # robot_moving_fdm = 'Time passed: ' + str(elapsed_time)
            robot_moving_fdm = 'Robot moving ...'
            # publishing the feedback
            feedback.robot_moving = robot_moving_fdm
            # result.goal_reached.append(robot_moving_fm)
            result.goal_reached = True           
            self.a_server.publish_feedback(feedback)

            #########################
            # DWA functions
            # u, predicted_trajectory,list_possible_paths = dwa_control(self.x, self.robot, self.goal, self.obs)
            # self.publish_pos_paths(list_possible_paths)
            # self.publish_pred_traj(predicted_trajectory)
            # self.x = motion(self.x, u, self.robot.dt)  # simulate robot
            # # print("next vector x: ", self.x)
            # trajectory = np.vstack((trajectory, self.x))  # store state history
            # self.publish_trajectory(trajectory)
            # v,w = self.x[3],self.x[4]
            # # print("linear and angular speed: ", v, w)
            # # check reaching goal                      
            # dist_to_goal = np.linalg.norm(np.array([self.x[0] - self.goal[0], 
            #                                         self.x[1] - self.goal[1]]))
            # if dist_to_goal <= self.robot.robot_radius + self.dist_threshold:
            #     print("Goal reached")
            #     self.__send_commnd__(0, 0)    
            #     break

            # # Publish velocity commands
            # self.__send_commnd__(v, w)

            ## PID controller
            v, w = self.move_to_point_smooth(self.goal)
            dist_to_goal = np.linalg.norm(np.array([self.x[0] - self.goal[0], 
                                                    self.x[1] - self.goal[1]]))
            # Publish velocity commands
            self.__send_commnd__(v, w)        
            if dist_to_goal <= self.dist_threshold:
                print("Goal reached")
                self.__send_commnd__(0, 0)    
                break
            
            # Publish velocity commands
            self.__send_commnd__(v, w)
            
    
        # Publishing the result
        if success:
            self.a_server.set_succeeded(result)
    
    
    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])

        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        self.x = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw
                           , self.linear_velocity, self.angular_velocity])
        self.current_velocities = np.array([self.linear_velocity, self.angular_velocity])
    
    def encoder_callback(self,states_msg):
        # assign the wheel encoder values
        wheel_radius = 0.035 # m
        names_list = list(states_msg.name)
        if "turtlebot/kobuki/wheel_left_joint" in names_list:
            left_index = names_list.index("turtlebot/kobuki/wheel_left_joint")
            # set left wheel
            self.wl = states_msg.velocity[left_index]
            self.left_wheel_flag = True

        if "turtlebot/kobuki/wheel_right_joint" in names_list:
            right_index = names_list.index("turtlebot/kobuki/wheel_right_joint")
            # set right wheel
            self.wr = states_msg.velocity[right_index]
            self.right_wheel_flag = True
        
        if self.right_wheel_flag and self.left_wheel_flag:
            # convert angular to linear velocity
            vl = self.wl * wheel_radius
            vr = self.wr * wheel_radius
            
            # linear and angular displacement of robot
            v = (vl+vr)/2
            w = (vl-vr)/0.235

            self.linear_velocity, self.angular_velocity =v,w
            # Reset the flags
            self.right_wheel_flag = False
            self.left_wheel_flag = False

            # Save current time
            # self.current_time = states_msg.header.stamp

    # Map callback: Gets the latest occupancy map published by Octomap server and update 
    # the state validity checker
    def get_gridmap(self, gridmap):
      
        # To avoid map update too often (change value '1' if necessary)
        # if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:
        if (not self.svc.there_is_map) or ((gridmap.header.stamp - self.last_map_time).to_sec() > 2):
            self.last_map_time = gridmap.header.stamp

        # Update State Validity Checker
        env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
        self.map_section = env
        origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
        self.svc.set(env, gridmap.info.resolution, origin)


    # Generate a square section around the robot to consider only the close obstacles
    def create_map_section(self): # Returns the list of obstacles position 
        x,y = self.current_pose[0], self.current_pose[1] 
        p = np.array([x,y])
        
        map_p = self.svc.__position_to_map__(p)
        map_dist = int(self.w_dist/self.svc.resolution)
        # Check the best space between obstacles with the coef
        coef_percent = 0.5 # 0.35 is the diameter of robot in m
        robot_diameter_on_cells = int(0.35*coef_percent/self.svc.resolution)
        
        obstacles = []
        
        for i in range(-map_dist,map_dist+1): # rows
            for j in range(-map_dist,map_dist+1): # columns
                point = map_p + np.array([i,j])
                try:
                    map_point = self.map_section[point[0],point[1]]
                    
                    # Consider only the contour obstacles ( close to an empty cell)
                    if (map_point == 100 and 
                        ((self.map_section[point[0]-1,point[1]] == 0 or 
                          self.map_section[point[0],point[1]-1] == 0 or
                          self.map_section[point[0]+1,point[1]] == 0 or 
                          self.map_section[point[0],point[1]+1] == 0))):
                        
                        if len(obstacles) == 0: 
                            obstacles.append([point[0],point[1]])
                        # Consider only the obstacle with a space between them equal to the robot diameter
                        if np.linalg.norm(np.array([point[0] - obstacles[-1][0],
                                                    point[1] - obstacles[-1][1]])) >= robot_diameter_on_cells:
                            obstacles.append([point[0],point[1]]) 
                except:
                    None

        # converting cell points into positions
        obs = obstacles
        obstacles = []
        for o in obs:
            pos = self.svc.__map_to_position__(o)
            obstacles.append(pos)
        # List of obstacles in real world position   
        self.view_close_obs(obstacles) 
        return obstacles


    # PUBLISHER HELPERS
    # Transform linear and angular velocity (v, w) into a Twist message and publish it
    def __send_commnd__(self, v, w):
        cmd = Twist()
        cmd.linear.x = np.clip(v, 0.02, self.robot.max_speed)
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        # cmd.angular.z = -w # np.clip(w, -self.robot.max_angular_speed, self.robot.max_angular_speed)
        if w > 0.05:
            w = max(0.5,w)
        elif w< -0.05:
            w = min(-0.5,w)

        cmd.angular.z = -w
            
        self.cmd_pub.publish(cmd)
    
    # Normal controller
    def wrap_angle(self, angle):
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    def move_to_point(self, next_p, Kv=0.5, Kw=0.5):
        d = ((self.current_pose[0]-next_p[0])**2 + (self.current_pose[1]-next_p[1])**2) ** 0.5
        points_angle = math.atan2(next_p[1] - self.current_pose[1], next_p[0] - self.current_pose[0])
        robot_angle = self.current_pose[2] #Odometry yaw
        dif_ang = points_angle - robot_angle
        angle = self.wrap_angle(dif_ang)
        w = angle * Kw
        # using a threshold angle difference of 0.05 rad
        v = 0.0 if abs(angle) > 0.05 else Kv * d
        return v, w
    
    def move_to_point_smooth(self,goal, Kp=0.2, Ki=0.1, Kd=0.2):
        # Compute distance and angle to goal
        dx = goal[0] - self.current_pose[0]
        dy = goal[1] - self.current_pose[1]
        dist = math.sqrt(dx**2 + dy**2)
        angle = wrap_angle(math.atan2(dy, dx) - self.current_pose[2])

        # Compute errors
        error_dist = dist
        error_angle = angle

        # Compute PID terms
        error_dist_deriv = (error_dist - self.prev_error_dist)
        error_angle_deriv = (error_angle - self.prev_error_angle)
        error_dist_integral = (error_dist + self.prev_error_dist)
        error_angle_integral = (error_angle + self.prev_error_angle)

        v = Kp * error_dist + Ki * error_dist_integral + Kd * error_dist_deriv
        w = Kp * error_angle + Ki * error_angle_integral + Kd * error_angle_deriv

        # Update previous errors
        self.prev_error_dist = error_dist
        self.prev_error_angle = error_angle

        # # Limit angular velocity to avoid overshooting
        if abs(angle) > 0.2:
            v = 0

        # Limit linear velocity
        if v >= 0.3:
            v = 0.3

        return v, w

if __name__ == "__main__":
    rospy.init_node("controller_node")
    # Set up a timer to call the callback function every second
    server = ControllerNode('/projected_map', '/odom', '/cmd_vel')
    rospy.spin()