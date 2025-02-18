#!/usr/bin/env python3

'''
    Node to set next best viewpoint from frontier-based exploration algorithm.
'''

import numpy as np
import rospy
import tf.transformations
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


from utils.frontier_exploration import FrontierBasedExploration
from utils.State_Validity_Checker import StateValidityChecker
from ho_planning_project.srv import ExplorationTrigger, ExplorationTriggerResponse

class ExplorationNode:
    def __init__(self, boundaries=[]) -> None:
        # Initialize gridmap properties
        self.map = np.empty((0,0))
        self.map_origin = [None,None]
        self.map_resolution = None

        # Robot pose [x,y,yaw]
        self.current_pose = np.zeros(3)

        # Map boundaries to be explored (in order: [x_min, x_max, y_min, y_max])
        self.boundaries = boundaries

        # Initialize number of clusters
        self.prev_n_cluster = 0

        # Minimum distance between waypoint and robot to be considered valid
        self.min_dist = 0.5 # m

        # Publishers
        self.viewpoint_pub = rospy.Publisher('~next_viewpoint',PointStamped,queue_size=1)
        self.frontiers_pub = rospy.Publisher('~frontiers', Marker, queue_size=1)

        # Subscribers
        self.map_sub = rospy.Subscriber('/projected_map',OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Services
        self.trigger_srv = rospy.Service('/compute_next_viewpoint', ExplorationTrigger, self.trigger_handle)


    def map_callback(self,map_msg):
        # Pre-process map data from msg and store it
        self.map = np.array(map_msg.data).reshape(map_msg.info.height, map_msg.info.width).T
        self.map_origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
        self.map_resolution = map_msg.info.resolution

    def odom_callback(self,odom_msg):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom_msg.pose.pose.orientation.x, 
                                                              odom_msg.pose.pose.orientation.y,
                                                              odom_msg.pose.pose.orientation.z,
                                                              odom_msg.pose.pose.orientation.w])

        # Store current position (x, y, yaw) as a np.array in self.current_pose var.
        self.current_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])

    def trigger_handle(self,trigger_req):
        # compute next viewpoint based on current map and robot pose
        next_viewpoint = self.compute_next_viewpoint(self.map, self.map_origin, self.map_resolution, self.current_pose[:2], self.boundaries)

        resp = ExplorationTriggerResponse()

        if next_viewpoint:
            # send the viewpoint if it exists
            self.visualize_viewpoint(next_viewpoint)
            resp.available = True
            resp.viewpoint = next_viewpoint
        else:
            resp.available = False
            resp.viewpoint = [0.0,0.0]

        # send computed viewpoint to client
        return resp
    
    def compute_next_viewpoint(self,map,origin,resolution,robot_pos, boundaries=[]):
        '''
            Function to compute the next best viewpoint, given an occupancy grid map and current robot pose.
            This function creates an instance of FrontierBasedExploration to compute the next best viewpoint.

            Parameters:
            map (Numpy array): grid map data
            origin (list): real-world pose of the cell (0,0) in the map.
            resolution (float): resolution of the occupancy grid map
            robot_pos (list or 1D Numpy array): position [x,y] of the robot (in cartesian, NOT in cells)
            boundaries (list): coordinates of the boundaries to be explored, in order: [x_min, x_max, y_min, y_max]

            Returns:
            viewpoint (list): next best viewpoint (in cartesian)
        '''

        # convert robot position into cells
        robot_pos_cell = self.__cartesian_to_gridmap__(robot_pos,origin,resolution,map.shape)

        # convert boundaries into cells if given
        if boundaries:
            min_bound = self.__cartesian_to_gridmap__([boundaries[0],boundaries[2]],origin,resolution,map.shape)
            max_bound = self.__cartesian_to_gridmap__([boundaries[1],boundaries[3]],origin,resolution,map.shape)
            if min_bound and max_bound:
                boundaries_cell = [min_bound[0],max_bound[0],min_bound[1],max_bound[1]]
            else:
                boundaries_cell = []
        else:
            boundaries_cell = []

        # instantiate FrontierBasedExploration
        cluster_size_threshold = 5*((2*0.177)/resolution)
        alg = FrontierBasedExploration(map,cluster_size_threshold=cluster_size_threshold,boundaries=boundaries_cell)

        # compute next best viewpoint
        # viewpoint_cell = alg.set_next_best_viewpoint(robot_pos_cell)
        viewpoint_cell_list, viewpoint_cost = alg.generate_viewpoints(robot_pos_cell)

        # visualize frontiers
        self.visualize_frontiers(alg.clustered_frontiers)

        # convert viewpoints into cartesian if available
        if viewpoint_cell_list:
            viewpoint_list = []
            for viewpoint_cell in viewpoint_cell_list:
                viewpoint = self.__gridmap_to_cartesian__(viewpoint_cell,origin,resolution,map.shape)
                viewpoint_list.append(viewpoint)
        else:
            return None
          
        # check validity of each viewpoint and return the first valid viewpoint
        # consider also the minimum distance between the viewpoint and the robot
        svc = StateValidityChecker(distance=0.2)
        svc.set(map,resolution,origin)
        for i in range(len(viewpoint_list)):
            viewpoint = viewpoint_list[i]
            if svc.is_valid(viewpoint) and (viewpoint_cost[i]*self.map_resolution >= self.min_dist):
                return viewpoint
            
        # if none of the viewpoints are valid     
        return None
        
    def __cartesian_to_gridmap__(self,p,origin,resolution,shape):
        '''
            Convert cartesian coordinates into cells in gridmap.
        '''
        # Convert position from world frame to grid map frame
        cell_x = int((p[0] - origin[0]) / resolution)
        cell_y = int((p[1] - origin[1]) / resolution)

        # Check if the computed cell index is within the grid map boundaries
        if 0 <= cell_x < shape[0] and 0 <= cell_y < shape[1]:
            return [cell_x, cell_y]
        else:
            return None
        
    def __gridmap_to_cartesian__(self,m,origin,resolution,shape):
        '''
            Convert cells in gridmap into cartesian coordinates.
        '''
        cell_x = m[0]
        cell_y = m[1]

        # checks if the given cell index is within the grid map boundaries
        if 0 <= cell_x < shape[0] and 0 <= cell_y < shape[1]:
            px = cell_x * resolution + origin[0]
            py = cell_y * resolution + origin[1]
            return [px,py]
        else:
            return None
    
    def visualize_viewpoint(self,viewpoint):
        p = PointStamped()
        p.header.frame_id = 'world_ned'
        p.header.stamp = rospy.Time.now()

        p.point.x = viewpoint[0]
        p.point.y = viewpoint[1]
        p.point.z = 0.0

        self.viewpoint_pub.publish(p)

    def visualize_frontiers(self, clustered_frontiers):
        n_cluster = len(clustered_frontiers)
        m = Marker()
        m.header.frame_id = 'world_ned'
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.POINTS
        m.ns = 'cluster'
        m.action = Marker.DELETE
        m.lifetime = rospy.Duration(5)
        self.frontiers_pub.publish(m)

        m.action = Marker.ADD
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0

        # Publish new frontiers
        for i in range(len(clustered_frontiers)):
            cluster = clustered_frontiers[i]
        
            color_cluster = ColorRGBA()
            color_cluster.r = 0
            color_cluster.g = (n_cluster - i) * (1/n_cluster)
            color_cluster.b = (i+1) * (1/n_cluster)
            color_cluster.a = 0.5

            for point in cluster:
                point_coordinates = self.__gridmap_to_cartesian__(point, self.map_origin, self.map_resolution, self.map.shape)
                p = Point()
                p.x = point_coordinates[0]
                p.y = point_coordinates[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_cluster)
        
        self.frontiers_pub.publish(m)

        # Keep track of the previous number of clusters
        self.prev_n_cluster = n_cluster



if __name__=='__main__':
    rospy.init_node('frontier_exploration_node') # initialize the node
    # node = ExplorationNode([-2.0,2.0,0.0,4.45]) # uncomment this if testing with circuit2
    # node = ExplorationNode([-0.5,3.0,-4.0,0.2])
    node = ExplorationNode()

    rospy.spin()