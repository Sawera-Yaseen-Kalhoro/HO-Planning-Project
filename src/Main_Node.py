#!/usr/bin/env python3

import time
import rospy
import py_trees
import actionlib
import numpy as np
import copy
import os

from nav_msgs.msg import OccupancyGrid
from ho_planning_project.msg import DwaAction, DwaGoal
from utils.State_Validity_Checker import StateValidityChecker
from ho_planning_project.srv import PlanPath, PlanPathRequest
from py_trees.behaviours import CheckBlackboardVariableExists
from ho_planning_project.srv import ExplorationTrigger, ExplorationTriggerRequest
from geometry_msgs.msg import Twist


class Frontier(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Frontier, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key("NBV", access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug("%s [Frontier::setup()]" % self.name)
        rospy.wait_for_service('/compute_next_viewpoint')
        try:
            self.server = rospy.ServiceProxy('/compute_next_viewpoint', ExplorationTrigger)
            self.logger.debug(" %s [Frontier::setup() Server Connected!]" % self.name)
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Frontier::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Frontier::initialise()]" % self.name)

    def update(self):
        try:
            self.logger.debug("  {}: call service /compute_next_viewpoint".format(self.name))
            resp = self.server(ExplorationTriggerRequest())
            if resp.available:
                print("New viewpoint generated.")
                viewpoint = resp.viewpoint
                self.blackboard.NBV = viewpoint
                self.logger.debug("Set NBV in blackboard: {}".format(viewpoint))
                return py_trees.common.Status.SUCCESS
            else:
                print("No viewpoint available.")
                return py_trees.common.Status.FAILURE
        except rospy.ServiceException as e:
            self.logger.debug("  {}: Error calling service /compute_next_viewpoint".format(self.name))
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug("  %s [Frontier::terminate().terminate()][%s->%s]" %
                          (self.name, self.status, new_status))


class Planner(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Planner, self).__init__(name)

        self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        self.blackboard.register_key("NBV", access=py_trees.common.Access.READ)

        #self.blackboard = self.attach_blackboard_client(name = "Blackboard")
        self.blackboard.register_key("Path", access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug("%s [Planner::setup()]"% self.name)
        rospy.wait_for_service('/plan_path')
        try:
            self.server = rospy.ServiceProxy('/plan_path', PlanPath)
            self.logger.debug(" %s [Planner::setup() Server Connected!]" % self.name)
        except rospy.ServiceException as e:
            self.logger.debug(" %s [Planner::setup() ERROR!]" % self.name)

    def initialise(self):
        self.logger.debug(" %s [Planner::initialise()]" % self.name)

    def update(self):
        try:
            self.logger.debug(
                "  {}: call service /plan_path".format(self.name))

            # Retrieve the NBV from the blackboard
            nbv = self.blackboard.NBV

            # Check if NBV is available
            if nbv is not None:
                # Create a request message and populate the 'goal' field with NBV
                request = PlanPathRequest()
                request.goal = nbv

                # Call the service with the populated request
                print("Generating new path...")
                resp = self.server(request)

                if resp.success:
                    # Extract waypoints from the planned path message
                    planned_path = resp.planned_path
                    planned_path_list = []
                    for pose_stamped in planned_path.poses:
                        # Extract x, y coordinates from the pose
                        x = pose_stamped.pose.position.x
                        y = pose_stamped.pose.position.y
                        # Append the coordinates to the list
                        planned_path_list.append((x, y))

                    # Save the list of waypoints in the blackboard
                    self.blackboard.Path = planned_path_list
                    self.logger.debug("Path in blackboard: {}".format(planned_path_list))
                    return py_trees.common.Status.SUCCESS
                else:
                    return py_trees.common.Status.FAILURE
            else:
                self.logger.debug("NBV is not available in the blackboard.")
                return py_trees.common.Status.FAILURE

        except rospy.ServiceException as e:
            self.logger.debug(
                "  {}: Error calling service /plan_path: {}".format(self.name, e))
            return py_trees.common.Status.FAILURE


    def terminate(self, new_status):
        self.logger.debug("  %s [Planner::terminate().terminate()][%s->%s]" %
                            (self.name, self.status, new_status))
        

class DWA(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(DWA, self).__init__(name)

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key("Path", access=py_trees.common.Access.READ)

        # Initialize action client
        self.client = actionlib.SimpleActionClient('dwa_act_server', DwaAction)
        self.client.wait_for_server()

        # Variables to store feedback
        self.feedback_received = False
        self.feedback_message = ""
        self.current_waypoint_index = 0  # Initialize current waypoint index

    def setup(self):
        self.logger.debug("%s [DWA::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("%s [DWA::initialise()]" % self.name)
        self.current_waypoint_index = 0

    def feedback_callback(self, feedback):
        # Update feedback variables
        self.feedback_received = True
        self.feedback_message = feedback.robot_moving

    def update(self):
        self.logger.debug("%s [DWA::update()]" % self.name)

        # Reset feedback variables
        self.feedback_received = False
        self.feedback_message = ""

        # Retrieve path from blackboard
        path = self.blackboard.Path
        path_copy = copy.deepcopy(path)
        if len(path_copy) > 0:
            del path_copy[0]

        # Check if there are waypoints left to navigate
        while self.current_waypoint_index < len(path_copy):
            # Retrieve current waypoint
            current_waypoint = path_copy[self.current_waypoint_index]

            # Create action goal
            goal = DwaGoal()
            goal.goal_pos = current_waypoint  # Set the current waypoint as the goal

            # Send goal to action server
            self.client.send_goal(goal, feedback_cb=self.feedback_callback)

            # Wait for the result within 1 second (in order to keep updating status)
            action_finish = self.client.wait_for_result(timeout=rospy.Duration(1))

            # Process result
            if action_finish:
                result = self.client.get_result()

                if result and result.goal_reached:
                    # Move to the next waypoint
                    self.current_waypoint_index += 1
                    self.logger.debug("Reached waypoint %d" % self.current_waypoint_index)
                    # Check if all waypoints are reached
                    if self.current_waypoint_index >= len(path_copy):    
                        # All waypoints reached successfully
                        self.logger.debug("Robot successfully reached all waypoints.")
                        # reset index
                        self.current_waypoint_index = 0
                        return py_trees.common.Status.SUCCESS
                else:
                    self.logger.debug("Failed to reach waypoint: %s" % current_waypoint)
                    return py_trees.common.Status.FAILURE
            
            return py_trees.common.Status.RUNNING

        # # If all waypoints are reached successfully
        # self.logger.debug("Robot successfully reached all waypoints.")
        # return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug("%s [DWA::terminate()][%s->%s]" %
                          (self.name, self.status, new_status))


class Is_Path_Valid(py_trees.behaviour.Behaviour):
    def __init__(self, name, gridmap_topic, distance_threshold):
        super(Is_Path_Valid, self).__init__(name)

        #self.distance_threshold = distance_threshold
        self.svc = StateValidityChecker(distance_threshold)
        self.last_map_time = None

        # Subscribe to the gridmap topic
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap)

        # Publisher to stop if invalid
        self.stop_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)

        # Attach blackboard
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key("Path", access=py_trees.common.Access.READ)

        # Initialize action client to cancel
        self.client = actionlib.SimpleActionClient('dwa_act_server', DwaAction)
        self.client.wait_for_server()

    def setup(self):
        self.logger.debug("%s [Is_Path_Valid::setup()]" % self.name)

        
    def initialise(self):
        self.logger.debug("%s [Is_Path_Valid::initialise()]" % self.name)

    def update(self):
       
        # Check the validity of the path
        path = self.blackboard.Path

        if not self.svc.check_path(path):
            print("Path is invalid. Stopping controller...")
            # cancel DWA
            self.client.cancel_all_goals()
            # stop the robot
            stop_cmd = Twist()
            stop_cmd.linear.x = 0
            stop_cmd.angular.z = 0
            self.stop_pub.publish(stop_cmd)
            # If any waypoint is invalid, return FAILURE
            return py_trees.common.Status.FAILURE
        # If all waypoints are valid, return RUNNING
        return py_trees.common.Status.RUNNING
        
    def get_gridmap(self, gridmap):
        # Check if the gridmap data is valid
        if gridmap is None:
            self.logger.warning("Received NoneType gridmap, skipping update.")
            return

         # To avoid map update too often (change value '1' if necessary)
        if (not self.svc.there_is_map) or ((gridmap.header.stamp - self.last_map_time).to_sec() > 5):           
            self.last_map_time = gridmap.header.stamp

            try:
                # Reshape gridmap data and extract origin information
                env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
                origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
                
                # Update State Validity Checker
                self.svc.set(env, gridmap.info.resolution, origin)
                
                # Update the last map update time
                self.last_map_time = gridmap.header.stamp
            except Exception as e:
                self.logger.error("Error updating State Validity Checker: {}".format(e))

    
    def terminate(self, new_status):
        self.logger.debug("%s [Is_Path_Valid::terminate()][%s->%s]" %
                          (self.name, self.status, new_status))


class Reset_BB(py_trees.behaviour.Behaviour):
    def __init__(self, name="Clear Blackboard"):
        super(Reset_BB, self).__init__(name=name)

    def update(self):
        # Clear all keys and values from the blackboard
        py_trees.blackboard.Blackboard.clear()
        
        # Indicate that the blackboard has been cleared
        print("Blackboard has been cleared.")
        
        # Return success since the blackboard has been successfully cleared
        return py_trees.common.Status.SUCCESS



def create_tree():

    # Frontier Behavior
    frontier = Frontier("Frontier")

    # Planner Behavior
    planner = Planner("Planner")

    # DWA Behavior
    dwa = DWA("DWA")

    # is_path_valid behavior
    is_path_valid = Is_Path_Valid("IsPathValid", gridmap_topic='/projected_map', distance_threshold=0.2)

    # Check_BB Behavior
    check_BB = CheckBlackboardVariableExists(name="CheckBB", variable_name="NBV")

    # Reset_BB Behavior
    reset_BB = Reset_BB("ResetBB")


    # If one child fails the whole node will return failure
    parallel = py_trees.composites.Parallel(
        name = "Path Following",  
        policy = py_trees.common.ParallelPolicy.SuccessOnOne(),
        children = [dwa, is_path_valid]
    )

    selector = py_trees.composites.Selector(
        name = "Selector",
        memory = True,
        children = [check_BB, frontier]
    )

    # Sequence of all actions (Root Node)
    root = py_trees.composites.Sequence(
        name = "Path Planning",
        memory = True,
        children = [selector, planner, parallel, reset_BB]
    )

    return root


# Function to run the behavior tree
def run(it=5000):
    root = create_tree()

    try:
        print("Call setup for all tree children")
        root.setup_with_descendants() 
        print("Setup done!\n\n")
        py_trees.display.ascii_tree(root)
        
        for _ in range(it):
            root.tick_once()
            time.sleep(1)
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    rospy.init_node("behavior_trees")

    # Create the behavior tree
    root = create_tree()
    
    # Display the behavior tree
    save_loc = os.path.expanduser('~/catkin_ws/src/ho_planning_project/img') # Define data file location
    py_trees.display.render_dot_tree(root, target_directory=save_loc) # generates a figure for the 'root' tree.

    # Run the behavior tree
    run()
