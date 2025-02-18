import math
import time
import random
import numpy as np
from utils import State_Validity_Checker

class Node:
    def __init__(self, state):
        self.state = state
        self.parent = None


class Planner:
    def __init__(self, state_validity_checker, max_time=5.0, delta_q=2, dominion=[-10, 10, -10, 10]):
        """
        Initialize the RRT-Connect planner with the given parameters.

        Parameters:
        - state_validity_checker: Instance of StateValidityChecker class.
        - max_time: Maximum planning time.
        - delta_q: Step size.
        - dominion: Bounds for generating random configurations.
        """

        self.state_validity_checker = state_validity_checker
        self.max_time = max_time
        self.delta_q = delta_q
        self.dominion = dominion

    def compute_path(self, q_start, q_goal):
            """
            Main RRT-Connect planning algorithm.

            Parameters:
            - q_start: Starting configuration tuple (x, y).
            - q_goal: Goal configuration tuple (x, y).

            Returns:
            - path: A list of tuples representing the path from q_start to q_goal if found, otherwise None.
            """
            self.q_start = Node(q_start)
            self.q_goal = Node(q_goal)
            self.G1 = [self.q_start]
            self.G2 = [self.q_goal]

            start_time = time.time()
            while time.time() < start_time + self.max_time:
                check_qnew_valid = False
                while (not check_qnew_valid) and (time.time() < start_time + self.max_time):
                    q_rand = self.rand_conf()
                    q_near1 = self.nearest_vertex(self.G1, q_rand.state)
                    q_new1 = self.new_conf(q_near1, q_rand.state, self.delta_q)
                    if self.state_validity_checker.is_valid(q_new1.state) and self.state_validity_checker.check_path([q_near1.state, q_new1.state]):
                        check_qnew_valid = True

                if check_qnew_valid:
                    if self.state_validity_checker.check_path([q_near1.state, q_new1.state]):
                        self.G1.append(q_new1)
                        q_near2 = self.nearest_vertex(self.G2, q_new1.state)
                        q_new2 = self.new_conf(q_near2, q_new1.state, self.delta_q)

                        if self.state_validity_checker.is_valid(q_new2.state):
                            if self.state_validity_checker.check_path([q_new2.state, q_near2.state]):
                                self.G2.append(q_new2)

                                while True:
                                    node_new_prim = self.new_conf(q_new2, q_new1.state, self.delta_q)

                                    if self.state_validity_checker.check_path([node_new_prim.state, q_new2.state]):

                                        self.G2.append(node_new_prim)
                                        q_new2 = self.change_node(q_new2, node_new_prim)
                                    else:
                                        break

                                    if self.is_node_same(q_new2, q_new1.state):
                                        break

                        if self.is_node_same(q_new2, q_new1.state):
                            path = self.extract_path(q_new1, q_new2)
                            smooth_path = self.smooth_path(path, self.state_validity_checker)

                            return smooth_path

                self.swap()

            return []



    def rand_conf(self):
        """
        Generate a new configuration randomly within the grid map.
        """

        qrand_x = random.uniform(self.dominion[0], self.dominion[1])
        qrand_y = random.uniform(self.dominion[2], self.dominion[3])
        qrand = Node((qrand_x, qrand_y))

        while not self.state_validity_checker.is_valid(qrand.state):
            qrand_x = random.uniform(self.dominion[0], self.dominion[1])
            qrand_y = random.uniform(self.dominion[2], self.dominion[3])
            qrand = Node((qrand_x, qrand_y))

        return qrand

    def nearest_vertex(self, node_list, q_rand):
        """
        Find the nearest vertex in a given list of nodes to a given random configuration.
        """
        return min(node_list, key=lambda node: math.hypot(node.state[0] - q_rand[0], node.state[1] - q_rand[1]))

    
    def new_conf(self, q_near1, q_rand, delta_q):
        """
        Generate a new configuration based on the nearest vertex and a random configuration.
        """
        while True:
            dist, theta = self.get_distance_and_angle(q_near1.state, q_rand)

            dist = min(delta_q, dist)
            new_x = q_near1.state[0] + dist * math.cos(theta)
            new_y = q_near1.state[1] + dist * math.sin(theta)
            node_new = Node((new_x, new_y))
            node_new.parent = q_near1
                
            return node_new


    def extract_path(self, q_new1, q_new2):
        """
            Extract the path from the start to the goal through the given configurations.

            constructs a path from start to goal by tracing the parent pointers from goal to start node.
        """

        path1 = [(q_new1.state[0], q_new1.state[1])]
        current_node = q_new1

        while current_node.parent is not None:
            current_node = current_node.parent
            path1.append((current_node.state[0], current_node.state[1]))

        path2 = [(q_new2.state[0], q_new2.state[1])]
        current_node = q_new2

        while current_node.parent is not None:
            current_node = current_node.parent
            path2.append((current_node.state[0], current_node.state[1]))

        path = list(list(reversed(path1)) + path2)

        if path[0] != (self.q_start.state[0], self.q_start.state[1]):
            path.reverse()

        return path

    def get_distance_and_angle(self, node_start, node_end):
        """
        Compute the distance and angle between two nodes.
        """
        dx = node_end[0] - node_start[0]
        dy = node_end[1] - node_start[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def is_node_same(self, q_new2, q_new1_state):
        """
        Check if two nodes have the same configuration.
        """
        if q_new2.state[0] == q_new1_state[0] and q_new2.state[1] == q_new1_state[1]:
            return True

        return False

    def change_node(self, q_new2, node_new_prim):
        """
        Update a node with a new configuration.
        """
        node_new = Node((node_new_prim.state[0], node_new_prim.state[1]))
        node_new.parent = q_new2
        return node_new

    def swap(self):
        """
        Swap the two trees.
        """
        self.G1, self.G2 = self.G2, self.G1


    def smooth_path(self, path, checker):
        """
        Smooth the given path using collision checking.

        Parameters:
        - path: A list of tuples representing the path to be smoothed.
        - checker: Instance of StateValidityChecker class for collision checking.

        Returns:
        - smooth_path: A list of tuples representing the smoothed path.
        """

        last_valid_index = len(path) - 1
        smooth_path = path

        while last_valid_index > 0:
            start_index = 0

            # Find the next valid segment of the path
            while start_index < last_valid_index - 1 and not checker.check_path([smooth_path[start_index], smooth_path[last_valid_index]]):
                start_index += 1

            if start_index < last_valid_index - 1:
                # Shorter valid path found
                smooth_path = smooth_path[:start_index + 1] + smooth_path[last_valid_index:]
                last_valid_index = start_index
            else:
                # No valid path found, move to the previous point
                last_valid_index -= 1

        return smooth_path


# Planner: This function has to plan a path from start_p to goal_p. To check if a position is valid the
# StateValidityChecker class has to be used. The planning dominion must be specified as well as the maximum planning time.
# The planner returns a path that is a list of poses ([x, y]).
def compute_path(start_p, goal_p, state_validity_checker, bounds, max_time=5.0):

    # Creating a planner instance
    planner = Planner(state_validity_checker, max_time=max_time, dominion=bounds)

    # Computing path using RRT algorithm only if the goal position is within the bounds.
    if (bounds[0] <= goal_p[0] <= bounds[1]) and (bounds[2] <= goal_p[1] <= bounds[3]):
        path = planner.compute_path(start_p, goal_p)
    else:
        path = []

    return path

