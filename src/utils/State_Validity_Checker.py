import numpy as np
import copy
import math


class StateValidityChecker:
    """ Checks if a position or a path is valid given an occupancy map."""

    # Constructor
    def __init__(self, distance=0.1, is_unknown_valid=True):
        # map: 2D array of integers which categorizes world occupancy
        self.map = None
        # map sampling resolution (size of a cell))
        self.resolution = None
        # world position of cell (0, 0) in self.map
        self.origin = None
        # set method has been called
        self.there_is_map = False
        # radius arround the robot used to check occupancy of a given position
        self.distance = distance
        # if True, unknown space is considered valid
        self.is_unknown_valid = is_unknown_valid

    # Set occupancy map, its resolution and origin.
    def set(self, data, resolution, origin):
        #print("set method called")
        self.map = data
        self.resolution = resolution
        self.origin = np.array(origin)# here is the 
        self.there_is_map = True

   # Given a pose, returns true if the pose is not in collision and false otherwise.
    def is_valid(self, pose):
        map_copy = copy.deepcopy(self.map)

        # Calculate the range around the pose to check
        lower_limit = tuple(i - self.distance for i in pose)
        upper_limit = tuple(i + self.distance for i in pose)

        # loop for all the surrounding pixels
        valid = True  # initialize bool variable for output

        # DEBUGGING
        self.surroundings = []

        for x in np.arange(lower_limit[0], upper_limit[0], self.resolution):
            for y in np.arange(lower_limit[1], upper_limit[1], self.resolution):
                # DEBUGGING
                self.surroundings.append((x,y)) 
                # convert coordinate into map index
                #print("x and y: ", x, y)
                cell = self.__position_to_map__((x, y))

                # check if this cell is outside the map (considered unknown)
                if cell is None:
                    if self.is_unknown_valid == False: # unknown is considered False
                        valid = False
                        break

                else: # if it is inside the map
                    # check if the cell is an obstacle
                    is_obstacle = map_copy[cell] == 100
                    # check if the cell is unknown
                    is_unknown = map_copy[cell] == -1

                    # combine the conditions for False
                    if is_obstacle or (is_unknown and self.is_unknown_valid == False):
                        valid = False
                        break
        
        return valid


    # Given a path, returns true if the path is not in collision and false otherwise.
    def check_path(self, path):
        # DEBUGGING
        #print("Checking path between: ", path[0], path[1])
        step_size = 0.05

        # DEBUGGING
        self.check_points = []

        # Discretization of the position between 2 waypoints
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]

            # Direction and distance between current and next points
            direction = (next_point[0] - current_point[0], next_point[1] - current_point[1])  # By vector subtraction
            distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)  # Euclidean Distance

            # Number of steps needed between current and next points
            num_steps = int(distance / step_size)

            # Discretization of the path
            for j in range(num_steps):
                x = current_point[0] + j * step_size * direction[0] / distance
                y = current_point[1] + j * step_size * direction[1] / distance
                self.check_points.append((x,y))
                # Checking Validity of each point
                if not self.is_valid((x, y)):
                    return False

        return True

    # Transform position with respect the map origin to cell coordinates
    def __position_to_map__(self, p):
        # Convert position from world frame to grid map frame
        #print("p: in svc node: ", p)
        #print("self.origin: ", self.origin)
        cell_x = int((p[0] - self.origin[0]) / self.resolution)
        cell_y = int((p[1] - self.origin[1]) / self.resolution)

        # Check if the computed cell index is within the grid map boundaries
        if 0 <= cell_x < self.map.shape[0] and 0 <= cell_y < self.map.shape[1]:
            return cell_x, cell_y
        else:
            return None
        
    
    def __map_to_position__(self, pos_map):
         # convert map coordinates into world position
        x = (pos_map[0]*self.resolution + self.origin[0]) + self.resolution/2
        y = (pos_map[1]*self.resolution + self.origin[1]) + self.resolution/2
        pose = np.array([x,y])
        return pose