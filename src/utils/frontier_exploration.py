import numpy as np
import math
from sklearn.cluster import DBSCAN
import statistics

class FrontierBasedExploration:

    # Constructor
    def __init__(self, map, cluster_size_threshold = 5, boundaries = []) -> None:
        # Occupancy grid map
        self.map = map

        # Cluster size threshold
        self.cluster_size_threshold = cluster_size_threshold

        # Map boundaries to be explored (in cells: [x_min, x_max, y_min, y_max])
        self.boundaries = boundaries

        # A complete list of unexplored frontier viewpoints and their corresponding costs
        self.open_viewpoints = []
        self.open_viewpoints_cost = []

        # A list of clustered frontiers (mainly for debugging)
        self.clustered_frontiers = []

    def set_next_best_viewpoint(self,current_pose):
        '''
            Method containing the main algorithm of frontier-based exploration to set the next point
            for the robot to go to in order to explore the map.
            Returns the next point to explore (list).
        '''

        free_cells = self.find_free_cells(self.map)
        frontier_points = self.find_new_frontiers(self.map, free_cells)
        self.open_viewpoints = self.cluster_frontiers(frontier_points)

        # Stop if there are no frontier clusters
        if len(self.open_viewpoints) == 0:
            print("No more frontiers to explore!")
            return None

        # Updates the cost of all open viewpoints based on the current robot pose
        new_cost = []

        for viewpoint in self.open_viewpoints:
            cost = np.linalg.norm(np.subtract(current_pose,viewpoint))
            new_cost.append(cost)
        
        self.open_viewpoints_cost = new_cost

        # Determine the next viewpoint to go, based on the updated cost
        next_viewpoint = self.open_viewpoints[np.argmin(self.open_viewpoints_cost)]

        return next_viewpoint
    
    def generate_viewpoints(self,current_pose):
        '''
            Method to generate possible viewpoints based on the gridmap.
            Returns a list of viewpoints and their associated costs, sorted from the lowest to highest cost.
        '''

        free_cells = self.find_free_cells(self.map)
        frontier_points = self.find_new_frontiers(self.map, free_cells)
        open_viewpoints = self.cluster_frontiers(frontier_points)

        # Stop if there are no frontier clusters
        if len(open_viewpoints) == 0:
            print("No more frontiers to explore!")
            return None,None

        # Updates the cost of all open viewpoints based on the current robot pose
        new_cost = []

        for viewpoint in open_viewpoints:
            cost = np.linalg.norm(np.subtract(current_pose,viewpoint))
            new_cost.append(cost)

        # Sort the viewpoints based on the cost
        zipped_lists = zip(new_cost, open_viewpoints)
        sorted_pairs = sorted(zipped_lists)
        sorted_cost, sorted_viewpoints = zip(*sorted_pairs) # unzip
        # Convert tuples back to list
        sorted_cost = list(sorted_cost)
        sorted_viewpoints = list(sorted_viewpoints)

        return sorted_viewpoints, sorted_cost       

    def find_free_cells(self,map):
        '''
            Method to find free cells in the map.
            Returns a list of free cells that corresponds to the map.
        '''

        free_cells = []

        if self.boundaries:
            # use the boundaries if set
            x_min = self.boundaries[0]
            x_max = self.boundaries[1]
            y_min = self.boundaries[2]
            y_max = self.boundaries[3]
        else:
            # use the whole gridmap is boundaries are not set
            x_min = 0
            x_max = map.shape[0]
            y_min = 0
            y_max = map.shape[1]

        # Loop for all cells in map
        for x in range(x_min,x_max):
            for y in range(y_min,y_max):
                if map[x,y] == 0: # cell in map is free
                    free_cells.append([x,y])

        return free_cells

    def find_new_frontiers(self,map, free_cells):
        '''
            Method to determine the frontier cells from the given gridmap and a list of new free cells.
            Frontier cells are defined as a free cell that is adjacent to an unknown cell.
            This method only search for new frontiers in the list of new free cells.
            Returns a list of frontier cells.
        '''

        frontiers_list = []

        for cell in free_cells:
            x = cell[0]
            y = cell[1]
            # loop for all surrounding cells
            for x_check in [x-1,x,x+1]:
                # skip checking if it's out of bounds
                if not (0 <= x_check < map.shape[0]):
                    continue

                for y_check in [y-1,y,y+1]:
                    if not (0 <= y_check < map.shape[1]):
                        continue
                    
                    cell_check = [x_check,y_check]
                    if cell_check == cell: # skip checking the current cell
                        continue
                    
                    if map[x_check,y_check] == -1:
                        frontiers_list.append(cell)
                        break

                if cell in frontiers_list:
                    break
        
        return frontiers_list

    def cluster_frontiers(self,frontier_points):
        '''
            Method to cluster frontier cells contained in frontier_points and determine the viewpoint of each cluster.
            Clustering is done using DBSCAN, and the viewpoint is taken from the median of each cluster.
            Returns a list of clustered frontier cells and a list of viewpoints.
        '''
        
        # Pre-processing
        X = np.array(frontier_points)

        # Perform clustering
        cluster_labels = DBSCAN(eps=1,min_samples=3).fit_predict(X)

        # separate the clusters and compute the viewpoint of each cluster
        n_labels = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        clustered_frontiers = [] # initialize a list of clustered frontiers
        viewpoints_list = [] # initialize a list of viewpoints

        for label in range(n_labels):
            # separate cluster
            cluster = [frontier_points[i] for i in range(len(frontier_points)) if cluster_labels[i] == label]

            # only consider a cluster if the number of frontier cells in the cluster is above the threshold
            if len(cluster) > self.cluster_size_threshold:
                clustered_frontiers.append(cluster)

                # compute viewpoint as the median of each cluster
                median_index = int((len(cluster)+1)/2)
                viewpoint = cluster[median_index]
                viewpoints_list.append(viewpoint)

        self.clustered_frontiers = clustered_frontiers

        return viewpoints_list
    

