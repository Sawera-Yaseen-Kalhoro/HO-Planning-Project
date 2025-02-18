import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class Robot: 
    """
    Robot parameters and constants
    """

    def __init__(self):
        # reading file 
        SCRIPTDIR = os.path.dirname(__file__)
        file_name = os.path.join(SCRIPTDIR, 'robot_parameters.yaml')

        with open(file_name, 'r') as f:
            data = yaml.safe_load(f)

        # Robot parameters from yaml file
        self.max_speed = data["robot"]["max_speed"]  # [m/s]
        self.min_speed = data["robot"]["min_speed"]  # [m/s]
        self.max_angular_speed = data["robot"]["max_angular_speed"]  # [rad/s]
        self.max_accel = data["robot"]["max_accel"] # [m/s**2]
        self.max_angular_accel = data["robot"]["max_angular_accel"]  # [rad/**2]
        self.v_resolution = data["robot"]["v_resolution"] # [m/s]
        self.w_resolution = data["robot"]["w_resolution"] * math.pi / 180.0  # [rad/s]
        self.dt = data["robot"]["dt"] # [s] Time tick for motion prediction
        self.predict_time = data["robot"]["predict_time"]  # [s] 
        self.dist_threshold = data["robot"]["dist_threshold"] #[m] adimissible distance between robot and goal
        # GAINS CONSTANTS
        self.to_goal_cost_gain = data["robot"]["to_goal_cost_gain"] # alpha
        self.speed_cost_gain = data["robot"]["speed_cost_gain"] # gamma 
        self.obstacle_cost_gain = data["robot"]["obstacle_cost_gain"] # beta
        self.robot_stuck = data["robot"]["robot_stuck"]  # constant to prevent robot stucked
        # self.robot_stuck = - data["robot"]["robot_stuck"]  # constant to prevent robot stucked
        
        # Robot Radius used to check if goal is reached
        self.robot_radius = data["robot"]["robot_radius"]  # [m] for collision check, consider using a safe radius ? 10 %


def dwa_control(x, robot, goal, obs):
    """
    Dynamic Window Approach control

    returns: 
        u pair of (v,w)
        trajectory list(x), where the robot is moving, used for plotting 
        
    """
    dw = calc_dynamic_window(x, robot)

    # u, trajectory = robot_control(x, dw, robot, goal, obs)
    u, trajectory,list_possible_paths = robot_control(x, dw, robot, goal, obs)

    return u, trajectory,list_possible_paths


def motion(x, u, dt):
    """
    motion model where :
    x position 
        x = [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    u pair of (v,w)

    returns: New x after applying u.

    """

    # x[0] += u[0] * np.cos(x[2]) * dt + 0.5* self.max_accel * dt^2 * np.cos(x[2])
    # x[1] += u[0] * np.sin(x[2]) * dt + 0.5* self.max_accel * dt^2 * np.sin(x[2])
    # x[2] += u[1] * dt + self.max_angular_accel * dt^2

    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, robot):
    """
    calculation of dynamic window based on current state x
    x = [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]

    returns: the limits where are all possible combinations of (v,w) 
    """

    # Dynamic window from robot specification
    Vs = [robot.min_speed, robot.max_speed,
          -robot.max_angular_speed, robot.max_angular_speed]

    # Dynamic window from motion model
    # Vd = [x[3] - robot.max_accel * robot.dt,
    #       x[3] + robot.max_accel * robot.dt,
    #       x[4] - robot.max_angular_accel * robot.dt,
    #       x[4] + robot.max_angular_accel * robot.dt]

    Vd = [0 - robot.max_accel * robot.dt,
          0 + robot.max_accel * robot.dt,
          0 - robot.max_angular_accel * robot.dt,
          0 + robot.max_angular_accel * robot.dt]

    #  [v_min, v_max, w_min, w_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    
    return dw


def predict_trajectory(x_init, v, w, robot):
    """
    predict trajectory with an input
    returns: trajectory list(x)
             the curvature trajectory the robot will do depending on the v and w
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= robot.predict_time:
        x = motion(x, [v, w], robot.dt)
        trajectory = np.vstack((trajectory, x))
        time += robot.dt

    return trajectory


def robot_control(x, dw, robot, goal, obs):
    """
    calculation of the dynamic window considering the 3 paramenters
    returns:
        u (v,w), best possible u for the robot to move according to the obs, pos and current vel
        trajectory, used for plotting  
    """

    x_init = x[:]
    min_cost = float("inf") # infinity number
    
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    list_possible_paths = []
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.linspace(dw[0], dw[1], 5):
        for w in np.linspace(dw[2], dw[3], 5):
            trajectory = predict_trajectory(x_init, v, w, robot)
            # calc cost
            to_goal_cost = robot.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = robot.speed_cost_gain * (robot.max_speed - trajectory[-1][3])
            obs_cost = robot.obstacle_cost_gain * calc_obstacle_cost(trajectory, obs, robot)
            last_x_traj = trajectory[-1]
            last_point_traj = (last_x_traj[0], last_x_traj[1])
            list_possible_paths.append(np.array([last_point_traj]))
            final_cost = to_goal_cost + speed_cost + obs_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, w]
                best_trajectory = trajectory
                # in order to not get the robot stuck in
                # best v=0 m/s (in front of an obstacle) and best w=0 rad/s (heading to the goal with
                # angle difference of 0) 


                # if abs(best_u[0]) < robot.robot_stuck \
                #         and abs(x[3]) < robot.robot_stuck:
                #     print("Activating robot stuck")
                #     best_u[1] = -robot.max_angular_speed


                # quick calculation of best robot position, right or left
                # unit_v = math.cos(x[3]) 

    # return best_u, best_trajectory
    return best_u, best_trajectory,list_possible_paths

##################################################################
#                     COST FUNCTIONS
##################################################################


def calc_obstacle_cost(trajectory, obs, robot):
    """
    calc obstacle cost inf: collision
    """
    # ox = obs[:, 0]
    # oy = obs[:, 1]

    # dx = trajectory[:, 0] - ox[:, None]
    # dy = trajectory[:, 1] - oy[:, None]
    # r = np.hypot(dx, dy)

    # if np.array(r <= robot.robot_radius).any():
    #     return float("Inf")
    # min_r = np.min(r)
    # # print(min_r)
    # # return 1 / min_r
    # return 0

    min_r = float('Inf')
    ox = obs[:, 0]
    oy = obs[:, 1]

    tx = trajectory[:, 0]
    ty = trajectory[:, 1]

    for i in range(len(ox)):
        for j in range(len(tx)):
            d = np.sqrt((ox[i] - tx[j])**2 + (oy[i] - ty[j])**2)
            if d < min_r:
                min_r = d
            # once one of the position in the trajectory is invalid returns infinity
            if min_r < robot.robot_radius:
                return float('Inf')
    # when no future position is in contact with an obstacle then
    # # return 1 / min_r
    if min_r == float('Inf'):
        return 0

    return 1.0/ min_r

def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """
    
    dx = goal[0] - trajectory[-1][0]
    dy = goal[1] - trajectory[-1][1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1][2]
    # cost_dist = 1/ (np.sqrt(dx**2 + dy**2))
    # cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))/np.pi
    cost = abs(wrap_angle(cost_angle))
    return cost# + cost_dist
    #return abs(cost_angle)



##################################################################
#                     PLOTTING FUNCTIONS
##################################################################

def plot_robot_arm(x, y, yaw, length=0.5, width=0.2):  # arm located at front of robot
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, robot):  # same size as kobuki
    
    circle = plt.Circle((x, y), robot.robot_radius, color="gray")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * robot.robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")


def plot_simulation(x, robot, goal, obs, predicted_trajectory):
    plt.cla()
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
    plt.plot(goal[0], goal[1], "xr")        
    plt.plot(obs[:, 0], obs[:, 1], "sb")
    plot_robot(x[0], x[1], x[2], robot)
    plot_robot_arm(x[0], x[1], x[2])
    plt.axis("equal")
    plt.grid(True)
    
    plt.pause(0.0001)
        

def DWA(goal, obs):
    robot = Robot()

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi*0 , 0.0, 0.0])
    
    dist_threshold = 0.2 #[m] adimissible distance between robot and goal

    trajectory = np.array(x)
    while True:
        u, predicted_trajectory = dwa_control(x, robot, goal, obs)
        x = motion(x, u, robot.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        plot_simulation(x, robot, goal, obs, predicted_trajectory)
        # check reaching goal
        dist_to_goal = np.linalg.norm(np.array([x[0] - goal[0], x[1] - goal[1]]))
        if dist_to_goal <= robot.robot_radius + dist_threshold:
            break

    # Additional plot
    # plot red trajetory of robot from initial pose to goal    
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    plt.pause(0.0001)
    plt.show()
    


if __name__ == '__main__':

    goal = np.array([0, 6])


    # obstacles [x(m) y(m), ... ]
    obs = np.array([[-1, -1],[0, 2],
                            # [1, 2],
                            [5.0, 3.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [-2.0, 3.0],
                            [-3.0, 3.0],
                            [-4.0, 3.0],
                            [-5.0, 3.0],
                            [2.0, 3.0],
                            [3.0, 3.0],
                            [4.0, 3.0],
                            [5.0, 3.0],
                            [6.0, 3.0],
                            [11.0, 3.0],
                            [10.0, 3.0],
                            [5.0, 9.0],
                            [6.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 9.0],
                            [9.0, 9.0],
                            [6.0, 5.0],
                            [-3.0, -3.0],
                            [5.0, -5.0],
                            [5.0, 5.0]
                            ])

    DWA(goal, obs)

