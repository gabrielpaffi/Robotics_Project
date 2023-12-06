# MARKOV THIS IS THE NEW VERSION !!!

# Imports
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sn 

import tdmclient.notebook


def display_heatmap(data_to_display):
    #hm = sn.heatmap(data = data_to_display, annot=False, linewidth=.5, vmin=0, vmax=1.0) 
    hm = sn.heatmap(data = data_to_display, annot=False, linewidth=.5, vmin=0, vmax=1.0) 
    hm.invert_yaxis()
    
def neighbouring_pixels(x,y,prob,any_map):
    global map_estimate_robot
 
    any_map[x-1][y-1] = prob
    any_map[x-1][y] = prob
    any_map[x-1][y+1] = prob

    any_map[x][y-1] = prob
    any_map[x][y+1] = prob

    any_map[x+1][y-1] = prob
    any_map[x+1][y] = prob
    any_map[x+1][y+1] = prob

 
def initialize_maps(length,height):
    global map, map_estimate_robot, map_estimate_CV
    
    # initialize map
    start_prob = 1/(length*height)
    map = np.empty((length,height), float)
    map.fill(start_prob)
    #display_heatmap(map)

    # initialize map_estimate_robot
    map_estimate_robot = np.empty((length,height), float)
    map_estimate_robot.fill(0.0)

    # initialize map_estimate_CV
    map_estimate_CV = np.empty((length,height), float)
    map_estimate_CV.fill(0)


def estimate_robot(x,y):
    global map_estimate_robot   
    
    # reset map to 0
    map_estimate_robot.fill(0)

    # add prob
    P_HIT = 0.8
    P_MISS = 0.2/8 #8 neighboring pixels
    map_estimate_robot[x][y] = P_HIT
    neighbouring_pixels(x,y,P_MISS,map_estimate_robot)



def estimate_CV(x,y,confidence_CV):
    # reset map to 0
    map_estimate_CV.fill(0)

    # add prob
    MEAS_P_HIT = confidence_CV
    MEAS_P_MISS = (1-confidence_CV)/8
    map_estimate_CV[x][y] = MEAS_P_HIT
    neighbouring_pixels(x,y,MEAS_P_MISS,map_estimate_CV)


def multiply_maps(x_robot,y_robot,length,height):
    global map, map_estimate_CV, map_estimate_robot
    temp = np.multiply(map_estimate_CV,map_estimate_robot)
    last_map_was_initialized_map = False
    
    # if the estimated position of the robot and by CV are too far apart, or the last position of the robot and its position now, the whole map after multiplication is = 0
    # in this case, set map = map_estimate_robot as it is less likely that the position measured by the robot is off by a lot than the CV
    
    
    
    '''
    if map[x_robot][y_robot] == 1/(length*height):
        last_map_was_initialized_map = True
    elif (not np.any(temp)):
        map = map
    else:
        map = np.multiply(map,temp)

    if (not np.any(map)):
        map = map_estimate_CV  # if start map and new position is too different, just take the position of the CV (e.g. displacing robot by hand)
    '''

    # special cases
    



    return last_map_was_initialized_map


def normalize_map():
    global map
  
    sum = np.sum(map)
    map = np.divide(map,sum)



def multiple_highest_prob_with_same_value(pos_robot,indices):

    # save estimated position of robot, estimated by itself
    pos_robot_x = pos_robot[0].item()
    pos_robot_y = pos_robot[1].item()
    point = np.array([pos_robot_x, pos_robot_y])

    # put values of pos with the same highest prob in a table, table[0] -> x, table[1] -> 1
    table = np.array(indices).tolist()

    # Transpose the table array
    table_array = np.array(table).T

    # Calculate Euclidean distances
    distances = np.linalg.norm(table_array - point, axis=1)

    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    X_ROBOT = table[0][1]
    Y_ROBOT = table[1][1]

    return X_ROBOT,Y_ROBOT


def filtered_pos_robot():
    indices = np.where(map == map.max())
    pos_robot = np.where(map_estimate_robot == map_estimate_robot.max())
    

    if (np.size(indices)==2): 
        X_ROBOT = indices[0].item()
        Y_ROBOT = indices[1].item()
    else:
        X_ROBOT, Y_ROBOT = multiple_highest_prob_with_same_value(pos_robot,indices)

    print(X_ROBOT,Y_ROBOT)
    return X_ROBOT,Y_ROBOT


def markov(x_robot,y_robot,x_CV,y_CV,confindence_CV,length,height):
    # Initialize local variables
    x_filtered_pos_robot = 0
    y_filtered_pos_robot = 0
    # take integer value (rounded down)
    x_robot = np.int64(x_robot)
    y_robot = np.int64(y_robot)
    x_CV = np.int64(x_CV)
    y_CV = np.int64(y_CV)
    
    
    estimate_robot(x_robot,y_robot)
    estimate_CV(x_CV,y_CV,confindence_CV)
    last_map_was_initialized_map = multiply_maps(x_robot,y_robot,length,height)
    print(last_map_was_initialized_map)
    normalize_map()
    x_filtered_pos_robot, y_filtered_pos_robot = filtered_pos_robot()

    if ((abs(x_CV-x_robot) >= 3) or (abs(x_CV-x_robot) >= 3)):
        print("in special case")
        x_filtered_pos_robot = x_CV
        y_filtered_pos_robot = y_CV
        map[x_robot][y_robot] = 0.8
        neighbouring_pixels(x_robot,y_robot,0.025,map)

    '''
    if last_map_was_initialized_map:
        print("in here")
        x_filtered_pos_robot = x_CV
        y_filtered_pos_robot = y_CV
    '''

    return x_filtered_pos_robot, y_filtered_pos_robot
