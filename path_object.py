import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import tdmclient.notebook



#update angle to be average between computer vision angle and robot estimated angle
def update_angle(last_robot_pos,start,current_angle):
    angle = get_angle(last_robot_pos,list(start))
    #check if the difference between the two angles is more than 180 degrees, then correct
    correction = 0
    if abs(angle-current_angle) > 180:
        correction = 180
    current_angle = (current_angle + angle)/2-correction
    return current_angle

#fix output from camera to always have the same order and always have the same length
def fix_output(sorted_object,sorted_object_last):
    if len(sorted_object) < 2:
        return False
    if sorted_object[0][0] != "robot": 
        sorted_object.insert(0,["robot",None])
    if sorted_object[1][0] != "mars": 
        sorted_object.insert(1,["mars",None])
    if sorted_object[2][0] != "earth":
        sorted_object.insert(2,["earth",None])

    #if the object is not detected, use the last known position
    if sorted_object[0][1] == None: 
        sorted_object[0][1] = sorted_object_last[0][1]
    if sorted_object[1][1] == None:
        sorted_object[1][1] = sorted_object_last[1][1]
    if sorted_object[2][1] == None:
        sorted_object[2][1] = sorted_object_last[2][1]

    return sorted_object 
sorted_object_last = [['robot', [0.1, 0.1, 0.12, 0.24], 0.912], ['mars',  [0.95, 0.95, 0.12, 0.24], 0.912], ['earth', [0.35, 0.89, 0.09, 0.15], 0.905]]







def get_angle(point1,point2):
    angle = math.atan2(point2[1]-point1[1],point2[0]-point1[0])
    return np.degrees(angle)
def get_distance(point1,point2):
    distance = math.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)
    return distance




def convert_coordinates_2d_list(coordinates_2d_list, old_width=100, old_height=75, new_width=1920, new_height=1080):
    new_coordinates_2d_list = []
    for old_coordinates in coordinates_2d_list:
        new_x = (old_coordinates[0] / old_width) * new_width
        new_y = ((old_height-old_coordinates[1] )/ old_height) * new_height
        new_coordinates_2d_list.append((int(round(new_x,0)), int(round(new_y,0))))
    return new_coordinates_2d_list

class Global_nav: 
    def __init__(self,scalefactor =1,max_val=100,movements='16N'): 

        #the scalefactor is used to convert the coordinates to cm. Standard is 
        self.scalefactor = scalefactor
        #edge of the map x direction. Map is 100x75
        self.max_val_x = max_val 
        #edge of the map y direction. Map is 100x75
        self.max_val_y = max_val*0.75
        #the movements are the possible movements of the robot.
        self.movements = movements
        #the occupancy grid is the map with the obstacles. 
        self.occupancy_grid = None

        #the radius of the robot and a added margin
        self.thymio_radius = 7.5

    
    def create_map(self, goal =[90,70], black_holes_centers = [(20,10),(30,40),(65,65)], black_holes_radiuss = [10,10,25]): 
        #some code is taken from class exercises and modified to our needs 

        #Goal is mars 
        self.goal = goal
        #Creating the grid
        max_val_x = 100 # Size of the map x
        max_val_y = 75# Size of the map  y

        self.radius_margin = 1.5 # we add a margin to make the robot not go directly to the edge of the black hole. 

        x,y = np.mgrid[0:max_val_x:1, 0:max_val_y:1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
        coords = list([(int(x[0]), int(x[1])) for x in pos])

        
        # Define the heuristic, here = distance to goal ignoring obstacles
        h = np.linalg.norm(pos - goal, axis=-1)
        h = dict(zip(coords, h))

        # Creating the occupancy grid
        occupancy_grid = np.zeros((max_val_x, max_val_y))  # Create a grid of 50 x 50 random values

        #color a circle of with radius of the black holes
        if black_holes_centers is not None:
            for radius, center in zip(black_holes_radiuss, black_holes_centers):
                for i in range(max_val_x):
                    for j in range(max_val_y):
                        if math.sqrt((i - center[0])**2 + (j - center[1])**2) < radius*self.radius_margin:
                            occupancy_grid[i,j] = 1

        #check if goal is in an obstacle
        if occupancy_grid[goal[0],goal[1]] == 1:
            #print error message 
            print('goal is in an obstacle')
            return None,None,None,None


        self.occupancy_grid = occupancy_grid
        self.h = h
        self.coords = coords
        return occupancy_grid, h, coords
    
    def convert_OPENCV_tovalues(self,coordinates):
        #function to convert OPEN CV coordinates to values that can be used in the A* algorithm

        #the output of OPEN CV is 0-1 for both x and y. Therfore multiplied by 100 and 75
        start = (int(coordinates[0][1][0]*self.max_val_x),int((1-coordinates[0][1][1])*self.max_val_y))
        goal = [int(coordinates[1][1][0]*self.max_val_x),int((1-coordinates[1][1][1])*self.max_val_y)]
        earth = [int(coordinates[2][1][0]*self.max_val_x),int((1-coordinates[2][1][1])*self.max_val_y)]

        #get the radius's of the robot and earth from the width given by OPEN CV
        robot_radius = coordinates[1][1][2]/2*self.max_val_x
        earth_radius = coordinates[2][1][2]/2*self.max_val_x

        #the measured width of earth is 9.8 cm. This is used to scale the map
        earth_real_radius= 9.8/2 #cm
        self.scale_factor = earth_real_radius/earth_radius


        black_holes_centers = []
        black_holes_radiuss = []
        for i in range(len(coordinates)-3):
            black_holes_centers_ = [int(np.round(coordinates[i+3][1][0]*self.max_val_x,0)),int(np.round((1-coordinates[i+3][1][1])*self.max_val_y,0))]
            black_holes_radiuss_ = coordinates[i+3][1][2]/2*self.max_val_x+robot_radius
            black_holes_centers.append(black_holes_centers_)
            black_holes_radiuss.append(black_holes_radiuss_)
        return start, goal, earth, robot_radius, earth_radius, black_holes_centers, black_holes_radiuss,self.scale_factor

    
    def get_path_straight(self,start = (1,1), goal = None): 
        #some of the code is taken from class exercises and modified to our needs

        if self.occupancy_grid is None:
            self.create_map()
        if goal is not None:
            self.goal = goal
       
        
        # these are the moves used if the robot is stuck in an obstacle
        self.backup_moves = [(5, 0, 5.0),
                            (0, 5, 5.0),
                            (-5, 0, 5.0),
                            (0, -5, 5.0),
                            (10, 0, 10.0),
                            (0, 10, 10.0),
                            (-10, 0, 10.0),
                            (0, -10, 10.0),
                            (10, 10, np.sqrt(200)),
                            (-10, 10, np.sqrt(200)),
                            (-10, -10, np.sqrt(200)),
                            (10, -10, np.sqrt(200)),
                            (5,5,np.sqrt(50)),
                            (-5,5,np.sqrt(50)),
                            (-5,-5,np.sqrt(50)),
                            (5,-5,np.sqrt(50))]
        
        #these are the moves used for the A* algorithm
        if self.movements == '8N': 
            s2 = math.sqrt(2)
            movements =[(1, 0, 1.0),
                        (0, 1, 1.0),
                        (-1, 0, 1.0),
                        (0, -1, 1.0),
                        (1, 1, s2),
                        (-1, 1, s2),
                        (-1, -1, s2),
                        (1, -1, s2)]
        #these are the moves used for the A* algorithm
        elif self.movements == '16N': 
            s2 = math.sqrt(2)
            s5 = math.sqrt(5)
            movements =[(1, 0, 1.0),
                        (0, 1, 1.0),
                        (-1, 0, 1.0),
                        (0, -1, 1.0),
                        (1, 1, s2),
                        (-1, 1, s2),
                        (-1, -1, s2),
                        (1, -1, s2),
                        (2,1,s5),
                        (2,-1,s5),
                        (-2,1,s5),
                        (-2,-1,s5),
                        (1,2,s5),
                        (1,-2,s5),
                        (-1,2,s5),
                        (-1,-2,s5)
                        ]
        else: 
            #32N 
            s2 = math.sqrt(2)
            s5 = math.sqrt(5)
            movements =[(1, 0, 1.0),
                        (0, 1, 1.0),
                        (-1, 0, 1.0),
                        (0, -1, 1.0),
                        (1, 1, s2),
                        (-1, 1, s2),
                        (-1, -1, s2),
                        (1, -1, s2),
                        (2,1,s5),
                        (2,-1,s5),
                        (-2,1,s5),
                        (-2,-1,s5),
                        (1,2,s5),
                        (1,-2,s5),
                        (-1,2,s5),
                        (-1,-2,s5),
                        (2,3,math.sqrt(13)),
                        (2,-3,math.sqrt(13)),
                        (-2,3,math.sqrt(13)),
                        (-2,-3,math.sqrt(13)),
                        (3,2,math.sqrt(13)),
                        (3,-2,math.sqrt(13)),
                        (-3,2,math.sqrt(13)),
                        (-3,-2,math.sqrt(13)), 
                        (3,1,math.sqrt(10)),
                        (3,-1,math.sqrt(10)),
                        (-3,1,math.sqrt(10)),
                        (-3,-1,math.sqrt(10)),
                        (1,3,math.sqrt(10)),
                        (1,-3,math.sqrt(10)),
                        (-1,3,math.sqrt(10)),
                        (-1,-3,math.sqrt(10))
                        ]
        
        
        # Initially, only the start node is known.
        openSet = [start]
        
        # nodes that are closed
        closedSet = set([])
        
        # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
        cameFrom = dict()
        # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
        gScore = dict(zip(self.coords, [np.inf for x in range(len(self.coords))]))
        gScore[start] = 0

        # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
        fScore = dict(zip(self.coords, [np.inf for x in range(len(self.coords))]))
        fScore[start] = self.h[start]

        def lowest_f(openSet): 
            #find lowest f score in openSet
            scores = np.ones(len(openSet))
            for i, node in enumerate(openSet):
                scores[i] = fScore[node]
            return openSet[np.argmin(scores)]
        
        def divide_path(path, max_length = 10): 
            #function used to dvivide the straight path into subsegments
            path_2 = []
            for i in range(len(path))[1:]: 
                point_1 = path[i-1]
                point_2 = path[i]


                dist = int(max((np.linalg.norm(np.array(point_1)-np.array(point_2))/max_length),1))
                
                line_x = np.linspace(point_1[0], point_2[0], num=dist)
                line_y = np.linspace(point_1[1], point_2[1], num=dist)
                
                line = np.array([(x,y) for x,y in zip(line_x,line_y)])
                path_2.extend(line)

            path_2.extend([path[-1]])
            return path_2
        
        def unique(opti_path):
            #function used to remove duplicate points in the path
            opti_path_ = [list(opti_path[0])]
            last_point = opti_path[0]
            for point in opti_path: 
                if (point[0] != last_point[0]) or (point[1] != last_point[1]):
                    opti_path_.append(list(point))
                last_point = point
            return opti_path_
        
        def straight_path(path):
            path_2 = [path[0]]
            while True:
                if path_2[-1] == self.goal: 
                    break
                for i in range(len(path))[1:]: 
                    #draw line between path_2[-1] and path[i]
                    #check if line is free
                    if path[i] == self.goal:
                        path_2.append(path[i])
                        break
                    point_2 = path[i]

                    #get the number of points we want to check along the line to check whether the line is free of obstacles. Here we check every 2 pixels
                    checks = max((np.linalg.norm(np.array(path_2[-1])-np.array(point_2))/2).astype(int),1)

                    line_x = np.linspace(path_2[-1][0], point_2[0], num=checks).astype(int)
                    line_y = np.linspace(path_2[-1][1], point_2[1], num=checks).astype(int)
                    line = np.array([(x,y) for x,y in zip(line_x,line_y)])
                    if np.any(self.occupancy_grid[line[:,0],line[:,1]] == 1): 
                        path_2.append(path[i-1])
                    else: 
                        continue
            
            #check if robot is close to mars then do smaller steps
            if np.linalg.norm(np.array(start)-np.array(goal)) < 5: 
                path_2 = divide_path(path_2, max_length = 1)
            else: 
                path_2 = divide_path(path_2, max_length = 3)
            
            #only return unique points
            path_2 = unique(path_2)

            return path_2



        current= start
        # while there are still nodes to visit (hint : change the while condition)
        while len(openSet)!=0: #DUMMY VALUE - TO BE CHANGED
        
            #find the unvisited node having the lowest fScore[] value
            closedSet.add(current)
            current = lowest_f(openSet)
            openSet.remove(current)
            
            #If the goal is reached, reconstruct and return the obtained path
            if current[0] == self.goal[0] and current[1] == self.goal[1]:
                path = [self.goal] 
              
                while True: 
                    path.append(cameFrom[tuple(path[-1])])
                    if path[-1] == start:
                        optimal_path = straight_path(path[::-1])

                        #if robot is inside black hole we skip the second point to make exit from black hole smoother
                        if self.occupancy_grid[int(optimal_path[0][0]),int(optimal_path[0][1])] == 1: 
                            del optimal_path[1]

                        return optimal_path, path[::-1], closedSet,openSet

            # If the goal was not reached, for each neighbor of current:
            for dx, dy, deltacost in movements:
                neighbor = (current[0]+dx, current[1]+dy) #DUMMY VALUE - TO BE CHANGED
                # if the node is not in the map, skip
    
                if neighbor[0]<0 or neighbor[1]<0 or neighbor[0]>=self.max_val_x or neighbor[1]>=self.max_val_y: 
                    continue
                
                # if the node is occupied or has already been visited, skip
                if self.occupancy_grid[neighbor[0], neighbor[1]]: 
                    continue
                if  neighbor in closedSet:
                    continue
                
                # compute the cost to reach the node through the given path
                tentative_gScore = gScore[current]+deltacost #DUMMY VALUE - TO BE CHANGED

                # If the computed cost if the best one for that node, then update the costs and 
                # node from which it came
            
                if tentative_gScore < gScore[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = gScore[neighbor] + self.h[neighbor]

                    if neighbor not in openSet:
                        openSet.append(neighbor)
                
                #backup plan for when stuck in obstacale 
            if (len(openSet) == 0): 
                for dx, dy, deltacost in self.backup_moves:
                    neighbor = (current[0]+dx, current[1]+dy)
                    # if the node is not in the map, skip
        
                    if neighbor[0]<0 or neighbor[1]<0 or neighbor[0]>=self.max_val_x or neighbor[1]>=self.max_val_y: 
                        continue
                    
                    # if the node is occupied or has already been visited, skip
                    if self.occupancy_grid[neighbor[0], neighbor[1]]: 
                        continue
                    if  neighbor in closedSet:
                        continue
                    
                    # compute the cost to reach the node through the given path
                    tentative_gScore = gScore[current]+deltacost

                    # If the computed cost if the best one for that node, then update the costs and 
                    # node from which it came
                
                    if tentative_gScore < gScore[neighbor]:
                        # This path to neighbor is better than any previous one. Record it!
                        cameFrom[neighbor] = current
                        gScore[neighbor] = tentative_gScore
                        fScore[neighbor] = gScore[neighbor] + self.h[neighbor]

                        if neighbor not in openSet:
                            openSet.append(neighbor)
                        
        return 'error' #no convergence
    
    

    
    

