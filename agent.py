__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2024, IN424, IPSA 2024"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"


import rclpy
from .Astar import a_star_search
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
import sys
import time
import numpy as np
import math
from .my_common import *#common variables are stored her 


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        #initialize attributes
        self.agents_pose = [None]*self.nb_agents    #[(x_1, y_1), (x_2, y_2), (x_3, y_3)] if there are 3 agents
        self.x = self.y = self.yaw = None   #the pose of this specific agent running the node
        self.front_dist = self.left_dist = self.right_dist = 0.0    #range values for each ultrasonic sensor

        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.init_map()

        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        #Subscribe to ultrasonic sensor topics for the corresponding agent
        self.create_subscription(Range, f"{self.ns}/us_front/range", self.us_front_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed at front of the robot
        self.create_subscription(Range, f"{self.ns}/us_left/range", self.us_left_cb, qos_profile=qos_profile_sensor_data)   #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the left side of the robot
        self.create_subscription(Range, f"{self.ns}/us_right/range", self.us_right_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the right of the robot
        
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.1s of period <=> 10 Hz
        self.create_timer(1, self.publish_maps) #1Hz

        self.strategy_called = False
        self.strategy_timer = self.create_timer(0.4, self.strategy)
    

    def load_params(self):
        """ Load parameters from launch file """
        self.declare_parameters(    #A node has to declare ROS parameters before getting their values from launch files
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),    #robot's namespace: either 1, 2 or 3
                ("robot_size", rclpy.Parameter.Type.DOUBLE),    #robot's diameter in meter
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),   #environment dimensions (width height)
                ("nb_agents", rclpy.Parameter.Type.INTEGER),    #total number of agents (this agent included) to map the environment
            ]
        )

        #Get launch file parameters related to this node
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value
    

    def init_map(self):
        """ Initialize the map to share with others if it is bot_1 """
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"    #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_msg.header.stamp = self.get_clock().now().to_msg() #get the current ROS time to send the msg
        self.map_msg.info.resolution = self.robot_size  #Map cell size corresponds to robot size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)   #nb of rows
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)    #nb of columns
        self.map_msg.info.origin.position.x = -self.env_size[1]/2   #x and y coordinates of the origin in map reference frame
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0    #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE #all the cells are unexplored initially
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height  
    

    def merged_map_cb(self, msg):
        """ 
            Get the current common map and update ours accordingly.
            This method is automatically called whenever a new message is published on the topic /merged_map.
            'msg' is a nav_msgs/msg/OccupancyGrid message.
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                    self.map[i, j] = received_map[i, j]


    def odom1_cb(self, msg):
        """ 
            Get agent 1 position.
            This method is automatically called whenever a new message is published on topic /bot_1/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)
    

    def odom2_cb(self, msg):
        """ 
            Get agent 2 position.
            This method is automatically called whenever a new message is published on topic /bot_2/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)
        # self.get_logger().info(f"Agent 2: ({x:.2f}, {y:.2f})")


    def odom3_cb(self, msg):
        """ 
            Get agent 3 position.
            This method is automatically called whenever a new message is published on topic /bot_3/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)
        # self.get_logger().info(f"Agent 3: ({x:.2f}, {y:.2f})")


    def map_update(self):
        """ Consider sensor readings to update the agent's map """
        if self.x != None and self.y != None:
            PosAgent = np.dot(np.array([[0,-1],[1,0]]),np.array([self.x,self.y]))

        """ Fill the free cases spoted """
    
        #Front
        if self.front_dist > 5:
            
            for i in range(1, 6):
                ObjectFREE_Front = np.dot(np.array([[0,-1],[1,0]]),np.array([[((3-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((3-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if ObjectFREE_Front[0] <=19 and ObjectFREE_Front[1] > -20:
                    self.map[int(ObjectFREE_Front[0]+20), int(ObjectFREE_Front[1]+20)] = FREE_SPACE_VALUE
            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE

        #Left
        if self.left_dist > 5:
            
            for i in range(1, 6):
                ObjectFREE_Left = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((3-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((3-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if ObjectFREE_Left[0] <=19 and ObjectFREE_Left[1] > -20:
                    self.map[int(ObjectFREE_Left[0]+20), int(ObjectFREE_Left[1]+20)] = FREE_SPACE_VALUE
            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE

        #Right
        if self.right_dist > 5:
            
            for i in range(1,6):
                ObjectFREE_Right = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((3-i/2-self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((3-i/2-self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if ObjectFREE_Right[0] <=19 and ObjectFREE_Right[1] > -20 and ObjectFREE_Right[1] <=19 and ObjectFREE_Right[0] > -20:
                    self.map[int(ObjectFREE_Right[0]+20), int(ObjectFREE_Right[1]+20)] = FREE_SPACE_VALUE
            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE

        if self.yaw != None:
            # Front sensor object position
            ObjectPos_Front = np.dot(np.array([[0,-1],[1,0]]),np.array([[((self.front_dist+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.x/self.map_msg.info.resolution],
                                    [((self.front_dist+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.y/self.map_msg.info.resolution]
                                    ]))
        
        # Left sensor object position
        ObjectPos_Left = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((self.left_dist+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                        [((self.left_dist+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                        ]))


        # Right sensor object position
        ObjectPos_Right = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((-self.right_dist-self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                        [((-self.right_dist-self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                        ]))
    
        if self.left_dist <= 3:

            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE
            self.map[int(ObjectPos_Left[0]+20), int(ObjectPos_Left[1]+20)] = OBSTACLE_VALUE #all the cells are unexplored initially

            for i in range(1, int(self.left_dist*2)):
                ObjectFREE_Left = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((self.left_dist-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((self.left_dist-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if self.map[int(ObjectFREE_Left[0]+20), int(ObjectFREE_Left[1]+20)] != OBSTACLE_VALUE:
                    if ObjectFREE_Left[0] <=19 and ObjectFREE_Left[1] > -20:
                        self.map[int(ObjectFREE_Left[0]+20), int(ObjectFREE_Left[1]+20)] = FREE_SPACE_VALUE

        if self.right_dist <= 3:

            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE
            self.map[int(ObjectPos_Right[0]+20), int(ObjectPos_Right[1]+20)] = OBSTACLE_VALUE #all the cells are unexplored initially

            for i in range(1, int(self.right_dist*2)):
                ObjectFREE_Right = np.dot(np.array([[0,-1],[1,0]]),np.array([[-((self.right_dist-i/2-self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((self.right_dist-i/2-self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if self.map[int(ObjectFREE_Right[0]+20), int(ObjectFREE_Right[1]+20)] != OBSTACLE_VALUE:
                    if ObjectFREE_Right[0] <=19 and ObjectFREE_Right[1] > -20:
                        self.map[int(ObjectFREE_Right[0]+20), int(ObjectFREE_Right[1]+20)] = FREE_SPACE_VALUE

        if self.front_dist <= 3:

            self.map[int(PosAgent[0]*2+20), int(PosAgent[1]*2+20)] = FREE_SPACE_VALUE
            self.map[int(ObjectPos_Front[0]+20), int(ObjectPos_Front[1]+20)] = OBSTACLE_VALUE

            for i in range(1, int(self.front_dist*2)):
                ObjectFREE_Front = np.dot(np.array([[0,-1],[1,0]]),np.array([[((self.front_dist-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.cos(self.yaw)+self.x/self.map_msg.info.resolution],
                                [((self.front_dist-i/2+self.robot_size/2)/self.map_msg.info.resolution)*np.sin(self.yaw)+self.y/self.map_msg.info.resolution]
                                ]))
                if self.map[int(ObjectFREE_Front[0]+20), int(ObjectFREE_Front[1]+20)] != OBSTACLE_VALUE:
                    if ObjectFREE_Front[0] <=19 and ObjectFREE_Front[1] > -20:
                        self.map[int(ObjectFREE_Front[0]+20), int(ObjectFREE_Front[1]+20)] = FREE_SPACE_VALUE
            

    def us_front_cb(self, msg):
        """ 
            Get measurement from the front ultrasonic sensor.
            This method is automatically called whenever a new message is published on topic /bot_x/us_front/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.front_dist = msg.range


    def us_left_cb(self, msg):
        """ 
            Get measurement from the ultrasonic sensor placed on the left.
            This method is automatically called whenever a new message is published on topic /bot_x/us_left/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.left_dist = msg.range


    def us_right_cb(self, msg):
        """ 
            Get measurement from the ultrasonic sensor placed on the right.
            This method is automatically called whenever a new message is published on topic /bot_x/us_right/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.right_dist = msg.range


    def publish_maps(self):
        """ 
            Publish updated map to topic /bot_x/map, where x is either 1, 2 or 3.
            This method is called periodically (1Hz) by a ROS2 timer, as defined in the constructor of the class.
        """
        self.map_msg.data = np.flipud(self.map).flatten().tolist()  #transform the 2D array into a list to publish it
        self.map_agent_pub.publish(self.map_msg)    #publish map to other agents
    

    def is_frontier_point(self, x, y):
        frontier_x1 = []
        frontier_x2 = []
        frontier_x3 = []
        frontier_y1 = []
        frontier_y2 = []
        frontier_y3 = []
        frontier_y = [frontier_y1,frontier_y2,frontier_y3]
        frontier_x = [frontier_x1,frontier_x2,frontier_x3]
        indices_x = [-1, 0, 1]
        indices_y = [-1, 0, 1]

        if x <=39 and x >=0 and y <=39 and y >= 0:
            if self.map[x, y] == FREE_SPACE_VALUE:
                for i in indices_x:
                    for j in indices_y:
                        if (x+i) <= 39 and (x+i) >= 0 and (y+j) <= 39 and (y+j) >= 0:
                            if self.map[x+i, y+j] == UNEXPLORED_SPACE_VALUE:
                                if int(self.ns[-1]) == 1:
                                    frontier_x[0].append(x+i)
                                    frontier_y[0].append(y+j)
                                if int(self.ns[-1]) == 2:
                                    frontier_x[1].append(x+i)
                                    frontier_y[1].append(y+j)
                                if int(self.ns[-1]) == 3:
                                    frontier_x[2].append(x+i)
                                    frontier_y[2].append(y+j)
        return frontier_x, frontier_y
    
    def explore_environment(self):
        PosAgent = np.dot(np.array([[0,-1],[1,0]]),np.array([self.x,self.y]))
        
        all_frontier_x1 = []
        all_frontier_y1 = []
        all_frontier_x2 = []
        all_frontier_y2 = []
        all_frontier_x3 = []
        all_frontier_y3 = []

        indices_x = list(range(-8, 8))
        indices_y = list(range(-8, 8))
        if int(self.ns[-1]) == 1:
            for i in indices_x:
                for j in indices_y:
                    frontier_x, frontier_y = self.is_frontier_point(int(PosAgent[0]*2+20)+i, int(PosAgent[1]*2+20)+j)
                    if frontier_x[0] and frontier_y[0]:
                        all_frontier_x1.append(frontier_x[0])
                        all_frontier_y1.append(frontier_y[0])

            for k in range(len(all_frontier_x1)):
                for j in range(len(all_frontier_y1[k])):
                    self.map[all_frontier_x1[k][j], all_frontier_y1[k][j]] = FRONTIER

        if int(self.ns[-1]) == 2:
            for i in indices_x:
                for j in indices_y:
                    frontier_x2, frontier_y2 = self.is_frontier_point(int(PosAgent[0]*2+20)+i, int(PosAgent[1]*2+20)+j)
                    if frontier_x2[1] and frontier_y2[1]:
                        all_frontier_x2.append(frontier_x2[1])
                        all_frontier_y2.append(frontier_y2[1])

            for k in range(len(all_frontier_x2)):
                for j in range(len(all_frontier_y2[k])):
                    
                    self.map[all_frontier_x2[k][j], all_frontier_y2[k][j]] = FRONTIER
        
        if int(self.ns[-1]) == 3:
            for i in indices_x:
                for j in indices_y:
                    frontier_x, frontier_y = self.is_frontier_point(int(PosAgent[0]*2+20)+i, int(PosAgent[1]*2+20)+j)
                    if frontier_x[2] and frontier_y[2]:
                        all_frontier_x3.append(frontier_x[2])
                        all_frontier_y3.append(frontier_y[2])

            for k in range(len(all_frontier_x3)):
                for j in range(len(all_frontier_y3[k])):
                    self.map[all_frontier_x3[k][j], all_frontier_y3[k][j]] = FRONTIER
        
        return all_frontier_x1, all_frontier_y1, all_frontier_x2, all_frontier_y2, all_frontier_x3, all_frontier_y3


    def group_frontier(self, frontier_points):
        grouped_frontiers = []
        for point in frontier_points:
            grouped = False
            for group in grouped_frontiers:
                if any(abs(point[0] - p[0]) <= 1 and abs(point[1] - p[1]) <= 1 for p in group):
                    group.append(point)
                    grouped = True
                    break
            if not grouped:
                grouped_frontiers.append([point])
            
        return grouped_frontiers   
           
    def frontier_find(self):
        all_frontier_x1, all_frontier_y1, all_frontier_x2, all_frontier_y2, all_frontier_x3, all_frontier_y3 = self.explore_environment()
        
        all_frontier1 = []
        all_frontier2 = []
        all_frontier3 = []
        grouped_frontiers1 = []
        grouped_frontiers2 = []
        grouped_frontiers3 = []
        grouped_frontiers = [grouped_frontiers1, grouped_frontiers2, grouped_frontiers3]

        if int(self.ns[-1]) == 1:
            for k in range(len(all_frontier_x1)):
                for j in range(len(all_frontier_x1[k])):
                    all_frontier1.append([all_frontier_x1[k][j], all_frontier_y1[k][j]])

            grouped_frontiers[0] = self.group_frontier(all_frontier1)
        
        if int(self.ns[-1]) == 2:
            for k in range(len(all_frontier_x2)):
                for j in range(len(all_frontier_x2[k])):
                    all_frontier2.append([all_frontier_x2[k][j], all_frontier_y2[k][j]])

            grouped_frontiers[1] = self.group_frontier(all_frontier2)

        if int(self.ns[-1]) == 3:
            for k in range(len(all_frontier_x3)):
                for j in range(len(all_frontier_x3[k])):
                    all_frontier3.append([all_frontier_x3[k][j], all_frontier_y3[k][j]])

            grouped_frontiers[2] = self.group_frontier(all_frontier3)

        return grouped_frontiers
    

    def centroid(self):
        sum_x = 0
        sum_y = 0
        centroids1 = []
        centroids2 = []
        centroids3 = []
        centroids = [centroids1, centroids2, centroids3]
        len_frontiers1 = []
        len_frontiers2 = []
        len_frontiers3 = []
        len_frontiers = [len_frontiers1, len_frontiers2, len_frontiers3]
        grouped_frontiers = self.frontier_find()

        if int(self.ns[-1]) == 1:
            for k in range(len(grouped_frontiers[0])):
                for j in range(len(grouped_frontiers[0][k])):
                    sum_x += grouped_frontiers[0][k][j][0]
                    sum_y += grouped_frontiers[0][k][j][1]
                len_frontiers[0].append(len(grouped_frontiers[0][k]))
                avg_x = int(sum_x/len(grouped_frontiers[0][k]))
                avg_y = int(sum_y/len(grouped_frontiers[0][k]))
                centroids[0].append([avg_x, avg_y])
                self.map[avg_x, avg_y] = CENTROID
                sum_x = 0
                sum_y = 0
    
        if int(self.ns[-1]) == 2:
            for k in range(len(grouped_frontiers[1])):
                for j in range(len(grouped_frontiers[1][k])):
                    sum_x += grouped_frontiers[1][k][j][0]
                    sum_y += grouped_frontiers[1][k][j][1]
                len_frontiers[1].append(len(grouped_frontiers[1][k]))
                avg_x = int(sum_x/len(grouped_frontiers[1][k]))
                avg_y = int(sum_y/len(grouped_frontiers[1][k]))
                centroids[1].append([avg_x, avg_y])
                self.map[avg_x, avg_y] = CENTROID
                sum_x = 0
                sum_y = 0

        if int(self.ns[-1]) == 3:
            for k in range(len(grouped_frontiers[2])):
                for j in range(len(grouped_frontiers[2][k])):
                    sum_x += grouped_frontiers[2][k][j][0]
                    sum_y += grouped_frontiers[2][k][j][1]
                len_frontiers[2].append(len(grouped_frontiers[2][k]))
                avg_x = int(sum_x/len(grouped_frontiers[2][k]))
                avg_y = int(sum_y/len(grouped_frontiers[2][k]))
                centroids[2].append([avg_x, avg_y])
                self.map[avg_x, avg_y] = CENTROID
                sum_x = 0
                sum_y = 0
        
        return centroids, len_frontiers
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    def find_best_centroid(self):
        PosAgent = np.dot(np.array([[0,-1],[1,0]]),np.array([self.x,self.y]))
        centroids, len_frontiers = self.centroid()
        i = 0
        L1 = 10
        L2 = 5
        L3 = 2
        path_list = []
        cost_list = []
        best_centroid1 = []
        best_centroid2 = []
        best_centroid3 = []
        best_centroid = [best_centroid1,best_centroid2,best_centroid3]
        best_path1 = []
        best_path2 = []
        best_path3 = []
        best_path = [best_path1,best_path2,best_path3]

        if int(self.ns[-1]) == 1:
            # Orientation calculations
            for centroid in centroids[0]:
                delta_x = centroid[0] - int(PosAgent[0])*2+20
                delta_y = centroid[1] - int(PosAgent[1])*2+20

                orientation = math.degrees(math.atan2(delta_y, delta_x))
            
                path = a_star_search(self.map, [int(PosAgent[0])*2+20, int(PosAgent[1])*2+20], centroid)
                path_list.append(path)

                if path != None:
                    cost = L1*len(path) - L2*len_frontiers[0][i] + L3*orientation
                    cost_list.append(cost)

                i =+1
            if cost_list:
                minimum_cost = min(cost_list)
                index_of_minimum = cost_list.index(minimum_cost)
                best_centroid[0] = centroids[0][index_of_minimum]
                best_path[0] = path_list[index_of_minimum]

                # COlor in green the best centroid for each agent
                self.map[best_centroid[0][0], best_centroid[0][1]] = BEST_CENTROID

        if int(self.ns[-1]) == 2:
            # Orientation calculations
            for centroid in centroids[1]:
                delta_x = centroid[0] - int(PosAgent[0])*2+20
                delta_y = centroid[1] - int(PosAgent[1])*2+20

                orientation = math.degrees(math.atan2(delta_y, delta_x))
            
                path = a_star_search(self.map, [int(PosAgent[0])*2+20, int(PosAgent[1])*2+20], centroid)
                path_list.append(path)

                if path != None:
                    cost = L1*len(path) - L2*len_frontiers[1][i] + L3*orientation
                    cost_list.append(cost)

                i =+1
            if cost_list:
                minimum_cost = min(cost_list)
                index_of_minimum = cost_list.index(minimum_cost)
                best_centroid[1] = centroids[1][index_of_minimum]
                best_path[1] = path_list[index_of_minimum]

                # COlor in green the best centroid for each agent
                self.map[best_centroid[1][0], best_centroid[1][1]] = BEST_CENTROID

        if int(self.ns[-1]) == 3:
            # Orientation calculations
            for centroid in centroids[2]:
                delta_x = centroid[0] - int(PosAgent[0])*2+20
                delta_y = centroid[1] - int(PosAgent[1])*2+20

                orientation = math.degrees(math.atan2(delta_y, delta_x))
            
                path = a_star_search(self.map, [int(PosAgent[0])*2+20, int(PosAgent[1])*2+20], centroid)
                path_list.append(path)

                if path != None:
                    cost = L1*len(path) - L2*len_frontiers[2][i] + L3*orientation
                    cost_list.append(cost)

                i =+1
            if cost_list:
                minimum_cost = min(cost_list)
                index_of_minimum = cost_list.index(minimum_cost)
                best_centroid[2] = centroids[2][index_of_minimum]
                best_path[2] = path_list[index_of_minimum]

                # COlor in green the best centroid for each agent
                self.map[best_centroid[2][0], best_centroid[2][1]] = BEST_CENTROID
            
        return best_centroid, best_path


    def movetobestcentroid(self, best_path):

        if int(self.ns[-1]) == 1:
            path = best_path[0]
            for i, point in enumerate(path[1:]):

                # Convert path points to the real world coordinates
                x_target = (point[1]-20)/2
                y_target = (20 -point[0])/2
                self.get_logger().info(f"x_target, y_target : {x_target}, {y_target}")

                # Calculate the angle to the target
                delta_x = x_target - self.x
                delta_y = y_target - self.y
                angle_to_target = math.atan2(delta_y, delta_x)

                angular_velocity= 1.0
                tolerance_angle = math.radians(1)
                start_time = time.time()

                while abs(angle_to_target - self.yaw) > tolerance_angle:
                    twist_msg = Twist()
                    if angle_to_target > self.yaw:
                        twist_msg.angular.z = angular_velocity
                    else:
                        twist_msg.angular.z = -angular_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    self.yaw += twist_msg.angular.z * elapsed_time  # Mise à jour de self.yaw en fonction de la rotation
                    start_time = current_time  # Mettre à jour le temps de début pour le prochain calcul

                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                
                distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
                linear_velocity = 1.0
                tolerance_distance = 0.5
                start_x, start_y = self.x, self.y

                while delta_x > tolerance_distance or delta_y > tolerance_distance:
                    twist_msg = Twist()
                    twist_msg.linear.x = linear_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    # Update self.x and self.y based on the traveled distance
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    distance_traveled = linear_velocity * elapsed_time
                    self.x += distance_traveled * math.cos(angle_to_target)
                    self.y += distance_traveled * math.sin(angle_to_target)
                    self.get_logger().info(f"self.x : {self.x}")
                    # Update distance to the target
                    delta_x = x_target - self.x
                    delta_y = y_target - self.y
                    distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

                    
                twist_msg.linear.x = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                time.sleep(2)
                self.get_logger().info(f"Im stopped")


    
        if int(self.ns[-1]) == 2:
            path = best_path[1]
            for i, point in enumerate(path[1:]):

                # Convert path points to the real world coordinates
                x_target = (point[1]-20)/2
                y_target = (20 -point[0])/2

                # Calculate the angle to the target
                delta_x = x_target - self.x
                delta_y = y_target - self.y
                angle_to_target = math.atan2(delta_y, delta_x)

                angular_velocity= 1.0

                start_time = time.time()
                tolerance_angle = math.radians(1)  # Tolerance angle for alignment

                while abs(angle_to_target - self.yaw) > tolerance_angle:
                    twist_msg = Twist()
                    if angle_to_target > self.yaw:
                        twist_msg.angular.z = angular_velocity
                    else:
                        twist_msg.angular.z = -angular_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    self.yaw += twist_msg.angular.z * elapsed_time  # Mise à jour de self.yaw en fonction de la rotation
                    start_time = current_time  # Mettre à jour le temps de début pour le prochain calcul

                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                
                distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
                linear_velocity = 1.0
                tolerance_distance = 0.5
                start_x, start_y = self.x, self.y

                while delta_x > tolerance_distance or delta_y > tolerance_distance:
                    twist_msg = Twist()
                    twist_msg.linear.x = linear_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    # Update self.x and self.y based on the traveled distance
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    distance_traveled = linear_velocity * elapsed_time
                    self.x += distance_traveled * math.cos(angle_to_target)
                    self.y += distance_traveled * math.sin(angle_to_target)
                    self.get_logger().info(f"self.x : {self.x}")
                    # Update distance to the target
                    delta_x = x_target - self.x
                    delta_y = y_target - self.y
                    distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

                    
                twist_msg.linear.x = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                time.sleep(2)
                self.get_logger().info(f"Im stopped")


        if int(self.ns[-1]) == 3:
            path = best_path[2]
            for i, point in enumerate(path[1:]):

                # Convert path points to the real world coordinates
                x_target = (point[1]-20)/2
                y_target = (20 -point[0])/2

                # Calculate the angle to the target
                delta_x = x_target - self.x
                delta_y = y_target - self.y
                angle_to_target = math.atan2(delta_y, delta_x)

                angular_velocity= 1.0

                start_time = time.time()
                tolerance_angle = math.radians(1)  # Tolerance angle for alignment

                while abs(angle_to_target - self.yaw) > tolerance_angle:
                    twist_msg = Twist()
                    if angle_to_target > self.yaw:
                        twist_msg.angular.z = angular_velocity
                    else:
                        twist_msg.angular.z = -angular_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    self.yaw += twist_msg.angular.z * elapsed_time  # Mise à jour de self.yaw en fonction de la rotation
                    start_time = current_time  # Mettre à jour le temps de début pour le prochain calcul

                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                
                distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
                linear_velocity = 1.0
                tolerance_distance = 0.5
                start_x, start_y = self.x, self.y

                while delta_x > tolerance_distance or delta_y > tolerance_distance:
                    twist_msg = Twist()
                    twist_msg.linear.x = linear_velocity
                    self.cmd_vel_pub.publish(twist_msg)

                    # Update self.x and self.y based on the traveled distance
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    distance_traveled = linear_velocity * elapsed_time
                    self.x += distance_traveled * math.cos(angle_to_target)
                    self.y += distance_traveled * math.sin(angle_to_target)
                    self.get_logger().info(f"self.x : {self.x}")
                    # Update distance to the target
                    delta_x = x_target - self.x
                    delta_y = y_target - self.y
                    distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

                    
                twist_msg.linear.x = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                time.sleep(2)
                self.get_logger().info(f"Im stopped")


    def strategy(self):
        """ Decision and action layers """
        if not hasattr(self, 'rotation_start_time'):
            self.rotation_start_time = self.get_clock().now()

        rotate_cmd = Twist()
        rotate_cmd.angular.z = 0.4  # Vitesse angulaire pour tourner à 360 degrés
        self.cmd_vel_pub.publish(rotate_cmd)

        # Si le temps de début de rotation est défini, vérifiez si le tour est terminé
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.rotation_start_time).nanoseconds/1e9  # Temps écoulé en secondes
        if elapsed_time >= 2 * math.pi/0.4:  # Si le temps écoulé est supérieur ou égal à 2 * pi (environ un tour complet)
            stop_cmd = Twist()  # Commande pour arrêter les mouvements
            self.cmd_vel_pub.publish(stop_cmd)
            
            best_centroids, best_path = self.find_best_centroid()
            self.movetobestcentroid(best_path)

            self.strategy_called = True
            if self.strategy_called:
                self.strategy_timer.cancel()


def main():
    rclpy.init()

    node = Agent()
    node.strategy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()