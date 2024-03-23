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
        self.create_timer(0.5, self.strategy)      #0.5s of period <=> 2 Hz
        self.create_timer(1, self.publish_maps) #1Hz
    

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
        # self.get_logger().info(f"Agent 1: ({x:.2f}, {y:.2f})")
    

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

        if self.map[int(PosAgent[0]*2+20), int(PosAgent[1])*2+20] == OBSTACLE_VALUE:
            self.map[int(PosAgent[0]*2+20), int(PosAgent[1])*2+20] = FREE_SPACE_VALUE
            

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


    # Define function to check if a point is a frontier point
    def is_frontier_point(self, x, y):
        frontier_x = []
        frontier_y  = []
        indices_x = [-1, 0, 1]
        indices_y = [-1, 0, 1]
        if x <=39 and x >=0 and y <=39 and y >= 0:
            if self.map[x, y] == FREE_SPACE_VALUE:
                for i in indices_x:
                    for j in indices_y:
                        if (x+i) <= 39 and (x+i) >= 0 and (y+j) <= 39 and (y+j) >= 0:
                            if self.map[x+i, y+j] == UNEXPLORED_SPACE_VALUE:
                                frontier_x.append(x+i)
                                frontier_y.append(y+j)
        
        return frontier_x, frontier_y
    
    def explore_environment(self):
        PosAgent = np.dot(np.array([[0,-1],[1,0]]),np.array([self.x,self.y]))
        
        all_frontier_x = []
        all_frontier_y = []
        indices_x = list(range(-15, 15))
        indices_y = list(range(-15, 15))
        for i in indices_x:
            for j in indices_y:
                frontier_x, frontier_y = self.is_frontier_point(int(PosAgent[0]*2+20)+i, int(PosAgent[0]*2+20)+j)
                if frontier_x and frontier_y:
                    all_frontier_x.append(frontier_x)
                    all_frontier_y.append(frontier_y)

        for k in range(len(all_frontier_x)):
            for j in range(len(all_frontier_x[k])):
                self.map[all_frontier_x[k][j], all_frontier_y[k][j]] = FRONTIER
        
        return all_frontier_x, all_frontier_y


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
        all_frontier_x, all_frontier_y = self.explore_environment()

        all_frontier = []
        for k in range(len(all_frontier_x)):
            for j in range(len(all_frontier_x[k])):
                all_frontier.append([all_frontier_x[k][j], all_frontier_y[k][j]])

        grouped_frontiers = self.group_frontier(all_frontier)

        return grouped_frontiers
    

    def centroid(self):
        sum_x = 0
        sum_y = 0
        centroids = []
        len_frontiers = []
        grouped_frontiers = self.frontier_find()
        #self.get_logger().info(f"len(grouped_frontiers) = : ({len(grouped_frontiers)})")
        #self.get_logger().info(f"grouped_frontiers = : ({grouped_frontiers})")
        for k in range(len(grouped_frontiers)):
            for j in range(len(grouped_frontiers[k])):
                sum_x += grouped_frontiers[k][j][0]
                sum_y += grouped_frontiers[k][j][1]
            len_frontiers.append(len(grouped_frontiers[k]))
            avg_x = int(sum_x/len(grouped_frontiers[k]))
            avg_y = int(sum_y/len(grouped_frontiers[k]))
            centroids.append([avg_x, avg_y])
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
        L1 = 2
        L2 = 2
        L3 = 2
        path_list = []
        cost_list = []

        # Orientation calculations
        for centroid in centroids:
            delta_x = centroid[0] - int(PosAgent[0])*2+20
            delta_y = centroid[1] - int(PosAgent[1])*2+20

            orientation = math.degrees(math.atan2(delta_y, delta_x))
        
            path = a_star_search(self.map, [int(PosAgent[0])*2+20, int(PosAgent[1])*2+20], centroid)
            path_list.append(path)
            self.get_logger().info(f"centroid => : ({centroid})")

            if path != None:
                cost = L1*len(path) - L2*len_frontiers[i] + L3*orientation
                cost_list.append(cost)
                self.get_logger().info(f"cost => : ({cost})")
                self.get_logger().info(f"path => : ({path})")
            i =+1
        if cost_list:
            minimum_cost = min(cost_list)
            index_of_minimum = cost_list.index(minimum_cost)
            best_centroid = centroids[index_of_minimum]
            self.get_logger().info(f"best_centroid => : ({best_centroid})")

            # COlor in green the best centroid for each agent
            self.map[best_centroid[0], best_centroid[1]] = BEST_CENTROID
        
            return best_centroid

        # size len(group) 
        # distance robot-centroid
        # orientation robot - centroid
        # cost = L1distance - L2size + L3orientation


    def strategy(self):
        """ Decision and action layers """

        if not hasattr(self, 'rotation_start_time'):
            # Si le temps de début de rotation n'est pas encore défini, définissez-le
            self.rotation_start_time = self.get_clock().now()
            rotate_cmd = Twist()
            rotate_cmd.angular.z = 0.4  # Vitesse angulaire pour tourner à 360 degrés
            self.cmd_vel_pub.publish(rotate_cmd)

        else:
            # Si le temps de début de rotation est défini, vérifiez si le tour est terminé
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.rotation_start_time).nanoseconds / 1e9  # Temps écoulé en secondes
            if elapsed_time >= 2 * math.pi/0.4:  # Si le temps écoulé est supérieur ou égal à 2 * pi (environ un tour complet)
                stop_cmd = Twist()  # Commande pour arrêter les mouvements
                self.cmd_vel_pub.publish(stop_cmd)

                best_centroid = self.find_best_centroid()
                # delattr(self, 'rotation_start_time')


def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()