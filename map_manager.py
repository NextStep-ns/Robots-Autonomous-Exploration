__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2024, IN424, IPSA 2024"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import numpy as np
from .my_common import *    #common variables are stored here



class MapManager(Node):
    """ This class is used to merge maps from all the agents and display it on RVIZ and for the agents """
    def __init__(self):
        Node.__init__(self, "map_manager")

        self.load_params()
        self.init_map()

        self.map_agents_pub = self.create_publisher(OccupancyGrid, "/merged_map", 1)
        self.map_rviz_pub = self.create_publisher(OccupancyGrid, "/map", 1)

        for i in range(1, self.nb_agents+1):   #subscribe to agents' map topic
            self.create_subscription(OccupancyGrid, f"/bot_{i}/map", self.agent_map_cb, 1)
        
        self.create_timer(1, self.publish_maps)
    

    def load_params(self):
        """ Load parameters from launch file """
        #Get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("nb_agents", rclpy.Parameter.Type.INTEGER),
                ("robot_size", rclpy.Parameter.Type.DOUBLE),
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY)
            ]
        )
        self.nb_agents = self.get_parameter("nb_agents").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
    

    def init_map(self):
        """ Initialize maps to publish """
        self.map_agents_msg = OccupancyGrid()
        self.map_agents_msg.header.frame_id = "map" #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_agents_msg.header.stamp = self.get_clock().now().to_msg()  #get the current ROS time to send the msg
        self.map_agents_msg.info.resolution = self.robot_size   #Map cell size corresponds to robot size
        self.map_agents_msg.info.height = int(self.env_size[0]/self.map_agents_msg.info.resolution) #nb of rows
        self.map_agents_msg.info.width = int(self.env_size[1]/self.map_agents_msg.info.resolution)  #nb of columns
        self.map_agents_msg.info.origin.position.x = -self.env_size[1]/2    #x and y coordinates of the origin in map reference frame
        self.map_agents_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_agents_msg.info.origin.orientation.w = 1.0 #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.merged_map = np.ones(shape=(self.map_agents_msg.info.height, self.map_agents_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE    #all the cells are unexplored initially
        self.w, self.h = self.map_agents_msg.info.width, self.map_agents_msg.info.height
        #Same for RVIZ map
        self.map_rviz_msg = OccupancyGrid()
        self.map_rviz_msg.header = self.map_agents_msg.header
        self.map_rviz_msg.info = self.map_agents_msg.info

    
    def agent_map_cb(self, msg):
        """ 
            Get new maps from agent and merge them.
            This method is automatically called whenever a new message is published on one of the following topics:
                /bot_1/map
                /bot_2/map
                /bot_3/map
            'msg' is a nav_msgs/msg/OccupancyGrid message
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (self.merged_map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                    self.merged_map[i, j] = received_map[i, j]

                elif ((self.merged_map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.merged_map[i, j] == FREE_SPACE_VALUE) or (self.merged_map[i, j] == FRONTIER)) and (received_map[i, j] == CENTROID):
                    self.merged_map[i, j] = received_map[i, j]
                
                elif ((self.merged_map[i, j] == UNEXPLORED_SPACE_VALUE) or (self.merged_map[i, j] == FREE_SPACE_VALUE) or (self.merged_map[i, j] == FRONTIER) or (self.merged_map[i, j] == CENTROID)) and (received_map[i, j] == BEST_CENTROID):
                    self.merged_map[i, j] = received_map[i, j]
                    
    def publish_maps(self):
        """ Publish maps on corresponding topics """
        self.map_rviz = self.merged_map.copy()
        #TODO: add frontiers on rviz map

        self.map_agents_msg.data = np.flipud(self.merged_map).flatten().tolist()    #transform the 2D array into a list to publish it
        self.map_rviz_msg.data = np.flipud(self.map_rviz).flatten().tolist()    #transform the 2D array into a list to publish it

        self.map_agents_pub.publish(self.map_agents_msg)    #publish the merged map to other agents on topic /merged_map
        self.map_rviz_pub.publish(self.map_rviz_msg)    #publish the merged map to RVIZ2 on topic /map



def main():
    rclpy.init()

    node = MapManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()