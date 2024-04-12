__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2024, IN424, IPSA 2024"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class TFPublishers(Node):
    """
        This class is used to publish on TF the current pose of each agent, based on odometry information provided by the simulator.
        As poses are published on TF, you can visualize the agents on RVIZ2.
    """
    def __init__(self):
        Node.__init__(self, "odom_tf_publishers")
        self.declare_parameter("nb_agents", 2)
        self.tf_br = TransformBroadcaster(self)
        self.nb_agents = self.get_parameter("nb_agents").get_parameter_value().integer_value
        self.agent_poses = [None]*self.nb_agents

        self.create_subscription(Odometry, "/bot_1/odom", self.odom1_cb, 1)
        if self.nb_agents >= 2:
            self.create_subscription(Odometry, "/bot_2/odom", self.odom2_cb, 1)
        if self.nb_agents == 3:
            self.create_subscription(Odometry, "/bot_3/odom", self.odom3_cb, 1)
        self.create_timer(0.1, self.manage_tf)


    def odom1_cb(self, msg):
        """ Get odometry msgs of agent 1 """
        self.agent_poses[0] = msg
    

    def odom2_cb(self, msg):
        """ Get odometry msgs of agent 2 """
        self.agent_poses[1] = msg


    def odom3_cb(self, msg):
        """ Get odometry msgs of agent 3 """
        self.agent_poses[2] = msg
        
    
    def manage_tf(self):
        """ Process incoming odometry msgs to publish TF msgs for all the agents """
        for i, pose in enumerate(self.agent_poses[:self.nb_agents]):
            if pose != None:
                self.publish_tf(i+1, pose)


    def publish_tf(self, agent_id, msg):
        """ Publish agent pose on TF """
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = f"map"
        transform.child_frame_id = f"bot_{agent_id}/base_footprint"

        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z

        transform.transform.rotation.x = msg.pose.pose.orientation.x
        transform.transform.rotation.y = msg.pose.pose.orientation.y
        transform.transform.rotation.z = msg.pose.pose.orientation.z
        transform.transform.rotation.w = msg.pose.pose.orientation.w
        self.tf_br.sendTransform(transform)



def main():
    rclpy.init()

    node = TFPublishers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()