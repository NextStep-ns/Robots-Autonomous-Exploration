from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nb_agents = 3
    robot_size = 0.5
    env_size = [20, 20]

    agent_nodes = []
    for i in range(nb_agents):
        agent_nodes.append(
            Node(
                package = "in424_nav",
                executable = "agent",
                parameters = [
                    {"ns": f"bot_{i+1}"},
                    {"robot_size": robot_size}, #robot's size/diameter in meter,
                    {"env_size": env_size},  #height and width of the environment in meters
                    {"nb_agents": nb_agents}  #number of agents in the environment
                ]
            )
        )

    ld = LaunchDescription()

    for agent_node in agent_nodes[:nb_agents]:
        ld.add_action(agent_node)
    
    ld.add_action(
        Node(
            package = "in424_nav",
            executable = "map_manager",
            parameters = [
                {"robot_size": robot_size}, #robot's size/diameter in meter,
                {"env_size": env_size},  #height and width of the environment in meters
                {"nb_agents": nb_agents}  #number of agents in the environment
            ]
        )
    )

    return ld