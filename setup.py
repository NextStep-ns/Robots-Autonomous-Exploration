from setuptools import setup
import os
from glob import glob

package_name = 'in424_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share/" + package_name), glob("launch/*_launch.py"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Johvany Gustave',
    maintainer_email='johvany.gustave@ipsa.fr',
    description='This package contains the approach implemented for the agents',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "tf_publishers = in424_nav.tf_publishers:main",
            "agent = in424_nav.agent:main",
            "map_manager = in424_nav.map_manager:main"
        ],
    },
)
