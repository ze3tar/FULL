from setuptools import setup
from glob import glob

package_name = 'full_integration'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/srv', glob('srv/*.srv')),
    ],
    install_requires=['setuptools', 'numpy', 'rclpy', 'trajectory_msgs'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@example.com',
    description='PPO-based trajectory refinement and integration utilities.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ppo_service_node = full_integration.ppo_service_node:main',
            'trajectory_refiner = full_integration.trajectory_refiner:main',
        ],
    },
)
