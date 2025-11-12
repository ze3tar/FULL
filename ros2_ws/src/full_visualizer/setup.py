from setuptools import setup

package_name = 'full_visualizer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/full_viz.launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/full_robot.urdf']),
        ('share/' + package_name + '/rviz', ['rviz/full.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ChatGPT',
    maintainer_email='example@example.com',
    description='Launch files and nodes to visualize the FULL robot in RViz.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chatgpt_codex_node = full_visualizer.chatgpt_codex_node:main',
        ],
    },
)
