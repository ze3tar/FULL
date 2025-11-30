from setuptools import setup

package_name = "apf_rrt_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="FULL maintainers",
    maintainer_email="maintainers@example.com",
    description="ROS 2 wrappers for APF-RRT planners and path exporters",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "apf_rrt_path_publisher = apf_rrt_ros2.path_publisher:main",
        ],
    },
)
