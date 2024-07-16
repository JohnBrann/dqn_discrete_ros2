from setuptools import find_packages, setup

package_name = 'dqn_discrete_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='csrobot',
    maintainer_email='noahrothgaber@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'environment_publisher = dqn_discrete_ros2.environment_publisher:main',
            'ros_gym_environment_helper = dqn_discrete_ros2.ros_gym_environment_helper:main'
        ],
    },
)
