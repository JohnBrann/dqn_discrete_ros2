import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/csrobot/pytorch_ws/src/dqn_discrete_ros2/install/dqn_discrete_ros2'
