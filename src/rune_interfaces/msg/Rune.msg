std_msgs/Header header
uint8 type                          # 0=UNKNOWN, 1=UNSTRUCK, 2=PENDING, 3=STRUCK
geometry_msgs/Point center_2d       # 图像中心
geometry_msgs/Pose pose             # 3D位姿
float32 angle                       # 当前转角(度)
float32 confidence