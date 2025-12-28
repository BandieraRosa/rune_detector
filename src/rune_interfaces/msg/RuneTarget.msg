uint8 UNKNOWN=0
uint8 STRUCK=1
uint8 UNSTRUCK=2
uint8 PENDING_STRUCK=3

std_msgs/Header header
bool tracking

# 0..4
uint8 target_index
uint8 target_type

# seconds, how far we predicted ahead
float32 predict_dt

# target pose in header.frame_id (output_frame, e.g. gimbal_odom)
geometry_msgs/Pose target_pose