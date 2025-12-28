uint8 UNKNOWN=0
uint8 STRUCK=1
uint8 UNSTRUCK=2
uint8 PENDING_STRUCK=3

std_msgs/Header header
bool valid

# Pose of rune group frame w.r.t header.frame_id (typically camera_optical_frame)
geometry_msgs/Pose group_pose

# 5 runes in fixed model order (0..4)
uint8[5] rune_type