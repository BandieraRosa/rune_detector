cmake_minimum_required(VERSION 3.8)
project(rune_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(rune_interfaces REQUIRED)
find_package(rune_detector REQUIRED)  # 复用 rune_detector 导出的内容/库（如 rm_vision_core 等）

# -----------------------
# Component library (shared)
# -----------------------
add_library(${PROJECT_NAME} SHARED
  src/rune_tracker_node.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  rclcpp_components
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  rune_interfaces
  rune_detector
)

# 生成可执行：rune_tracker_node
# ⚠️ 把 PLUGIN 替换成你实际的 Node 类型（namespace::ClassName）
rclcpp_components_register_node(
  ${PROJECT_NAME}
  PLUGIN "rune_tracker::RuneTrackerNode"
  EXECUTABLE rune_tracker_node
)

# -----------------------
# Install
# -----------------------
install(
  TARGETS ${PROJECT_NAME} rune_tracker_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 如果你有头文件目录 include/（推荐保留）
install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()
