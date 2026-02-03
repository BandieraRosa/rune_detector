cmake_minimum_required(VERSION 3.10)
project(rune_tracker)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===========================
# 查找依赖
# ===========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rune_interfaces REQUIRED)
find_package(rm_vision_core REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ===========================
# 创建组件库
# ===========================
add_library(${PROJECT_NAME}_component SHARED src/rune_tracker_node.cpp)

target_include_directories(
  ${PROJECT_NAME}_component
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

# 在 ament_target_dependencies 中添加 tf2 相关依赖
ament_target_dependencies(
  ${PROJECT_NAME}_component
  rclcpp
  rclcpp_components
  geometry_msgs
  rune_interfaces
  tf2
  tf2_ros
  tf2_geometry_msgs)

# 链接 rm_vision_core
target_link_libraries(${PROJECT_NAME}_component rm_vision_core::rm_vision_core)

# 注册组件
rclcpp_components_register_node(
  ${PROJECT_NAME}_component PLUGIN "RuneTrackerNode" EXECUTABLE
  ${PROJECT_NAME}_node)

# ===========================
# 安装
# ===========================
install(
  TARGETS ${PROJECT_NAME}_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
endif()

# ===========================
# 测试
# ===========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_component)

# 在导出依赖中也添加 tf2 相关包
ament_export_dependencies(
  rclcpp
  rclcpp_components
  geometry_msgs
  rune_interfaces
  rm_vision_core
  tf2
  tf2_ros
  tf2_geometry_msgs)

ament_package()
