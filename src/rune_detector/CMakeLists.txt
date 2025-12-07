cmake_minimum_required(VERSION 3.8)
project(rune_detector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ##############################################################################
# Find dependencies
# ##############################################################################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

ament_auto_find_build_dependencies()

# ##############################################################################
# Build
# ##############################################################################
ament_auto_add_library(${PROJECT_NAME} SHARED DIRECTORY src)

# 链接库
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)

# 注册组件
rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN rune_detector::RuneDetectorNode EXECUTABLE
  rune_detector_node)

# ##############################################################################
# Install
# ##############################################################################

# 配置头文件安装路径
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

# ##############################################################################
# Package
# ##############################################################################
ament_auto_package(INSTALL_TO_SHARE)
