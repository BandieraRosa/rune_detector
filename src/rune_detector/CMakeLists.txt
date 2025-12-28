cmake_minimum_required(VERSION 3.10)
project(rune_detector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 关键：让库默认用 -fPIC

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rune_interfaces REQUIRED)

# -----------------------
# Build rm_vision_core as an internal static library (PIC enabled)
# -----------------------
set(RM_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rm_vision_core)

file(GLOB_RECURSE RM_VISION_CORE_SOURCES ${RM_CORE_DIR}/common/*.cpp
     ${RM_CORE_DIR}/modules/*.cpp)

file(GLOB RM_VISION_CORE_INCLUDE_DIRS CONFIGURE_DEPENDS
     ${RM_CORE_DIR}/common/*/include ${RM_CORE_DIR}/common/modules/*/include
     ${RM_CORE_DIR}/modules/*/*/include)
list(FILTER RM_VISION_CORE_INCLUDE_DIRS INCLUDE REGEX ".*/include$")
list(REMOVE_DUPLICATES RM_VISION_CORE_INCLUDE_DIRS)

add_library(rm_vision_core STATIC ${RM_VISION_CORE_SOURCES})
set_target_properties(rm_vision_core PROPERTIES POSITION_INDEPENDENT_CODE ON
)# 双保险

target_compile_definitions(rm_vision_core PUBLIC VC_DISABLE_AUTO_YML=1)
target_include_directories(rm_vision_core PUBLIC ${RM_VISION_CORE_INCLUDE_DIRS})

# 注意：这里用 plain 形式，别用 PUBLIC/PRIVATE，避免和 ament_* 混用
target_link_libraries(rm_vision_core ${OpenCV_LIBS})

# -----------------------
# Build component library (shared) and register node executable
# -----------------------
add_library(${PROJECT_NAME} SHARED src/rune_detector_node.cpp)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rune_detector> # 兼容
                                                                         # #include
                                                                         # "rune_detector_node.h"
    $<INSTALL_INTERFACE:include>)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  geometry_msgs
  image_transport
  cv_bridge
  message_filters
  tf2
  tf2_geometry_msgs
  rune_interfaces)

# 仍然用 plain 形式，保持一致
target_link_libraries(${PROJECT_NAME} rm_vision_core ${OpenCV_LIBS})

# 这句会自动 add_executable(rune_detector_node ...) —— 所以你不要再手写 add_executable 了
rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN rune_detector::RuneDetectorNode EXECUTABLE
  rune_detector_node)

install(
  TARGETS rm_vision_core ${PROJECT_NAME} rune_detector_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_package()
