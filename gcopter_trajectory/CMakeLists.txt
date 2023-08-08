cmake_minimum_required(VERSION 3.0.2)
project(gcopter_trajectory)

find_package(Eigen3 REQUIRED)

find_package(catkin)
if(catkin_FOUND)
  include_directories(include ${catkin_INCLUDE_DIRS})
  catkin_package(INCLUDE_DIRS include)
  return()
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
