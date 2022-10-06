cmake_minimum_required(VERSION 3.10)
project(kalman)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        pcl_ros
        pcl_conversions
        )

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)

find_package(Eigen3 QUIET)
if (NOT EIGEN3_FOUND)
    # Fallback to cmake_modules
    find_package(cmake_modules REQUIRED)
    find_package(Eigen REQUIRED)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
    set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
    # Possibly map additional variables to the EIGEN3_ prefix.
else ()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-O3 -g -Wall ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)

catkin_package(
        CATKIN_DEPENDS
        roscpp
        std_msgs
        sensor_msgs
        pcl_ros
        pcl_conversions
)



include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    include/kalman
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# kalman
add_executable(kalman
        src/kalman.cpp
        )
add_dependencies(kalman
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(kalman
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
install(TARGETS kalman
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


# kalman_clustering
add_executable(kalman_clustering
src/kalman_clustering.cpp
)
add_dependencies(kalman_clustering
${catkin_EXPORTED_TARGETS}
)
target_link_libraries(kalman_clustering
${catkin_LIBRARIES}
${PCL_LIBRARIES}
)
install(TARGETS kalman_clustering
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
