project(object_location)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "DRIVER find OpenCV:\"${OpenCV_INCLUDE_DIRS}\",ENABLE_OpenCV macro is on")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    add_definitions(${OpenCV_DEFINITIONS})
endif (OpenCV_FOUND)

find_package(PCL REQUIRED)
if (PCL_FOUND)
    message(STATUS "DRIVER find PCL:\"${PCL_INCLUDE_DIRS}\",ENABLE_PCL macro is on")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif (PCL_FOUND)

find_package(Eigen3 REQUIRED)

include_directories(
    include
)

add_library(localMatcher SHARED src/local_matcher.cpp)
add_library(globalMatcher SHARED src/global_matcher.cpp)

add_executable(test_match test/test_match.cpp)
add_executable(test_recognize test/test_recognize.cpp)

target_link_libraries(test_match localMatcher ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} config++)
target_link_libraries(test_recognize globalMatcher ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} config++)

