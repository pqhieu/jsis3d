cmake_minimum_required (VERSION 2.6)
project(densecrf)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories( include )

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (external)
