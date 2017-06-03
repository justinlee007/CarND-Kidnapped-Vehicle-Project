cmake_minimum_required(VERSION 3.7)
project(CarND_Particle_Filter)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/particle_filter.cpp src/main.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

add_executable(CarND_Particle_Filter ${sources})

target_link_libraries(CarND_Particle_Filter z ssl uv uWS)

