cmake_minimum_required(VERSION 3.20)
project(sharpening-filter LANGUAGES CXX)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the -O3 optimization flag and other flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -Wpedantic -flto -ffast-math")

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}") # Add additional flags if needed
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(sharpening-filter src/main.cpp)
target_link_libraries(sharpening-filter PUBLIC ${OpenCV_LIBS})
