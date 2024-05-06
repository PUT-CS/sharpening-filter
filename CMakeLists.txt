cmake_minimum_required(VERSION 3.20)
project(sharpening-filter LANGUAGES CXX CUDA)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Enable generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the -O3 optimization flag and other flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -flto -ffast-math")

# Add CUDA flags and set architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")

# Add CUDA files to the project
file(GLOB CUDA_SOURCES src/*.cu)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Add executable with CUDA sources
add_executable(sharpening-filter src/main.cpp ${CUDA_SOURCES})
target_link_libraries(sharpening-filter PUBLIC ${OpenCV_LIBS})
