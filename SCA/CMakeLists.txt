cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sca)

find_package(Torch REQUIRED)

add_executable(sca sca.cpp)
target_link_libraries(sca "${TORCH_LIBRARIES}")
set_property(TARGET sca PROPERTY CXX_STANDARD 14)
