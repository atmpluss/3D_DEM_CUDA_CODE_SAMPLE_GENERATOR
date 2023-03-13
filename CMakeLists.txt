cmake_minimum_required(VERSION 3.19)
project(3_cuda_settling_particles CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(3_cuda_settling_particles main.cu)

set_target_properties(
        3_cuda_settling_particles
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)