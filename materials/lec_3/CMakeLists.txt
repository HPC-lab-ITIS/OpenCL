cmake_minimum_required(VERSION 2.8)
project(vexcl_ex)

find_package(VexCL)

add_definitions(-DVEXCL_CACHE_KERNELS)

add_executable(vexcl_hello  vexcl_hello.cpp)
add_executable(vexcl_func_cl  vexcl_func.cpp)
add_executable(vexcl_func_cu  vexcl_func.cpp)

target_link_libraries(vexcl_hello VexCL::OpenCL)
target_link_libraries(vexcl_func_cl VexCL::OpenCL)
target_link_libraries(vexcl_func_cu VexCL::CUDA)
