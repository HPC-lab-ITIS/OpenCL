cmake_minimum_required(VERSION 2.8)

project(vexcl_ex)

find_package(Boost COMPONENTS system thread chrono filesystem)

set(VEXCL_ROOT     $ENV{VEXCL_ROOT}    CACHE STRING "VexCL root")
set(CMAKE_MODULE_PATH ${VEXCL_ROOT}/cmake/opencl)
find_package(OpenCL REQUIRED)

MESSAGE(STATUS "Found VexCL: ${VEXCL_ROOT}")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${VEXCL_ROOT}
    ${Boost_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    )

add_definitions(-DBOOST_DISABLE_ASSERTS)

function(add_vexcl_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
endfunction()

add_vexcl_executable(ex_3  ex_3.cpp)
add_vexcl_executable(ex_4  ex_4.cpp)
add_vexcl_executable(test test.cpp)
