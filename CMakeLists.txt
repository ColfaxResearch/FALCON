cmake_minimum_required(VERSION 2.8.9)
project(FALCON)


set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

find_library(MEMKIND_LIB memkind /usr/lib64 DEFAULT_PATH)
if(MEMKIND_LIB)
else(MEMKIND_LIB)
    unset(MEMKIND_LIB_FOUND)
    message(FATAL_ERROR "Cannot find MEMKIND RUNTIME LIBRARY")
endif (MEMKIND_LIB)

set (CMAKE_C_COMPILER "icc")
set (CMAKE_C_FLAGS "-O2 -xHost -restrict -qopenmp ")
set(FALCON_LIB_SRCS "${CMAKE_SOURCE_DIR}/src/falcon.c")
include_directories(include)

add_library(falcon SHARED ${FALCON_LIB_SRCS})
target_link_libraries(falcon ${BLAS_LIBRARIES} ${MEMKIND_LIB})

set(EXAMPLE_SRCS "${CMAKE_SOURCE_DIR}/example/vgg_winograd.c")

add_executable(vgg_winograd ${EXAMPLE_SRCS})
target_link_libraries(vgg_winograd falcon ${BLAS_LIBRARIES} ${MEMKIND_LIB})


