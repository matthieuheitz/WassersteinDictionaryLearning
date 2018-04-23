# Author : Matthieu Heitz
# Date : 03/11/2016

cmake_minimum_required(VERSION 2.8)
project(inverseWasserstein)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

#######################
### Project options ###
#######################

option(WITH_HALIDE "Use Halide to compute convolutions" OFF)
option(WITH_EIGEN "Use Eigen to compute convolutions" OFF)
option(WITH_AVX_SUPPORT "Use AVX to vectorize convolutions" ON)
option(WITH_OPENMP "Use OpenMP" OFF)

option(BUILD_APP_DICTIONARY_LEARNING "Build the Dictionary Learning application" ON)


############################
### General dependencies ###
############################

IF(WITH_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  IF(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    ADD_DEFINITIONS("-DHAS_OPENMP ")
    message(STATUS "OpenMP found.")
  ELSE(OPENMP_FOUND)
    message(FATAL_ERROR "OpenMP support not available.")
  ENDIF(OPENMP_FOUND)
ENDIF(WITH_OPENMP)


set(INCLUDE_DIRS
    lbfgs/
)

set(ADDITIONAL_SOURCE_FILES
    ${ADDITIONAL_SOURCE_FILES}
    lbfgs/lbfgs.c)

if(WITH_HALIDE)
    message(STATUS "WITH_HALIDE")
    find_package(Halide REQUIRED)
    add_definitions(-DHAS_HALIDE)
    include_directories(${HALIDE_INCLUDE_DIR})
    set(ADDITIONAL_LINKER_FLAG ${ADDITIONAL_LINKER_FLAG} ${HALIDE_LIBRARY_DIR} -ldl -lncurses -lz)
endif(WITH_HALIDE)

if(WITH_EIGEN)
    message(STATUS "WITH_EIGEN")
    set(EIGEN_INCLUDE_DIR_HINTS "NO_HINT" CACHE STRING "Hints to Eigen include directory")
    find_package(Eigen REQUIRED)
    add_definitions(-DHAS_EIGEN)
    include_directories(${EIGEN_INCLUDE_DIRS})
endif(WITH_EIGEN)

if(WITH_AVX_SUPPORT)
    add_definitions(-DAVX_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mtune=native")
endif(WITH_AVX_SUPPORT)


#########################################
### Application-specific dependencies ###
#########################################

# Applications that need CImg
if(BUILD_APP_DICTIONARY_LEARNING)

    set(INCLUDE_DIRS
        ${INCLUDE_DIRS}
        cimg/
    )
    set(ADDITIONAL_SOURCE_FILES
        ${ADDITIONAL_SOURCE_FILES}

    )
    set(ADDITIONAL_LINKER_FLAG ${ADDITIONAL_LINKER_FLAG} -lpthread)
    # Set display library type : 0: no display, 1:  X11 based, 2: Windows-GDI
    add_definitions(-Dcimg_display=0)
endif(BUILD_APP_DICTIONARY_LEARNING)


include_directories(${INCLUDE_DIRS})

#####################
### Build targets ###
#####################

if(BUILD_APP_DICTIONARY_LEARNING)
    set(SOURCE_FILES
        main_dictionary_learning.cpp
    )
    add_executable(app_dictionary_learning ${SOURCE_FILES} ${ADDITIONAL_SOURCE_FILES})
    target_link_libraries(app_dictionary_learning ${ADDITIONAL_LINKER_FLAG})
endif(BUILD_APP_DICTIONARY_LEARNING)
