cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 11)

# Define project and related variables
project(tensorflow)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

# Setup JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
endif()

# Include directories
include_directories(.)
include_directories(./generated)
include_directories(./include)
include_directories(./ops)
include_directories(${JNI_INCLUDE_DIRS})

# Find Native TensorFlow Library to link
find_library(LIB_TENSORFLOW tensorflow HINTS ENV LD_LIBRARY_PATH)
if(NOT LIB_TENSORFLOW)
  message(FATAL_ERROR "Library `tensorflow` not found.")
endif()

find_library(LIB_TENSORFLOW_FRAMEWORK tensorflow_framework HINTS ENV LD_LIBRARY_PATH)
if(NOT LIB_TENSORFLOW_FRAMEWORK)
  message(FATAL_ERROR "Library `tensorflow_framework` not found.")
endif()

# Collect sources for the JNI and the op libraries

file(GLOB JNI_LIB_SRC
  "*.cc"
  "generated/*.cc"
  "include/tensorflow/c/*.cc"
  "include/tensorflow/core/distributed_runtime/*.cc"
)

file(GLOB OP_LIB_SRC
  "ops/*.cc"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# Setup installation targets
set(JNI_LIB_NAME "${PROJECT_NAME}_jni")
add_library(${JNI_LIB_NAME} MODULE ${JNI_LIB_SRC})
target_link_libraries(${JNI_LIB_NAME} ${LIB_TENSORFLOW} ${LIB_TENSORFLOW_FRAMEWORK})
install(TARGETS ${JNI_LIB_NAME} LIBRARY DESTINATION .)

set(OP_LIB_NAME "${PROJECT_NAME}_ops")
add_library(${OP_LIB_NAME} MODULE ${OP_LIB_SRC})
target_link_libraries(${OP_LIB_NAME} ${LIB_TENSORFLOW} ${LIB_TENSORFLOW_FRAMEWORK})
install(TARGETS ${OP_LIB_NAME} LIBRARY DESTINATION .)
