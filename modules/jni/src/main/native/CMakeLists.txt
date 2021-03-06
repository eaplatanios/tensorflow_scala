cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Set the minimum OS X deployment version
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X Deployment Version")

# Define project and related variables
project(tensorflow)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 6)
set(PROJECT_VERSION_PATCH 0)

add_definitions(-DNOMINMAX)

# Set up JNI
if(NOT JAVA_HOME)
  set(JAVA_HOME "/Users/eaplatanios/Library/Java/JavaVirtualMachines/adopt-openjdk-11.0.7/Contents/Home")
endif()
find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
endif()

# Include directories
include_directories(.)
include_directories(./generated)
include_directories(./include)
include_directories(./include/third_party)
include_directories(./ops)
include_directories(${JNI_INCLUDE_DIRS})

# Find Native TensorFlow Library to link
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2.4.0 .so.2 .so .2.4.0.dylib .2.dylib .dylib .lib)
find_library(LIB_TENSORFLOW tensorflow HINTS ENV LD_LIBRARY_PATH)
if(NOT LIB_TENSORFLOW)
  message(FATAL_ERROR "Library `tensorflow` not found.")
endif()
if(NOT WIN32)
  find_library(LIB_TENSORFLOW_FRAMEWORK tensorflow_framework HINTS ENV LD_LIBRARY_PATH)
  if(NOT LIB_TENSORFLOW_FRAMEWORK)
    message(FATAL_ERROR "Library `tensorflow_framework` not found.")
  endif()
else()
  set(LIB_TENSORFLOW_FRAMEWORK "")
endif()

# Collect sources for the JNI and the op libraries
file(GLOB JNI_LIB_SRC "*.cc" "generated/*.cc")
file(GLOB OP_LIB_SRC "ops/*.cc")

# Find Python and get the TensorFlow header files from there.
if(WIN32)
  set(PYTHON python)
else()
  set(PYTHON python3)
endif()
set(TF_CPP_MIN_LOG_LEVEL 3)
execute_process(COMMAND ${PYTHON} -c
  "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))"
  OUTPUT_VARIABLE TF_CFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON} -c
  "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))"
  OUTPUT_VARIABLE TF_LFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS} ${TF_LFLAGS}")

set(CMAKE_BUILD_WITH_INSTALL_RPATH 1)
if(${APPLE})
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_INSTALL_NAME_DIR "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/")
endif()

# Setup installation targets
set(JNI_LIB_NAME "${PROJECT_NAME}_jni")
add_library(${JNI_LIB_NAME} MODULE ${JNI_LIB_SRC})
target_link_libraries(${JNI_LIB_NAME} ${LIB_TENSORFLOW} ${LIB_TENSORFLOW_FRAMEWORK})
install(TARGETS ${JNI_LIB_NAME} LIBRARY DESTINATION .)

set(OP_LIB_NAME "${PROJECT_NAME}_ops")
add_library(${OP_LIB_NAME} MODULE ${OP_LIB_SRC})
target_link_libraries(${OP_LIB_NAME} ${LIB_TENSORFLOW} ${LIB_TENSORFLOW_FRAMEWORK})
install(TARGETS ${OP_LIB_NAME} LIBRARY DESTINATION .)
