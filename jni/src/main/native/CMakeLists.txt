cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 11)

# Define project and related variables
project (tensorflow_jni)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

# Setup JNI
find_package(JNI REQUIRED)
if (JNI_FOUND)
    message (STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
endif()

# Include directories
include_directories(.)
include_directories(generated)
include_directories(${JNI_INCLUDE_DIRS})

# Find Native TensorFlow Library to Link
find_library(LIBTENSORFLOW tensorflow HINTS ${LD_LIBRARY_PATH})
if(NOT LIBTENSORFLOW)
  message(FATAL_ERROR "Library `tensorflow` not found.")
endif()

# Sources
file(GLOB_RECURSE LIB_SRC
  "*.c"
  "*.cc"
  "*.cpp"
)

# Setup installation targets
set (LIB_NAME ${PROJECT_NAME}) # ${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED ${LIB_SRC})
target_link_libraries(${LIB_NAME} ${LIBTENSORFLOW})
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION .)
