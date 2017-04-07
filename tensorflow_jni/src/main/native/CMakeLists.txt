################################################################
# A minimal CMake file that is compatible with sbt-jni         #
#                                                              #
# All settings required by sbt-jni have been marked so, please #
# add/modify/remove settings to build your specific library.   #
################################################################

cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 11)

# Define project and related variables
# (required by sbt-jni) please use semantic versioning
#
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
include_directories(include)
include_directories(${JNI_INCLUDE_DIRS})

# Link directories
link_directories(lib)

# Sources
file(GLOB LIB_SRC
  "*.c"
  "*.cc"
  "*.cpp"
)

# Setup installation targets
# (required by sbt-jni) major version should always be appended to library name
#
set (LIB_NAME ${PROJECT_NAME}) # ${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED ${LIB_SRC})
target_link_libraries(${LIB_NAME} "tensorflow")
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION .)
