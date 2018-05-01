cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Define project and related variables
project(horovod)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 12)
set(PROJECT_VERSION_PATCH 1)

# Set up JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
endif()

# Set up MPI
set(MPI_HOME "/usr")
find_package(MPI REQUIRED)
if(MPI_FOUND)
  message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
endif()

# Set up CUDA
find_package(CUDA)
if(CUDA_FOUND)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  find_library(NCCL_STATIC_LIBRARY NAMES libnccl_static.a PATHS ${CUDA_TOOLKIT_ROOT_DIR} ${HOROVOD_NCCL_HOME}/lib ${HOROVOD_NCCL_HOME}/lib64)
  if (NCCL_STATIC_LIBRARY)
    message(STATUS "Found NCCL.")
    add_definitions(-DHAVE_NCCL)
    add_definitions(-DHOROVOD_GPU_ALLREDUCE=N)
  else()
    add_definitions(-DHOROVOD_GPU_ALLREDUCE)
  endif()
endif()

# Include directories
include_directories(.)
include_directories(./common)
include_directories(./ops)

# Needed for the TensorFlow header files.
include_directories(../../../../jni/src/main/native/include)
include_directories(../../../../../../jni/src/main/native/include)
include_directories(/tensorflow_scala/jni/src/main/native/include)

include_directories(${JNI_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

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
  "common/*.cc"
  "common/wire/*.cc"
  "common/wire/flatbuffers/*.cc"
  "ops/*.cc"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

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
target_link_libraries(${JNI_LIB_NAME} ${LIB_TENSORFLOW} ${LIB_TENSORFLOW_FRAMEWORK} ${MPI_LIBRARIES})
if(CUDA_FOUND)
  target_link_libraries(${JNI_LIB_NAME} ${CUDA_LIBRARIES})
  if (NCCL_STATIC_LIBRARY)
    target_link_libraries(${JNI_LIB_NAME} ${NCCL_STATIC_LIBRARY})
  endif()
endif()
install(TARGETS ${JNI_LIB_NAME} LIBRARY DESTINATION .)
