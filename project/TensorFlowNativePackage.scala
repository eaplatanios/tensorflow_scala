/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import java.io.PrintWriter
import java.nio.file.Files

import scala.collection.JavaConverters._

import sbt._
import sbt.Keys._
//import sys.process._

/**
  *
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowNativePackage extends AutoPlugin {
  override def requires: Plugins = plugins.JvmPlugin

  val supportedPlatforms = Map(
    "linux-x86_64" -> "x86_64-linux-gnu",
    "darwin-x86_64" -> "x86_64-apple-darwin14",
    "windows-x86_64" -> "x86_64-w64-mingw32")

  object autoImport {
    val nativePlatforms: SettingKey[Set[String]] =
      settingKey[Set[String]]("###")

    val projectRoot: SettingKey[String] =
      settingKey[String]("###")

    val tensorFlowBinaryVersion: SettingKey[String] =
      settingKey[String]("###")

    val nativeCrossCompile: TaskKey[Map[String, Set[File]]] =
      taskKey[Map[String, Set[File]]]("###")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    nativePlatforms in nativeCrossCompile := supportedPlatforms.keySet,
    tensorFlowBinaryVersion in nativeCrossCompile := "nightly",
    target in nativeCrossCompile := target.value / "native",
    clean in nativeCrossCompile := {
      streams.value.log.info("Cleaning generated cross compilation files.")
      val path = (target in nativeCrossCompile).value.toPath
      if (Files.exists(path))
        Files.walk(path)
            .iterator()
            .asScala
            .toSeq
            .reverse
            .foreach(Files.deleteIfExists)
    },
    nativeCrossCompile := {
      val targetDir = (target in nativeCrossCompile).value
      IO.createDirectory(targetDir)
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        val log = streams.value.log
        log.info(s"Cross-compiling '${name.value}' for platform '$platform'.")
        log.info(s"Using ${baseDirectory.value} as the base directory.")
        val platformTargetDir = targetDir / platform
        IO.createDirectory(platformTargetDir / "docker")
        val dockerfile = platformTargetDir / "docker" / "Dockerfile"
        log.info(s"Generating Dockerfile in '$dockerfile'.")
        new PrintWriter(dockerfile) {
          write(generateDockerfile(platform, "tensorflow_scala", scalaVersion.value, sbtVersion.value))
          close()
        }
        val cMakeLists = platformTargetDir / "docker" / "CMakeLists.txt"
        log.info(s"Generating CMakeLists in '$cMakeLists'.")
        new PrintWriter(cMakeLists) {
          write(generateCMakeListsFile(platform))
          close()
        }
        val crossCompileScript = platformTargetDir / "docker" / "script.sh"
        log.info(s"Generating script in '$crossCompileScript'.")
        new PrintWriter(crossCompileScript) {
          write(generateCrossCompileScript(
            platform,
            "tensorflow_scala",
            s"${name.value}",
            baseDirectory.value.getPath,
            platformTargetDir.getPath,
            (tensorFlowBinaryVersion in nativeCrossCompile).value
          ))
          close()
        }
        log.info(s"Generating binaries in '$platformTargetDir'.")
        val exitCode = Process(s"bash $crossCompileScript") ! log
        if (exitCode != 0) sys.error(s"An error occurred while cross-compiling for '$platform'. Exit code: $exitCode.")
        platform -> (platformTargetDir ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet
      }).toMap
    },
    // Make the SBT clean task also cleans the generated cross-compilation files
    clean := clean.dependsOn(clean in nativeCrossCompile).value
  )

  override lazy val projectSettings: Seq[Setting[_]] = settings

  def generateDockerfile(platform: String, tmpFolderName: String, scalaVersion: String, sbtVersion: String): String = {
    s"""
       |# Pull base image
       |FROM multiarch/crossbuild
       |
       |# Install CMake and Java
       |RUN echo "deb http://httpredir.debian.org/debian/ jessie-backports main" > \\
       |  /etc/apt/sources.list.d/jessie-backports.list
       |RUN apt-get update
       |RUN apt-get -t jessie-backports -y --no-install-recommends install cmake openjdk-8-jdk
       |RUN /usr/sbin/update-java-alternatives -s java-1.8.0-openjdk-amd64
       |ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
       |
       |# Define working directory
       |WORKDIR /root
     """.stripMargin
  }

  def generateCMakeListsFile(platform: String): String = {
    val cMakeSystemName = platform match {
      case "linux-x86_64" => "Linux"
      case "darwin-x86_64" => "Darwin"
      case "windows-x86_64" => "Windows"
    }
    val cMakeToolsPath = platform match {
      case "linux-x86_64" => "/usr"
      case "darwin-x86_64" | "windows-x86_64" => s"/usr/${supportedPlatforms(platform)}"
    }
    val cMakeCCompiler = platform match {
      case "linux-x86_64" => s"$cMakeToolsPath/bin/gcc"
      case "darwin-x86_64" => s"/usr/osxcross/bin/${supportedPlatforms(platform)}-clang"
      case "windows-x86_64" => s"$cMakeToolsPath/bin/gcc"
    }
    val cMakeCXXCompiler = platform match {
      case "linux-x86_64" => s"$cMakeToolsPath/bin/gcc"
      case "darwin-x86_64" => s"/usr/osxcross/bin/${supportedPlatforms(platform)}-clang++"
      case "windows-x86_64" => s"$cMakeToolsPath/bin/g++"
    }
    val cMakeCXXFlags = platform match {
      case "linux-x86_64" => "-std=c++11"
      case "darwin-x86_64" => "-std=c++11 -stdlib=libc++"
      case "windows-x86_64" => "-std=c++11"
    }
    val cMakeTargetSuffix = platform match {
      case "linux-x86_64" => ".so"
      case "darwin-x86_64" => ".dylib"
      case "windows-x86_64" => ".dll"
    }
    s"""cmake_minimum_required(VERSION 3.1.0)
       |
       |set(CMAKE_C_COMPILER_WORKS 1)
       |set(CMAKE_CXX_COMPILER_WORKS 1)
       |
       |# Define project and related variables
       |project(tensorflow_jni CXX)
       |set(PROJECT_VERSION_MAJOR 0)
       |set(PROJECT_VERSION_MINOR 0)
       |set(PROJECT_VERSION_PATCH 0)
       |
       |# Set up JNI
       |find_package(JNI REQUIRED)
       |if (JNI_FOUND)
       |    message (STATUS "JNI include directories: $${JNI_INCLUDE_DIRS}")
       |endif()
       |
       |# Set up the cross-compilation environment
       |set(CMAKE_SYSTEM_NAME $cMakeSystemName)
       |set(CMAKE_C_COMPILER $cMakeCCompiler)
       |set(CMAKE_CXX_COMPILER $cMakeCXXCompiler)
       |set(CMAKE_CXX_FLAGS "$${CMAKE_CXX_FLAGS} $cMakeCXXFlags")
       |${if (platform == "windows-x86_64") "set(CMAKE_POSITION_INDEPENDENT_CODE OFF)" else ""}
       |
       |# Include directories
       |include_directories(.)
       |include_directories(generated)
       |include_directories($${JNI_INCLUDE_DIRS})
       |
       |# Sources
       |file(GLOB_RECURSE LIB_SRC
       |  "*.c"
       |  "*.cc"
       |  "*.cpp"
       |)
       |
       |# Setup installation targets
       |set(LIB_NAME $${PROJECT_NAME})
       |add_library($${LIB_NAME} MODULE $${LIB_SRC})
       |target_link_libraries($${LIB_NAME} -ltensorflow)
       |set_target_properties($${LIB_NAME} PROPERTIES SUFFIX "$cMakeTargetSuffix")
       |install(TARGETS $${LIB_NAME} LIBRARY DESTINATION .)
     """.stripMargin
  }

  def generateCrossCompileScript(
      platform: String, tmpFolderName: String, dockerImageName: String, projectDir: String, targetDir: String,
      tensorFlowVersion: String): String = {
    val tfLibFilename = {
      if (platform == "windows-x86_64")
        s"libtensorflow-cpu-$platform.zip"
      else
        s"libtensorflow-cpu-$platform.tar.gz"
    }
    val tfLibUrl = tensorFlowVersion match {
      case "nightly" =>
        val urlPrefix = "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow"
        platform match {
          case "linux-x86_64" => s"$urlPrefix/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
          case "darwin-x86_64" => s"$urlPrefix/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
          case "windows-x86_64" => s"$urlPrefix-windows/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
        }
      case v =>
        if (platform == "windows-x86_64")
          s"https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$platform-$v.zip"
        else
          s"https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$platform-$v.tar.gz"
    }
    val tfLibExtract = platform match {
      case "linux-x86_64" => s"tar xvf /root/$tfLibFilename -C /usr"
      case "darwin-x86_64" => s"tar xvf /root/$tfLibFilename -C /usr/x86_64-linux-gnu/${supportedPlatforms(platform)}"
      case "windows-x86_64" => s"unzip /root/$tfLibFilename -d /usr/x86_64-linux-gnu/${supportedPlatforms(platform)}/lib"
    }
    s"""#!/bin/bash
       |
       |IMAGE=$dockerImageName
       |PLATFORM=$platform
       |CONTAINER=$$IMAGE_$$PLATFORM
       |
       |# Create the necessary Docker image
       |
       |if [[ "$$(docker images -q $$IMAGE 2> /dev/null)" == "" ]]; then
       |  docker build -t $$IMAGE $targetDir/docker/
       |fi
       |
       |# Delete existing Docker containers that are not running anymore
       |if [[ $$(docker ps -a -q | head -c1 | wc -c) -ne 0 ]]; then
       |  docker rm -v $$(docker ps -a -q)
       |fi
       |
       |# Create a new container and copy the repository code in it
       |docker run --name $$CONTAINER --net host -dit $$IMAGE /bin/bash
       |
       |if [[ ! -f $targetDir/lib/$tfLibFilename ]]; then
       |  wget -P $targetDir/lib/ $tfLibUrl
       |fi
       |
       |docker cp $targetDir/lib/$tfLibFilename $$CONTAINER:/root/$tfLibFilename
       |
       |# Extract the TensorFlow dynamic library in the container
       |docker exec $$CONTAINER bash -c "$tfLibExtract"
       |
       |# Compile and package the JNI bindings
       |docker cp $projectDir/src/main/native $$CONTAINER:/root/src
       |docker cp $targetDir/docker/CMakeLists.txt $$CONTAINER:/root/src/CMakeLists.txt
       |
       |docker exec $$CONTAINER bash -c '\\
       |  export LD_LIBRARY_PATH=/usr/x86_64-linux-gnu/${supportedPlatforms(platform)}/lib:"$$LD_LIBRARY_PATH" && \\
       |  export PATH=/usr/x86_64-linux-gnu/${supportedPlatforms(platform)}/bin:"$$PATH" && \\
       |  cd /root && mkdir bin && mkdir src/build && cd src/build && \\
       |  cmake -DCMAKE_INSTALL_PREFIX:PATH=/root/bin -DCMAKE_BUILD_TYPE=Release /root/src && \\
       |  make VERBOSE=1 && make install'
       |
       |# Copy the compiled library back to the host
       |docker cp $$CONTAINER:/root/bin $targetDir/
       |
       |# Kill and delete the Docker container
       |docker kill $$CONTAINER
       |docker rm $$CONTAINER
     """.stripMargin
  }
}
