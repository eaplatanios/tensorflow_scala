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

import java.nio.file.{Files, Paths}

import scala.collection.JavaConverters._

import sbt._
import sbt.Keys._

import sys.process._

/**
  *
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowNativePackage extends AutoPlugin {
  override def requires: Plugins = JniNative && plugins.JvmPlugin

  object autoImport {
    lazy val CrossCompile = config("cross").extend(Test).describedAs("Native code cross-compiling configuration.")

    val nativePlatforms: SettingKey[Set[Platform]] =
      settingKey[Set[Platform]]("###")

    val projectRoot: SettingKey[String] =
      settingKey[String]("###")

    val tensorFlowBinaryVersion: SettingKey[String] =
      settingKey[String]("###")

    val compileTFLib: SettingKey[Boolean] =
      settingKey[Boolean](
        "If `true`, the native TensorFlow library will be compiled on this machine. If `false`, pre-compiled " +
            "binaries will be downloaded from the TensorFlow CI server.")

    val tfLibRepository: SettingKey[String] =
      settingKey[String](
        "Git repository from which to obtain the sources of the native TensorFlow library, if it is to be compiled.")

    val tfLibRepositoryBranch: SettingKey[String] =
      settingKey[String](
        "Git repository branch from which to obtain the sources of the native TensorFlow library, if it is to be " +
            "compiled.")

    val nativeCrossCompile: TaskKey[Map[Platform, ((File, Set[File]), Set[File])]] =
      taskKey[Map[Platform, ((File, Set[File]), Set[File])]]("###")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    nativePlatforms := Set(LINUX_x86_64, DARWIN_x86_64, WINDOWS_x86_64),
    tensorFlowBinaryVersion := "nightly",
    compileTFLib := false,
    tfLibRepository := "https://github.com/tensorflow/tensorflow.git",
    tfLibRepositoryBranch := "master",
    target := (target in Compile).value / "native",
    clean := {
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
      val log = streams.value.log
      val baseDir = baseDirectory.value
      val targetDir = (target in nativeCrossCompile).value
      val tfVersion = (tensorFlowBinaryVersion in nativeCrossCompile).value
      val compileTfLibValue = compileTFLib.value
      val tfLibRepositoryValue = tfLibRepository.value
      val tfLibRepositoryBranchValue = tfLibRepositoryBranch.value
      val moduleNameValue = moduleName.value
      IO.createDirectory(targetDir)
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        log.info(s"Cross-compiling '$moduleNameValue' for platform '$platform'.")
        log.info(s"Using '$baseDir' as the base directory.")
        val platformTargetDir = targetDir / platform.name

        IO.createDirectory(platformTargetDir)
        IO.createDirectory(platformTargetDir / "docker")
        IO.createDirectory(platformTargetDir / "downloads")
        IO.createDirectory(platformTargetDir / "downloads" / "lib")
        IO.createDirectory(platformTargetDir / "lib")

        if (compileTfLibValue) {
          IO.createDirectory(platformTargetDir / "code")
          TensorFlowNativeCrossCompiler.compile(
            (platformTargetDir / "code").toPath, platformTargetDir.getPath, tfLibRepositoryValue,
            tfLibRepositoryBranchValue, platform) ! log
        }

        // Generate Dockerfile
        val dockerfilePath = platformTargetDir / "docker" / "Dockerfile"
        log.info(s"Generating Dockerfile in '$dockerfilePath'.")
        IO.write(dockerfilePath, platform.dockerfile)

        // Generate CMakeLists.txt
        val cMakeLists = platformTargetDir / "docker" / "CMakeLists.txt"
        log.info(s"Generating CMakeLists in '$cMakeLists'.")
        IO.write(cMakeLists, platform.cMakeLists)

        // Download the native TensorFlow library
        log.info(s"Downloading the TensorFlow native library.")
        var exitCode = platform.downloadTfLib(platformTargetDir.getPath, tfVersion).map(_ ! log)

        if (exitCode.getOrElse(0) == 0) {
          // Compile and generate binaries
          log.info(s"Generating binaries in '$platformTargetDir'.")
          val dockerImage = moduleNameValue
          exitCode = platform.build(
            dockerImage, (baseDirectory.value / "src" / "main" / "native").getPath, platformTargetDir.getPath)
              .map(_ ! log)
          log.info("Cleaning up after build.")
          platform.cleanUpAfterBuild(dockerImage).foreach(_ ! log)
        }
        if (exitCode.getOrElse(0) != 0)
          sys.error(s"An error occurred while cross-compiling for '$platform'. Exit code: $exitCode.")
        platform -> (
            (platformTargetDir / "lib",
                (platformTargetDir / "lib" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet),
            (platformTargetDir / "bin" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet)
      }).toMap
    },
    compile := (compile in Compile).dependsOn(nativeCrossCompile).value,
    // Make the SBT clean task also cleans the generated cross-compilation files
    clean in Compile := (clean in Compile).dependsOn(clean in nativeCrossCompile).value
  )

  override lazy val projectSettings: Seq[Def.Setting[_]] =
    inConfig(CrossCompile)(settings) ++
        Seq(crossPaths := false) // We do not add the Scala version to the native JAR files

  // def nativeLibraries(libraries: Map[Platform, (Set[File], Set[File])], resourceManaged: File): Seq[File] = {
  //   val nativeLibraries: Seq[(File, String)] = libraries.flatMap { case (platform, (nativeLibs, _)) =>
  //     nativeLibs.map(l => l -> s"/native/${platform.name}/${l.name}")
  //   } toSeq
  //   val resources: Seq[File] = for ((file, path) <- nativeLibraries) yield {
  //     // Native library as a managed resource file.
  //     val resource = resourceManaged / path
  //     // Copy native library to a managed resource, so that it is always available on the classpath, even when not
  //     // packaged in a JAR file.
  //     IO.copyFile(file, resource)
  //     resource
  //   }
  //   resources
  // }

  def jniLibraries(libraries: Map[Platform, ((File, Set[File]), Set[File])], resourceManaged: File): Seq[File] = {
    val jniLibraries: Seq[(File, String)] = libraries.flatMap { case (platform, (_, jniLibs)) =>
      jniLibs.map(l => l -> s"/native/${platform.name}/${l.name}")
    } toSeq
    val resources: Seq[File] = for ((file, path) <- jniLibraries) yield {
      // Native library as a managed resource file.
      val resource = resourceManaged / path
      // Copy native library to a managed resource, so that it is always available on the classpath, even when not
      // packaged in a JAR file.
      IO.copyFile(file, resource)
      resource
    }
    resources
  }

  def nativeLibsToJar(
      platform: Platform, dir: File, files: Set[File], tfVersion: String, logger: ProcessLogger): File = {
    val dirPath = Paths.get(dir.getPath)
    val jarPath = dirPath.resolveSibling(s"tensorflow-native-${platform.name}.jar").toString
    val filePaths = files.map(f => dirPath.relativize(Paths.get(f.getPath))).toList
    Process("jar" :: "cf" :: jarPath :: "-C" :: dir.getPath :: Nil ++ filePaths.map(_.toString)) ! logger
    new File(jarPath)
  }

  sealed trait Platform {
    val name       : String
    val tag        : String
    val crossTriple: String

    val tfLibFilename      : String
    val tfLibExtractCommand: String

    val tfLibUrlPrefix       : String = "https://storage.googleapis.com/tensorflow/libtensorflow"
    val tfLibNightlyUrlPrefix: String = "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow"

    def tfLibUrl(version: String): String

    val cMakeSystemName  : String
    val cMakeToolsPath   : String
    val cMakeCCompiler   : String
    val cMakeCXXCompiler : String
    val cMakeTargetSuffix: String
    val cMakeCXXFlags    : String
    val cMakePath        : String
    val cMakeLibPath     : String

    val cMakeListsAdditions: String = ""

    val dockerfile: String = {
      s"""# Pull base image
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
         |""".stripMargin
    }

    def cMakeLists: String = {
      s"""cmake_minimum_required(VERSION 3.1.0)
         |
         |set(CMAKE_C_COMPILER_WORKS 1)
         |set(CMAKE_CXX_COMPILER_WORKS 1)
         |
         |# Define project and related variables
         |project(tensorflow CXX)
         |set(PROJECT_VERSION_MAJOR 0)
         |set(PROJECT_VERSION_MINOR 0)
         |set(PROJECT_VERSION_PATCH 0)
         |
         |# Set up JNI
         |find_package(JNI REQUIRED)
         |if (JNI_FOUND)
         |    message(STATUS "JNI include directories: $${JNI_INCLUDE_DIRS}")
         |endif()
         |
         |# Set up the cross-compilation environment
         |set(CMAKE_SYSTEM_NAME $cMakeSystemName)
         |set(CMAKE_C_COMPILER $cMakeCCompiler)
         |set(CMAKE_CXX_COMPILER $cMakeCXXCompiler)
         |set(CMAKE_CXX_FLAGS "$${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 $cMakeCXXFlags")
         |
         |# set(CMAKE_SKIP_RPATH TRUE)
         |
         |# Include directories
         |include_directories(.)
         |include_directories(./generated)
         |include_directories(./include)
         |include_directories(./ops)
         |include_directories($${JNI_INCLUDE_DIRS})
         |
         |# Find Native TensorFlow Library to link
         |find_library(LIB_TENSORFLOW tensorflow HINTS ENV LD_LIBRARY_PATH)
         |if(NOT LIB_TENSORFLOW)
         |  message(FATAL_ERROR "Library `tensorflow` not found.")
         |endif()
         |
         |find_library(LIB_TENSORFLOW_FRAMEWORK tensorflow_framework HINTS ENV LD_LIBRARY_PATH)
         |if(NOT LIB_TENSORFLOW_FRAMEWORK)
         |  message(FATAL_ERROR "Library `tensorflow_framework` not found.")
         |endif()
         |
         |# Collect sources for the JNI and the op libraries
         |
         |file(GLOB JNI_LIB_SRC
         |  "*.cc"
         |  "generated/*.cc"
         |  "include/tensorflow/c/*.cc"
         |  "include/tensorflow/core/distributed_runtime/*.cc"
         |)
         |
         |file(GLOB OP_LIB_SRC
         |  "ops/*.cc"
         |)
         |
         |# Setup installation targets
         |set(JNI_LIB_NAME "$${PROJECT_NAME}_jni")
         |add_library($${JNI_LIB_NAME} MODULE $${JNI_LIB_SRC})
         |target_link_libraries($${JNI_LIB_NAME} $${LIB_TENSORFLOW} $${LIB_TENSORFLOW_FRAMEWORK})
         |install(TARGETS $${JNI_LIB_NAME} LIBRARY DESTINATION .)
         |
         |set(OP_LIB_NAME "$${PROJECT_NAME}_ops")
         |add_library($${OP_LIB_NAME} MODULE $${OP_LIB_SRC})
         |target_link_libraries($${OP_LIB_NAME} $${LIB_TENSORFLOW} $${LIB_TENSORFLOW_FRAMEWORK})
         |install(TARGETS $${OP_LIB_NAME} LIBRARY DESTINATION .)
         |
         |$cMakeListsAdditions
         |""".stripMargin
    }

    def downloadTfLib(targetDir: String, tfVersion: String): Option[ProcessBuilder] = {
      val path = s"$targetDir/downloads/lib/$tfLibFilename"
      val downloadProcess = {
        if (Files.notExists(Paths.get(path)))
          url(tfLibUrl(tfVersion)) #> file(path)
        else
          Process(true)
      }
      val extractProcess = {
        if (tfLibFilename.endsWith(".tar.gz"))
          Process("tar" :: "xf" :: path :: Nil, new File(s"$targetDir/"))
        else
          Process("unzip" :: "-qq" :: "-u" :: path :: Nil, new File(s"$targetDir/"))
      }
      Some(downloadProcess #&& extractProcess)
    }

    def build(dockerImage: String, srcDir: String, targetDir: String): Option[ProcessBuilder] = {
      val dockerContainer: String = s"${dockerImage}_$name"
      val hostTfLibPath: String = s"$targetDir/downloads/lib/$tfLibFilename"
      val hostCMakeListsPath: String = s"$targetDir/docker/CMakeLists.txt"
      // Create the necessary Docker image
      val process = Process(
        "/bin/bash" :: "-c" ::
            s"(docker images -q $dockerImage | grep -q . || " +
                s"docker build -t $dockerImage $targetDir/docker/) || true" :: Nil) #&&
          // Delete existing Docker containers that are not running anymore
          Process("/bin/bash" :: "-c" ::
                      "(docker ps -a -q | grep -q . && " +
                          "docker rm -fv $(docker ps -a -q)) || true" :: Nil) #&&
          // Create a new container and copy the repository code in it
          Process("docker" :: "run" :: "--name" :: dockerContainer :: "-dit" :: dockerImage :: "/bin/bash" :: Nil) #&&
          Process("docker" :: "cp" :: hostTfLibPath :: s"$dockerContainer:/root/$tfLibFilename" :: Nil) #&&
          Process("docker" :: "cp" :: s"$targetDir/lib/." :: s"$dockerContainer:$cMakeLibPath" :: Nil) #&&
          // Extract the TensorFlow dynamic library in the container
          // Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" :: tfLibExtractCommand :: Nil) #&&
          // Compile and package the JNI bindings
          Process("docker" :: "cp" :: srcDir :: s"$dockerContainer:/root/src" :: Nil) #&&
          Process("docker" :: "cp" :: hostCMakeListsPath :: s"$dockerContainer:/root/src/CMakeLists.txt" :: Nil) #&&
          Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" ::
                      s"export LD_LIBRARY_PATH=$cMakeLibPath:${'"'}$$LD_LIBRARY_PATH${'"'} && " +
                          s"export PATH=$cMakePath:${'"'}$$PATH${'"'} && " +
                          "cd /root && mkdir bin && mkdir src/build && cd src/build && " +
                          "cmake -DCMAKE_INSTALL_PREFIX:PATH=/root/bin -DCMAKE_BUILD_TYPE=Release /root/src &&" +
                          "make VERBOSE=1 && make install" :: Nil) #&&
          // Copy the compiled library back to the host
          Process("docker" :: "cp" :: s"$dockerContainer:/root/bin" :: targetDir :: Nil)
      Some(process)
    }

    def cleanUpAfterBuild(dockerImage: String): Option[ProcessBuilder] = {
      // Kill and delete the Docker container
      val dockerContainer: String = s"${dockerImage}_$name"
      val process = Process(
        "/bin/bash" :: "-c" ::
            s"(docker ps -a -q -f ${'"'}name=$dockerContainer${'"'} | grep -q . && " +
                s"docker kill $dockerContainer >/dev/null && " +
                s"docker rm -fv $dockerContainer) >/dev/null || true" :: Nil)
      Some(process)
    }

    override def toString: String = name
  }

  object LINUX_x86_64 extends Platform {
    override val name       : String = "linux-x86_64"
    override val tag        : String = "linux-cpu-x86_64"
    override val crossTriple: String = "x86_64-linux-gnu"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Linux"
    override val cMakeToolsPath   : String = "/usr"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXFlags    : String = "-std=c++11"
    override val cMakeTargetSuffix: String = "so"
    override val cMakePath        : String = "/usr/bin"
    override val cMakeLibPath     : String = "/usr/lib"
  }

  object LINUX_GPU_x86_64 extends Platform {
    override val name       : String = "linux-gpu-x86_64"
    override val tag        : String = "linux-gpu-x86_64"
    override val crossTriple: String = "x86_64-linux-gnu"

    override val tfLibFilename      : String = s"libtensorflow-gpu-${LINUX_x86_64.name}.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=gpu-linux/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-gpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Linux"
    override val cMakeToolsPath   : String = "/usr"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXFlags    : String = "-std=c++11"
    override val cMakeTargetSuffix: String = "so"
    override val cMakePath        : String = "/usr/bin"
    override val cMakeLibPath     : String = "/usr/lib"

    override def build(dockerImage: String, srcDir: String, targetDir: String): Option[ProcessBuilder] = None
    override def cleanUpAfterBuild(dockerImage: String): Option[ProcessBuilder] = None
  }

  object DARWIN_x86_64 extends Platform {
    override val name       : String = "darwin-x86_64"
    override val tag        : String = "darwin-cpu-x86_64"
    override val crossTriple: String = "x86_64-apple-darwin14"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr/x86_64-linux-gnu/$crossTriple"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Darwin"
    override val cMakeToolsPath   : String = "/usr/osxcross"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/$crossTriple-clang"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/$crossTriple-clang++"
    override val cMakeCXXFlags    : String = "-std=c++11 -stdlib=libc++"
    override val cMakeTargetSuffix: String = "dylib"
    override val cMakePath        : String = s"/usr/x86_64-linux-gnu/$crossTriple/bin"
    override val cMakeLibPath     : String = s"/usr/x86_64-linux-gnu/$crossTriple/lib"
  }

  object WINDOWS_x86_64 extends Platform {
    override val name       : String = "windows-x86_64"
    override val tag        : String = "windows-cpu-x86_64"
    override val crossTriple: String = "x86_64-w64-mingw32"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.zip"
    override val tfLibExtractCommand: String = s"unzip /root/$tfLibFilename -d /usr/x86_64-linux-gnu/$crossTriple/lib"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix-windows/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Windows"
    override val cMakeToolsPath   : String = s"/usr/$crossTriple"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/g++"
    override val cMakeCXXFlags    : String = "-std=c++11"
    override val cMakeTargetSuffix: String = "dll"
    override val cMakePath        : String = s"/usr/x86_64-linux-gnu/$crossTriple/bin"
    override val cMakeLibPath     : String = s"/usr/x86_64-linux-gnu/$crossTriple/lib"

    override val cMakeListsAdditions: String = "set(CMAKE_POSITION_INDEPENDENT_CODE OFF)"
  }
}
