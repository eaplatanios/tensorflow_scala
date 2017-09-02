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
import java.nio.file.{Files, Paths}

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

  object autoImport {
    val enableNativeCrossCompilation: SettingKey[Boolean] =
      settingKey[Boolean](
        "Determines if native cross-compilation is enabled. If not enabled, only pre-compiled libraries in " +
            "'unmanagedNativeDirectories' will be packaged.")

    val nativePlatforms: SettingKey[Set[Platform]] =
      settingKey[Set[Platform]]("###")

    val projectRoot: SettingKey[String] =
      settingKey[String]("###")

    val tensorFlowBinaryVersion: SettingKey[String] =
      settingKey[String]("###")

    val nativeCrossCompile: TaskKey[Map[Platform, (Set[File], Set[File])]] =
      taskKey[Map[Platform, (Set[File], Set[File])]]("###")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    enableNativeCrossCompilation := true,
    nativePlatforms in nativeCrossCompile := Set(LINUX_x86_64, DARWIN_x86_64, WINDOWS_x86_64),
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
        log.info(s"Cross-compiling '${moduleName.value}' for platform '$platform'.")
        log.info(s"Using ${baseDirectory.value} as the base directory.")
        val platformTargetDir = targetDir / platform.name

        IO.createDirectory(platformTargetDir)
        IO.createDirectory(platformTargetDir / "docker")
        IO.createDirectory(platformTargetDir / "lib")

        // Generate Dockerfile
        val dockerfilePath = platformTargetDir / "docker" / "Dockerfile"
        log.info(s"Generating Dockerfile in '$dockerfilePath'.")
        new PrintWriter(dockerfilePath) {write(dockerfile); close()}

        // Generate CMakeLists.txt
        val cMakeLists = platformTargetDir / "docker" / "CMakeLists.txt"
        log.info(s"Generating CMakeLists in '$cMakeLists'.")
        new PrintWriter(cMakeLists) {write(platform.cMakeLists); close()}

        // Download the native TensorFlow library
        log.info(s"Downloading the TensorFlow native library.")
        val tfVersion = (tensorFlowBinaryVersion in nativeCrossCompile).value
        var exitCode = platform.downloadTfLib(platformTargetDir.getPath, tfVersion) ! log

        if (exitCode == 0) {
          // Compile and generate binaries
          log.info(s"Generating binaries in '$platformTargetDir'.")
          val dockerImage = s"${moduleName.value}"
          exitCode = platform.build(
            dockerImage, (baseDirectory.value / "src" / "main" / "native").getPath, platformTargetDir.getPath) ! log
          log.info("Cleaning up after build.")
          platform.cleanUpAfterBuild(dockerImage) ! log
        }
        if (exitCode != 0)
          sys.error(s"An error occurred while cross-compiling for '$platform'. Exit code: $exitCode.")
        platform -> (
            (platformTargetDir / "lib" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet,
            (platformTargetDir / "bin" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet)
      }).toMap
    },
    resourceGenerators in Compile += Def.taskDyn[Seq[File]] {
      val enableCrossCompilation = enableNativeCrossCompilation.value
      if (enableCrossCompilation) Def.task {
        val libraries: Map[Platform, (Set[File], Set[File])] = nativeCrossCompile.value
        val jniLibraries: Seq[(File, String)] = libraries.flatMap { case (platform, (tfLibs, jniLibs)) =>
          jniLibs.map(l => l -> s"/native/${platform.name}/${l.name}")
        } toSeq
        val resources: Seq[File] = for ((file, path) <- jniLibraries) yield {
          // Native library as a managed resource file.
          val resource = (resourceManaged in Compile).value / path
          // Copy native library to a managed resource, so that it is always available on the classpath, even when not
          // packaged in a JAR file.
          IO.copyFile(file, resource)
          resource
        }
        resources
      } else Def.task {
        Seq.empty
      }
    }.taskValue,
    // Make the SBT clean task also cleans the generated cross-compilation files
    clean := clean.dependsOn(clean in nativeCrossCompile).value
  )

  override lazy val projectSettings: Seq[Setting[_]] = settings

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

  sealed trait Platform {
    val name       : String
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
    val cMakeListsAdditions: String = ""
    val cMakePath          : String = s"/usr/x86_64-linux-gnu/$crossTriple/bin"
    val cMakeLibPath       : String = s"/usr/x86_64-linux-gnu/$crossTriple/lib"

    def cMakeLists: String = {
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
         |$cMakeListsAdditions
         |set(CMAKE_SKIP_RPATH TRUE)
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
         |""".stripMargin
    }

    def downloadTfLib(targetDir: String, tfVersion: String): ProcessBuilder = {
      val path = s"$targetDir/lib/$tfLibFilename"
      if (Files.notExists(Paths.get(path)))
        url(tfLibUrl(tfVersion)) #> file(path)
      else
        Process(true)
    }

    def build(dockerImage: String, srcDir: String, targetDir: String): ProcessBuilder = {
      val dockerContainer: String = s"${dockerImage}_$name"
      val hostTfLibPath: String = s"$targetDir/lib/$tfLibFilename"
      val hostCMakeListsPath: String = s"$targetDir/docker/CMakeLists.txt"
      // Create the necessary Docker image
      Process("/bin/bash" :: "-c" ::
                  s"(docker images -q $dockerImage | grep -q . || " +
                      s"docker build -t $dockerImage $targetDir/docker/) || true" :: Nil) #&&
          // Delete existing Docker containers that are not running anymore
          Process("/bin/bash" :: "-c" ::
                      "(docker ps -a -q | grep -q . && " +
                          "docker rm -fv $(docker ps -a -q)) || true" :: Nil) #&&
          // Create a new container and copy the repository code in it
          Process("docker" :: "run" :: "--name" :: dockerContainer :: "-dit" :: dockerImage :: "/bin/bash" :: Nil) #&&
          Process("docker" :: "cp" :: hostTfLibPath :: s"$dockerContainer:/root/$tfLibFilename" :: Nil) #&&
          // Extract the TensorFlow dynamic library in the container
          Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" :: tfLibExtractCommand :: Nil) #&&
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
    }

    def cleanUpAfterBuild(dockerImage: String): ProcessBuilder = {
      // Kill and delete the Docker container
      val dockerContainer: String = s"${dockerImage}_$name"
      Process(
        "/bin/bash" :: "-c" ::
            s"(docker ps -a -q -f ${'"'}name=$dockerContainer${'"'} | grep -q . && " +
                s"docker kill $dockerContainer >/dev/null && " +
                s"docker rm -fv $dockerContainer) >/dev/null || true" :: Nil)
    }

    override def toString: String = name
  }

  object LINUX_x86_64 extends Platform {
    override val name       : String = "linux-x86_64"
    override val crossTriple: String = "x86_64-linux-gnu"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xvf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Linux"
    override val cMakeToolsPath   : String = "/usr"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXFlags    : String = "-std=c++11"
    override val cMakeTargetSuffix: String = ".so"
  }

  object DARWIN_x86_64 extends Platform {
    override val name       : String = "darwin-x86_64"
    override val crossTriple: String = "x86_64-apple-darwin14"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xvf /root/$tfLibFilename -C /usr/x86_64-linux-gnu/$crossTriple"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName  : String = "Darwin"
    override val cMakeToolsPath   : String = "/usr/osxcross"
    override val cMakeCCompiler   : String = s"$cMakeToolsPath/bin/$crossTriple-clang"
    override val cMakeCXXCompiler : String = s"$cMakeToolsPath/bin/$crossTriple-clang++"
    override val cMakeCXXFlags    : String = "-std=c++11 -stdlib=libc++"
    override val cMakeTargetSuffix: String = ".dylib"
  }

  object WINDOWS_x86_64 extends Platform {
    override val name       : String = "windows-x86_64"
    override val crossTriple: String = "x86_64-w64-mingw32"

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.zip"
    override val tfLibExtractCommand: String = s"unzip /root/$tfLibFilename -d /usr/x86_64-linux-gnu/$crossTriple/lib"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix-windows/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override val cMakeSystemName    : String = "Windows"
    override val cMakeToolsPath     : String = s"/usr/$crossTriple"
    override val cMakeCCompiler     : String = s"$cMakeToolsPath/bin/gcc"
    override val cMakeCXXCompiler   : String = s"$cMakeToolsPath/bin/g++"
    override val cMakeCXXFlags      : String = "-std=c++11"
    override val cMakeTargetSuffix  : String = ".dll"
    override val cMakeListsAdditions: String = "set(CMAKE_POSITION_INDEPENDENT_CODE OFF)"
  }
}
