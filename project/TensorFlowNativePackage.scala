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
    nativePlatforms := Set(LINUX_x86_64, DARWIN_x86_64),
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
            .foreach(Files.deleteIfExists(_))
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
        IO.createDirectory(platformTargetDir / "code")
        IO.createDirectory(platformTargetDir / "docker")
        IO.createDirectory(platformTargetDir / "downloads")
        IO.createDirectory(platformTargetDir / "downloads" / "lib")
        IO.createDirectory(platformTargetDir / "lib")

        if (compileTfLibValue) {
          TensorFlowNativeCrossCompiler.compile(
            (platformTargetDir / "code").toPath, platformTargetDir.getPath, tfLibRepositoryValue,
            tfLibRepositoryBranchValue, platform) ! log
        }

        // Generate Dockerfile
        val dockerfilePath = platformTargetDir / "docker" / "Dockerfile"
        log.info(s"Generating Dockerfile in '$dockerfilePath'.")
        IO.write(dockerfilePath, platform.dockerfile)

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

  override lazy val projectSettings: Seq[Def.Setting[_]] = inConfig(CrossCompile)(settings)

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
    Process("jar" :: "cf" :: jarPath :: Nil ++ filePaths.flatMap("-C" :: dir.getPath :: _.toString :: Nil)) ! logger
    new File(jarPath)
  }

  sealed trait Platform {
    val name      : String
    val tag       : String
    val dockerfile: String

    val tfLibFilename      : String
    val tfLibExtractCommand: String

    val tfLibUrlPrefix       : String = "https://storage.googleapis.com/tensorflow/libtensorflow"
    val tfLibNightlyUrlPrefix: String = "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow"

    def tfLibUrl(version: String): String

    val cMakePath   : String = "/usr/bin"
    val cMakeLibPath: String = "/usr/lib"

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

    def build(dockerImage: String, srcDir: String, tgtDir: String): Option[ProcessBuilder] = {
      val dockerContainer: String = s"${dockerImage}_$name"
      val hostTfLibPath: String = s"$tgtDir/downloads/lib/$tfLibFilename"
      // Create the necessary Docker image
      val process = Process(
        "/bin/bash" :: "-c" ::
            s"(docker images -q $dockerImage | grep -q . || " +
                s"docker build -t $dockerImage $tgtDir/docker/) || true" :: Nil) #&&
          // Delete existing Docker containers that are not running anymore
          Process("/bin/bash" :: "-c" ::
              "(docker ps -a -q | grep -q . && " +
                  "docker rm -fv $(docker ps -a -q)) || true" :: Nil) #&&
          // Create a new container and copy the repository code in it
          Process("docker" :: "run" :: "--name" :: dockerContainer :: "-dit" :: dockerImage :: "/bin/bash" :: Nil) #&&
          Process("docker" :: "cp" :: hostTfLibPath :: s"$dockerContainer:/root/$tfLibFilename" :: Nil) #&&
          Process("docker" :: "cp" :: s"$tgtDir/lib/." :: s"$dockerContainer:$cMakeLibPath" :: Nil) #&&
          // Extract the TensorFlow dynamic library in the container
          // Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" :: tfLibExtractCommand :: Nil) #&&
          // Compile and package the JNI bindings
          Process("docker" :: "cp" :: srcDir :: s"$dockerContainer:/root/src" :: Nil) #&&
          Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" ::
              s"export LD_LIBRARY_PATH=$cMakeLibPath:${'"'}$$LD_LIBRARY_PATH${'"'} && " +
                  s"export PATH=$cMakePath:${'"'}$$PATH${'"'} && " +
                  "cd /root && mkdir bin && mkdir src/build && cd src/build && " +
                  "cmake -DCMAKE_INSTALL_PREFIX:PATH=/root/bin -DCMAKE_BUILD_TYPE=Release /root/src &&" +
                  "make VERBOSE=1 && make install" :: Nil) #&&
          // Copy the compiled library back to the host
          Process("docker" :: "cp" :: s"$dockerContainer:/root/bin" :: tgtDir :: Nil)
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
    override val name      : String = "linux-x86_64"
    override val tag       : String = "linux-cpu-x86_64"
    override val dockerfile: String =
      """
        |FROM eaplatanios/tensorflow_scala:linux-cpu-x86_64-0.1.1
        |WORKDIR /root
      """.stripMargin

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }
  }

  object LINUX_GPU_x86_64 extends Platform {
    override val name      : String = "linux-gpu-x86_64"
    override val tag       : String = "linux-gpu-x86_64"
    override val dockerfile: String =
      """
        |FROM eaplatanios/tensorflow_scala:linux-gpu-x86_64-0.1.1
        |WORKDIR /root
      """.stripMargin

    override val tfLibFilename      : String = s"libtensorflow-gpu-${LINUX_x86_64.name}.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=gpu-linux/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-gpu-$name-$version.tar.gz"
    }
  }

  object DARWIN_x86_64 extends Platform {
    override val name      : String = "darwin-x86_64"
    override val tag       : String = "darwin-cpu-x86_64"
    override val dockerfile: String =
      """
        |FROM eaplatanios/tensorflow_scala:darwin-cpu-x86_64-0.1.1
        |WORKDIR /root
      """.stripMargin

    override val tfLibFilename      : String = s"libtensorflow-cpu-$name.tar.gz"
    override val tfLibExtractCommand: String = s"tar xf /root/$tfLibFilename -C /usr"

    override def tfLibUrl(version: String): String = version match {
      case "nightly" => s"$tfLibNightlyUrlPrefix/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/$tfLibFilename"
      case _ => s"$tfLibUrlPrefix/libtensorflow-cpu-$name-$version.tar.gz"
    }

    override def build(dockerImage: String, srcDir: String, tgtDir: String): Option[ProcessBuilder] = {
      val process = Process("cp" :: "-rp" :: srcDir :: s"$tgtDir/code/jni" :: Nil) #&&
          Process(s"cd $tgtDir/code") #&&
          Process("rm" :: "-rf" :: "jni/temp_build" :: Nil) #&&
          Process("mkdir" :: "jni/temp_build" :: Nil) #&&
          Process("cd" :: "jni/temp_build" :: Nil) #&&
          Process("cmake" :: s"-DCMAKE_INSTALL_PREFIX:PATH=$tgtDir/bin" :: "-DCMAKE_BUILD_TYPE=Release" :: s"$tgtDir/code/jni" :: Nil) #&&
          Process("make" :: "VERBOSE=1" :: Nil) #&&
          Process("make" :: "install" :: Nil)
      Some(process)
    }

    override def cleanUpAfterBuild(dockerImage: String): Option[ProcessBuilder] = None
  }
}
