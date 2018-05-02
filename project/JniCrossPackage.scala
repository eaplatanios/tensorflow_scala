/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import sbt._
import sbt.Keys._

import java.nio.file.{Files, Paths}

import scala.collection.JavaConverters._
import scala.sys.process.{Process, ProcessBuilder}

/**
  * @author Emmanouil Antonios Platanios
  */
object JniCrossPackage extends AutoPlugin {
  override def requires: Plugins = JniNative && plugins.JvmPlugin

  object autoImport {
    lazy val JniCross = config("cross")
        .describedAs("Native code cross-compiling configuration.")

    val dockerImagePrefix: SettingKey[String] =
      settingKey[String]("###")

    val nativePlatforms: SettingKey[Set[Platform]] =
      settingKey[Set[Platform]]("Set of native platforms for which to cross-compile.")

    val nativeArtifactName: SettingKey[String] =
      settingKey[String]("###")

    val nativeLibPath: TaskKey[Map[Platform, File]] =
      taskKey[Map[Platform, File]]("###")

    val nativeCrossCompile: TaskKey[Map[Platform, CrossCompilationOutput]] =
      taskKey[Map[Platform, CrossCompilationOutput]]("###")
  }

  case class CrossCompilationOutput(
      managedResources: Set[File],
      packagedArtifactsDir: File,
      packagedArtifacts: Set[File])

  import autoImport._
  import JniNative.autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    nativePlatforms := Set(LINUX_x86_64, LINUX_GPU_x86_64, DARWIN_x86_64),
    target := (target in Compile).value / "native",
    nativeLibPath := {
      val targetDir = (target in nativeCrossCompile).value
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        platform -> targetDir / platform.name
      }).toMap
    },
    // Make the SBT clean task also cleans the generated cross-compilation files
    clean := {
      (clean in Compile).value
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
    nativeCompile := {
      nativeCrossCompile.value
      Seq.empty
    },
    nativeCrossCompile := {
      val log = streams.value.log
      val baseDir = baseDirectory.value
      val targetDir = (target in nativeCrossCompile).value
      IO.createDirectory(targetDir)
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        log.info(s"Cross-compiling '${moduleName.value}' for platform '$platform'.")
        log.info(s"Using '$baseDir' as the base directory.")
        val platformTargetDir = targetDir / platform.name

        IO.createDirectory(platformTargetDir)
        IO.createDirectory(platformTargetDir / "code")
        IO.createDirectory(platformTargetDir / "docker")
        IO.createDirectory(platformTargetDir / "lib")

        // Generate Dockerfile
        platform.dockerfile.foreach(d => {
          val dockerfilePath = platformTargetDir / "docker" / "Dockerfile"
          log.info(s"Generating Dockerfile in '$dockerfilePath'.")
          IO.write(dockerfilePath, d)
        })

        // Compile and generate binaries
        log.info(s"Generating binaries in '$platformTargetDir'.")
        val dockerContainer = s"${moduleName.value}_${platform.name}"
        val exitCode = platform.build(
          dockerImage = s"${dockerImagePrefix.value}_${platform.name}",
          dockerContainer = dockerContainer,
          srcDir = (baseDirectory.value / "src" / "main" / "native").getPath,
          tgtDir = platformTargetDir.getPath,
          libPath = nativeLibPath.value(platform).getPath).map(_ ! log)
        log.info("Cleaning up after build.")
        platform.cleanUpAfterBuild(dockerContainer).foreach(_ ! log)
        if (exitCode.getOrElse(0) != 0)
          sys.error(s"An error occurred while cross-compiling for '$platform'. Exit code: $exitCode.")

        val managedResources = (platformTargetDir / "bin" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet
        val packagedArtifactsDir = platformTargetDir / "lib"
        val packagedArtifacts = (platformTargetDir / "lib" ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet
        platform -> CrossCompilationOutput(managedResources, packagedArtifactsDir, packagedArtifacts)
      }).toMap
    },
    resourceGenerators := Seq(Def.task {
      getManagedResources(
        nativeCrossCompile.value,
        (resourceManaged in Compile).value)
    }.taskValue))

  override lazy val projectSettings: Seq[Def.Setting[_]] = inConfig(JniCross)(settings)

  def getManagedResources(
      crossCompilationOutputs: Map[Platform, CrossCompilationOutput],
      resourceManaged: File
  ): Seq[File] = {
    val managedResources: Seq[(File, String)] = crossCompilationOutputs.flatMap {
      case (platform, CrossCompilationOutput(r, _, _)) =>
        r.map(l => l -> s"/native/${platform.name}/${l.name}")
    } toSeq
    val resources: Seq[File] = for ((file, path) <- managedResources) yield {
      // Native library as a managed resource file.
      val resource = resourceManaged / path
      // Copy native library to a managed resource, so that it is always available on the classpath, even when not
      // packaged in a JAR file.
      IO.copyFile(file, resource)
      resource
    }
    resources
  }

  def getPackagedArtifacts(
      platform: Platform,
      crossCompilationOutput: CrossCompilationOutput
  ): Option[File] = {
    val dir = crossCompilationOutput.packagedArtifactsDir.getPath
    val dirPath = Paths.get(dir)
    // TODO: [BUILD] !!! Make the following name more generic.
    val jarPath = dirPath.resolveSibling(s"tensorflow-native-${platform.name}.jar").toString
    val filePaths = crossCompilationOutput.packagedArtifacts.map(f => dirPath.relativize(Paths.get(f.getPath))).toList
    if (filePaths.nonEmpty) {
      Process("jar" :: "cf" :: jarPath :: Nil ++ filePaths.flatMap("-C" :: dir :: _.toString :: Nil)).!
      Some(new File(jarPath))
    } else {
      None
    }
  }

  sealed trait Platform {
    val name: String
    val tag : String
    val dockerfile: Option[String] = None

    val cMakePath   : String = "/usr/bin"
    val cMakeLibPath: String = "/usr/lib"

    def build(
        dockerImage: String,
        dockerContainer: String,
        srcDir: String,
        tgtDir: String,
        libPath: String
    ): Option[ProcessBuilder] = {
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
          Process("docker" :: "cp" :: s"$libPath/lib/." :: s"$dockerContainer:$cMakeLibPath" :: Nil) #&&
          // Compile and package the JNI bindings
          Process("docker" :: "cp" :: srcDir :: s"$dockerContainer:/root/src" :: Nil) #&&
          Process("docker" :: "exec" :: dockerContainer :: "bash" :: "-c" ::
              s"export LD_LIBRARY_PATH=$cMakeLibPath:${'"'}$$LD_LIBRARY_PATH${'"'} && " +
                  s"export PATH=$cMakePath:${'"'}$$PATH${'"'} && " +
                  "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 && " +
                  "cd /root && mkdir bin && mkdir src/build && cd src/build && " +
                  "cmake -DCMAKE_INSTALL_PREFIX:PATH=/root/bin -DCMAKE_BUILD_TYPE=Release /root/src &&" +
                  "make VERBOSE=1 && make install" :: Nil) #&&
          // Copy the compiled library back to the host
          Process("docker" :: "cp" :: s"$dockerContainer:/root/bin" :: tgtDir :: Nil)
      Some(process)
    }

    def cleanUpAfterBuild(dockerContainer: String): Option[ProcessBuilder] = {
      // Kill and delete the Docker container
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
    override val name: String = "linux-x86_64"
    override val tag : String = "linux-cpu-x86_64"

    // The parent Docker image is defined in the .circleci/images directory at the root of this repository.
    override val dockerfile: Option[String] = Some(
      """
        |FROM eaplatanios/tensorflow_scala:linux-cpu-x86_64-0.2.0
        |WORKDIR /root
      """.stripMargin)
  }

  object LINUX_GPU_x86_64 extends Platform {
    override val name: String = "linux-gpu-x86_64"
    override val tag : String = "linux-gpu-x86_64"

    // The parent Docker image is defined in the .circleci/images directory at the root of this repository.
    override val dockerfile: Option[String] = Some(
      """
        |FROM eaplatanios/tensorflow_scala:linux-gpu-x86_64-0.2.0
        |WORKDIR /root
      """.stripMargin)
  }

  object DARWIN_x86_64 extends Platform {
    override val name: String = "darwin-x86_64"
    override val tag : String = "darwin-cpu-x86_64"

    override def build(
        dockerImage: String,
        dockerContainer: String,
        srcDir: String,
        tgtDir: String,
        libPath: String
    ): Option[ProcessBuilder] = {
      val process = Process("bash" :: "-c" ::
          s"cp -rp $srcDir $tgtDir/code/native && " +
              s"rm -rf $tgtDir/code/native/build && " +
              s"mkdir $tgtDir/code/native/build && " +
              s"cd $tgtDir/code/native/build && " +
              s"cmake -DCMAKE_INSTALL_PREFIX:PATH=$tgtDir/bin -DCMAKE_BUILD_TYPE=Release $tgtDir/code/native && " +
              "make VERBOSE=1 && make install" :: Nil)
      Some(process)
    }

    override def cleanUpAfterBuild(dockerImage: String): Option[ProcessBuilder] = None
  }
}
