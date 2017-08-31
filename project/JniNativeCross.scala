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
import java.util.Comparator

import scala.collection.JavaConverters._

import sbt._
import sbt.Keys._
//import sys.process._

/**
  *
  * @author Emmanouil Antonios Platanios
  */
object JniNativeCross extends AutoPlugin {
  override def requires: Plugins = plugins.JvmPlugin

  val supportedPlatforms = Map(
    "linux-x86_64" -> "x86_64-linux-gnu",
    "darwin-x86_64" -> "x86_64-apple-darwin",
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
    nativePlatforms in nativeCrossCompile := Set("linux-x86_64", "darwin-x86_64", "windows-x86_64"),
    tensorFlowBinaryVersion in nativeCrossCompile := "nightly",
    target in nativeCrossCompile := target.value / "native",
    clean in nativeCrossCompile := {
      streams.value.log.info("Cleaning generated cross compilation files.")
      val path = (target in nativeCrossCompile).value.toPath
      Files.walk(path)
          .sorted(Comparator.reverseOrder())
          .iterator()
          .asScala
          .map(_.toFile)
          .foreach(_.delete)
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
        val dockerFile = platformTargetDir / "docker" / "Dockerfile"
        log.info(s"Generating Dockerfile in '$dockerFile'.")
        new PrintWriter(dockerFile) {write(dockerfile(platform, "tmp", scalaVersion.value, sbtVersion.value)); close()}
        val scriptFile = platformTargetDir / "docker" / "script.sh"
        log.info(s"Generating script in '$scriptFile'.")
        new PrintWriter(scriptFile) {
          write(crossCompileScript(
            platform,
            "tmp",
            s"${name.value}_$platform",
            baseDirectory.value.getPath,
            platformTargetDir.getPath,
            (tensorFlowBinaryVersion in nativeCrossCompile).value
          ))
          close()
        }
        log.info(s"Generating binaries in '$platformTargetDir'.")
        val exitCode = Process(s"bash $scriptFile") ! log
        if (exitCode != 0) sys.error(s"An error occurred while cross-compiling for '$platform'. Exit code: $exitCode.")
        platform -> (platformTargetDir ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toSet
      }).toMap
    },
    // Make the SBT clean task also cleans the generated cross-compilation files
    clean := clean.dependsOn(clean in nativeCrossCompile).value
  )

  override lazy val projectSettings: Seq[Setting[_]] = settings

  def dockerfile(platform: String, tmpFolderName: String, scalaVersion: String, sbtVersion: String): String = {
    s"""
       |# Pull base image
       |FROM multiarch/crossbuild
       |ENV CROSS_TRIPLE=${supportedPlatforms(platform)}
       |
       |# Install CMake and Java
       |RUN echo "deb http://httpredir.debian.org/debian/ jessie-backports main" > \\
       |  /etc/apt/sources.list.d/jessie-backports.list
       |RUN apt-get update
       |RUN apt-get -t jessie-backports -y --no-install-recommends install cmake openjdk-8-jdk
       |RUN /usr/sbin/update-java-alternatives -s java-1.8.0-openjdk-amd64
       |ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
       |
       |# Install Scala and SBT
       |
       |ENV SCALA_VERSION $scalaVersion
       |ENV SBT_VERSION $sbtVersion
       |
       |# Scala expects this file
       |RUN touch $$JAVA_HOME/release
       |
       |# Install Scala
       |## Piping curl directly in tar
       |RUN \\
       |  curl -fsL https://downloads.typesafe.com/scala/$$SCALA_VERSION/scala-$$SCALA_VERSION.tgz | \\
       |  tar xfz - -C /root/ && \\
       |  echo >> /root/.bashrc && \\
       |  echo 'export PATH=~/scala-$$SCALA_VERSION/bin:$$PATH' >> /root/.bashrc
       |
       |# Install sbt
       |RUN \\
       |  curl -L -o sbt-$$SBT_VERSION.deb https://dl.bintray.com/sbt/debian/sbt-$$SBT_VERSION.deb && \\
       |  dpkg -i sbt-$$SBT_VERSION.deb && \\
       |  rm sbt-$$SBT_VERSION.deb && \\
       |  apt-get update && \\
       |  apt-get install sbt && \\
       |  sbt sbtVersion
       |
       |# Create the directory that will contain the JNI binding sources
       |RUN mkdir /home/$tmpFolderName
       |
       |# Define working directory
       |WORKDIR /home/$tmpFolderName
       |
     """.stripMargin
  }

  def crossCompileScript(
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
    s"""
       |#!/bin/bash
       |
       |IMAGE=$dockerImageName
       |PLATFORM=$platform
       |CONTAINER=${dockerImageName}_${platform}_tmp
       |
       |# Create the necessary Docker image
       |
       |if [[ "$$(docker images -q $$IMAGE:$$PLATFORM 2> /dev/null)" == "" ]]; then
       |  docker build -t $$IMAGE:$$PLATFORM $targetDir/docker/
       |fi
       |
       |# Delete existing Docker containers that are not running anymore
       |if [[ $$(docker ps -a -q | head -c1 | wc -c) -ne 0 ]]; then
       |  docker rm -v $$(docker ps -a -q)
       |fi
       |
       |# Create a new container and copy the repository code in it
       |docker run --name $$CONTAINER --net host -dit $$IMAGE:$$PLATFORM /bin/bash
       |
       |if [[ ! -f $targetDir/lib/$tfLibFilename ]]; then
       |  wget -P $targetDir/lib/ $tfLibUrl
       |fi
       |
       |docker cp $targetDir/lib/$tfLibFilename $$CONTAINER:/home/$tfLibFilename
       |
       |# Install the TensorFlow dynamic library
       |if [[ "$$PLATFORM" == "windows-x86_64" ]]; then
       |  docker exec $$CONTAINER bash -c "unzip /home/$tfLibFilename -d /usr/lib"
       |else
       |  docker exec $$CONTAINER bash -c "tar xvf /home/$tfLibFilename -C /usr"
       |fi
       |
       |# Compile and package the JNI bindings
       |docker cp $projectDir/.. $$CONTAINER:/home/$tmpFolderName
       |docker exec $$CONTAINER bash -c "cd /home/$tmpFolderName && sbt jni/compile"
       |
       |# Copy the compiled library back to the host
       |docker cp $$CONTAINER:/home/$tmpFolderName/jni/target/native/$platform/bin $targetDir/
       |
       |# Kill and delete the Docker container
       |docker kill $$CONTAINER
       |docker rm $$CONTAINER
       |
     """.stripMargin
  }
}
