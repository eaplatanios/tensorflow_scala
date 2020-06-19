/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import JniCrossPackage._

import sbt._
import sbt.Keys._
import sys.process._

import java.nio.file.{Files, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowNativePackage extends AutoPlugin {
  override def requires: Plugins = JniCrossPackage && plugins.JvmPlugin

  object autoImport {
    val tfBinaryVersion: SettingKey[String] = settingKey[String](
      "Version of the TensorFlow pre-compiled binaries to (optionally) download.")

    val tfLibCompile: SettingKey[Boolean] = settingKey[Boolean](
      "If `true`, the native TensorFlow library will be compiled on this machine. If `false`, pre-compiled " +
          "binaries will be downloaded from the TensorFlow CI server.")

    val tfLibRepository: SettingKey[String] = settingKey[String](
      "Git repository from which to obtain the sources of the TensorFlow library, if it is to be compiled.")

    val tfLibRepositoryBranch: SettingKey[String] = settingKey[String](
      "Git repository branch from which to obtain the sources of the TensorFlow library, if it is to be compiled.")
  }

  import autoImport._
  import JniCrossPackage.autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    tfBinaryVersion := "nightly",
    tfLibCompile := false,
    tfLibRepository := "https://github.com/tensorflow/tensorflow.git",
    tfLibRepositoryBranch := "master",
    nativeArtifactName := "tensorflow",
    nativeLibPath := {
      val log = streams.value.log
      val targetDir = (target in nativeCrossCompile).value
      val tfVersion = (tfBinaryVersion in nativeCrossCompile).value
      val compileTfLibValue = tfLibCompile.value
      val tfLibRepositoryValue = tfLibRepository.value
      val tfLibRepositoryBranchValue = tfLibRepositoryBranch.value
      IO.createDirectory(targetDir)
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        val platformTargetDir = targetDir / platform.name
        IO.createDirectory(platformTargetDir / "downloads")
        IO.createDirectory(platformTargetDir / "downloads" / "lib")

        if (compileTfLibValue) {
          val workingDir = (platformTargetDir / "code").toPath
          workingDir.resolve("tensorflow").toFile.mkdirs()
          TensorFlowNativeCrossCompiler.compile(
            workingDir, platformTargetDir.getPath, tfLibRepositoryValue,
            tfLibRepositoryBranchValue, platform) ! log
        }

        // Download the native TensorFlow library.
        log.info(s"Downloading the TensorFlow native library for platform '${platform.name}'.")
        val exitCode = downloadTfLib(platform, platformTargetDir.getPath, tfVersion).map(_ ! log)

        if (exitCode.getOrElse(0) != 0) {
          sys.error(
            s"An error occurred while preparing the native TensorFlow libraries for '$platform'. Exit code: $exitCode.")
        }

        platform -> platformTargetDir
      }).toMap
    }
  )

  override lazy val projectSettings: Seq[Def.Setting[_]] = inConfig(JniCross)(settings)

  // val tfLibUrlPrefix       : String = "https://storage.googleapis.com/tensorflow/libtensorflow"
  val tfLibUrlPrefix       : String = "https://www.dropbox.com/s"
  val tfLibNightlyUrlPrefix: String = "https://storage.googleapis.com/tensorflow-nightly/github/tensorflow/lib_package"

  def tfLibFilename(platform: Platform): String = platform match {
    case LINUX_x86_64 => s"libtensorflow-cpu-${platform.name}.tar.gz"
    case LINUX_GPU_x86_64 => s"libtensorflow-gpu-${LINUX_x86_64.name}.tar.gz"
    case DARWIN_x86_64 => s"libtensorflow-cpu-${platform.name}.tar.gz"
  }

  def tfLibExtractCommand(platform: Platform): String = platform match {
    case LINUX_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
    case LINUX_GPU_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
    case DARWIN_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
  }

  def tfLibUrl(platform: Platform, version: String): String = (platform, version) match {
    // case (LINUX_x86_64, "nightly") => s"$tfLibNightlyUrlPrefix/${tfLibFilename(platform)}"
    // case (LINUX_GPU_x86_64, "nightly") => s"$tfLibNightlyUrlPrefix/${tfLibFilename(platform)}"
    // case (DARWIN_x86_64, "nightly") => s"$tfLibNightlyUrlPrefix/${tfLibFilename(platform)}"
    // case (LINUX_x86_64, v) => s"$tfLibUrlPrefix/libtensorflow-cpu-${platform.name}-$v.tar.gz"
    // case (LINUX_GPU_x86_64, v) => s"$tfLibUrlPrefix/libtensorflow-gpu-${LINUX_x86_64.name}-$v.tar.gz"
    // case (DARWIN_x86_64, v) => s"$tfLibUrlPrefix/libtensorflow-cpu-${platform.name}-$v.tar.gz"
    case (LINUX_x86_64, v) => s"$tfLibUrlPrefix/uah9cnka3c83ir6/libtensorflow-$v-cpu-${platform.name}.tar.gz?dl=1"
    case (LINUX_GPU_x86_64, v) => s"$tfLibUrlPrefix/abfd9o9xibtxtle/libtensorflow-$v-gpu-${LINUX_x86_64.name}.tar.gz?dl=1"
    case (DARWIN_x86_64, v) => s"$tfLibUrlPrefix/h9rj6noaz1ai0uv/libtensorflow-$v-cpu-${platform.name}.tar.gz?dl=1"
  }

  def downloadTfLib(
      platform: Platform,
      targetDir: String,
      tfVersion: String
  ): Option[ProcessBuilder] = {
    val path = s"$targetDir/downloads/lib/${tfLibFilename(platform)}"
    val downloadProcess = {
      if (Files.notExists(Paths.get(path)))
        url(tfLibUrl(platform, tfVersion)) #> file(path)
      else
        Process(true)
    }
    val extractProcess = {
      if (tfLibFilename(platform).endsWith(".tar.gz"))
        Process("tar" :: "xf" :: path :: Nil, new File(s"$targetDir/"))
      else
        Process("unzip" :: "-qq" :: "-u" :: path :: Nil, new File(s"$targetDir/"))
    }
    Some(downloadProcess #&& extractProcess)
  }
}
