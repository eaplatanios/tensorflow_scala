/* Copyright 2017-20, Emmanouil Antonios Platanios. All Rights Reserved.
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
  }

  import autoImport._
  import JniCrossPackage.autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    tfBinaryVersion := "2.3.0",
    nativeArtifactName := "tensorflow",
    nativeLibPath := {
      val log       = streams.value.log
      val targetDir = (target in nativeCrossCompile).value
      val tfVersion = (tfBinaryVersion in nativeCrossCompile).value
      IO.createDirectory(targetDir)
      (nativePlatforms in nativeCrossCompile).value.map(platform => {
        val platformTargetDir = targetDir / platform.name
        IO.createDirectory(platformTargetDir / "downloads")

        // Download the TensorFlow native library.
        log.info(s"Downloading the TensorFlow native library for platform '${platform.name}'.")
        val exitCode = downloadAndExtractLibrary(platform, platformTargetDir.getPath, tfVersion).map(_ ! log)

        if (exitCode.getOrElse(0) != 0) {
          sys.error(
            s"An error occurred while preparing the native TensorFlow libraries for '$platform'. Exit code: $exitCode.")
        }

        platform -> platformTargetDir
      }).toMap
    }
  )

  override lazy val projectSettings: Seq[Def.Setting[_]] = inConfig(JniCross)(settings)

  val tfLibUrlPrefix: String = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow"

  def tfLibFilename(platform: Platform): String = platform match {
    case LINUX_x86_64 | LINUX_GPU_x86_64 | DARWIN_x86_64 => "libtensorflow.tar.gz"
    case WINDOWS_x86_64 | WINDOWS_GPU_x86_64 => "libtensorflow.zip"
  }

  def tfLibExtractCommand(platform: Platform): String = platform match {
    case LINUX_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
    case LINUX_GPU_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
    case WINDOWS_x86_64 => ???
    case WINDOWS_GPU_x86_64 => ???
    case DARWIN_x86_64 => s"tar xf /root/${tfLibFilename(platform)} -C /usr"
  }

  def tfLibUrl(platform: Platform, version: String): String = (platform, version) match {
    case (LINUX_x86_64, v) => s"$tfLibUrlPrefix-cpu-linux-x86_64-$v.tar.gz"
    case (LINUX_GPU_x86_64, v) => s"$tfLibUrlPrefix-gpu-linux-x86_64-$v.tar.gz"
    case (WINDOWS_x86_64, v) => s"$tfLibUrlPrefix-cpu-windows-x86_64-$v.zip"
    case (WINDOWS_GPU_x86_64, v) => s"$tfLibUrlPrefix-gpu-windows-x86_64-$v.zip"
    case (DARWIN_x86_64, v) => s"$tfLibUrlPrefix-cpu-darwin-x86_64-$v.tar.gz"
  }

  def downloadAndExtractLibrary(platform: Platform, targetDir: String, tfVersion: String): Option[ProcessBuilder] = {
    // TODO: Setup cross-compilation environments (maybe using CircleCI).
    None
//    val path = s"$targetDir/downloads/${tfLibFilename(platform)}"
//    platform match {
//      case WINDOWS_x86_64 | WINDOWS_GPU_x86_64 =>
//        if (Files.notExists(Paths.get(targetDir).resolve("lib"))) {
//          throw new IllegalStateException("The Windows TensorFlow library must have already been downloaded manually.")
//        }
//        None
//      case _ =>
//        val downloadProcess = if (Files.notExists(Paths.get(path))) {
//          url(tfLibUrl(platform, tfVersion)) #> file(path)
//        } else {
//          Process(true)
//        }
//        val extractProcess  = if (tfLibFilename(platform).endsWith(".tar.gz")) {
//          Process("tar" :: "xf" :: path :: Nil, new File(s"$targetDir/"))
//        } else {
//          Process("unzip" :: "-qq" :: "-u" :: path :: Nil, new File(s"$targetDir/"))
//        }
//        Some(downloadProcess #&& extractProcess)
//    }
  }
}
