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

import BuildTool._

import sbt._
import sbt.Keys._

/** Adds functionality for using a native build tool through SBT.
  *
  * Borrowed from [the sbt-jni plugin](https://github.com/jodersky/sbt-jni), and modified.
  *
  * @author Emmanouil Antonios Platanios
  */
object JniNative extends AutoPlugin {
  override def requires: Plugins = plugins.JvmPlugin

  private[this] val os = {
    val name = System.getProperty("os.name").toLowerCase
    if (name.contains("linux")) "linux"
    else if (name.contains("os x") || name.contains("darwin")) "darwin"
    else if (name.contains("windows")) "windows"
    else name.replaceAll("\\s", "")
  }

  private[this] val architecture = {
    val arch = System.getProperty("os.arch").toLowerCase
    if (arch == "amd64") "x86_64"
    else arch
  }

  private[this] val currentNativePlatform: String = s"$os-$architecture"

  object autoImport {
    val nativeCompile: TaskKey[Seq[File]] =
      taskKey[Seq[File]]("Builds a native library by calling the native build tool.")

    val nativeClean: TaskKey[Unit] =
      taskKey[Unit]("Cleans the native libraries that native compilation generates.")

    val nativePlatform: SettingKey[String] =
      settingKey[String]("Platform (architecture-kernel) of the system this build is running on.")

    val nativeBuildTool: TaskKey[BuildTool] =
      taskKey[BuildTool]("The build tool to be used when building a native library.")
  }

  import autoImport._

  val nativeBuildToolInstance: TaskKey[BuildTool#Instance] =
    taskKey[BuildTool#Instance]("Get an instance of the current native build tool.")

  lazy val settings: Seq[Setting[_]] = Seq(
    nativePlatform := currentNativePlatform,
    sourceDirectory in nativeCompile := sourceDirectory.value / "native",
    target in nativeCompile := target.value / "native" / nativePlatform.value,
    nativeBuildTool := {
      val tools = Seq(Make, Autotools, CMake)
      val srcDir = (sourceDirectory in nativeCompile).value
      if (!srcDir.exists || !srcDir.isDirectory)
        sys.error(
          s"The provided 'sourceDirectory in nativeCompile' (currently set to '$srcDir') either does not exist, " +
              s"or is not a directory.")
      tools.find(t => t.detect(srcDir)) getOrElse sys.error(
        s"No supported build tool was detected. Make sure that the setting 'sourceDirectory in nativeCompile' " +
            s"(currently set to '$srcDir') points to a directory containing a supported build script. Supported " +
            s"build tools are: ${tools.map(_.name).mkString(",")}.")
    },
    nativeBuildToolInstance := {
      val tool = nativeBuildTool.value
      val srcDir = (sourceDirectory in nativeCompile).value
      val buildDir = (target in nativeCompile).value / "build"
      IO.createDirectory(buildDir)
      tool.getInstance(
        baseDirectory = srcDir,
        buildDirectory = buildDir,
        logger = streams.value.log)
    },
    nativeClean := {
      streams.value.log.info("Cleaning native build")
      try {
        nativeBuildToolInstance.value.clean()
      } catch {
        case exception: Exception =>
          streams.value.log.debug(s"Native build tool clean operation failed with exception: $exception.")
      }
    },
    clean in nativeCompile := nativeClean.value,
    nativeCompile := {
      val tool = nativeBuildTool.value
      val toolInstance = nativeBuildToolInstance.value
      val targetDir = (target in nativeCompile).value / "bin"
      IO.createDirectory(targetDir)
      streams.value.log.info(s"Building library with native build tool ${tool.name}.")
      val libraries = toolInstance.libraries(targetDir)
      streams.value.log.success(s"Libraries built in:\n\t- ${libraries.map(_.getAbsolutePath).mkString("\n\t- ")}")
      libraries
    },
    compile in Compile :=(compile in Compile).dependsOn(nativeCompile).value,
    // Make the SBT clean task also cleans the native sources.
    clean := clean.dependsOn(clean in nativeCompile).value)

  override lazy val projectSettings: Seq[Setting[_]] = settings
}
