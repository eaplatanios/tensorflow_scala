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

import java.io.File

import sbt._
import sbt.io.Path._
import sbt.Keys._

/** Packages libraries built with JniNative. */
object JniPackage extends AutoPlugin {
  // The JvmPlugin is required or else it will override the resource generators when first included.
  override def requires: Plugins = JniNative && plugins.JvmPlugin
  override def trigger: PluginTrigger = allRequirements

  object autoImport {
    val unmanagedNativeDirectories: SettingKey[Seq[File]] =
      settingKey[Seq[File]](
        "Unmanaged directories containing native libraries. The libraries must be regular files contained in a " +
            "subdirectory corresponding to a platform. For example, " +
            "'<unmanagedNativeDirectory>/x86_64-linux/libfoo.so' is an unmanaged library for machines having the " +
            "'x86_64' architecture and running the Linux kernel.")

    val unmanagedNativeLibraries: TaskKey[Seq[(File, String)]] =
      taskKey[Seq[(File, String)]](
        "Reads 'unmanagedNativeDirectories' and maps libraries to their locations in the classpath " +
            "(i.e., their path in a fat JAR file).")

    val managedNativeLibraries: TaskKey[Seq[(File, String)]] =
      taskKey[Seq[(File, String)]](
        "Maps locally built, platform-dependant libraries to their locations on the classpath.")

    val nativeLibraries: TaskKey[Seq[(File, String)]] =
      taskKey[Seq[(File, String)]]("All native libraries, managed and unmanaged.")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    unmanagedNativeDirectories := Seq(baseDirectory.value / "lib_native"),
    unmanagedNativeLibraries := {
      unmanagedNativeDirectories.value.flatMap(dir => (dir ** "*").get.filter(_.isFile).pair(rebase(dir, "/native")))
    },
    managedNativeLibraries := {
      val platform: String = JniNative.autoImport.nativePlatform.value
      JniNative.autoImport.nativeCompile.value.map(l => l -> s"/native/$platform/${l.name}")
    },
    nativeLibraries := unmanagedNativeLibraries.value ++ managedNativeLibraries.value,
    resourceGenerators += Def.task {
      val libraries: Seq[(File, String)] = nativeLibraries.value
      val resources: Seq[File] = for ((file, path) <- libraries) yield {
        // Native library as a managed resource file.
        val resource = resourceManaged.value / path
        // Copy native library to a managed resource, so that it is always available on the classpath, even when not
        // packaged in a JAR file.
        IO.copyFile(file, resource)
        resource
      }
      resources
    }.taskValue
  )

  override lazy val projectSettings: Seq[Def.Setting[_]] = inConfig(Compile)(settings) ++ inConfig(Test)(settings)
}
