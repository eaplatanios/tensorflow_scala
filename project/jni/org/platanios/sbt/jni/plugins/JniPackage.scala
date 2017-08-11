package plugin.org.platanios.sbt.jni.plugins

import sbt._
import sbt.Keys._
import java.io.File

/** Packages libraries built with JniNative. */
object JniPackage extends AutoPlugin {

  // JvmPlugin is required or else it will override resource generators when first included
  override def requires = JniNative && plugins.JvmPlugin
  override def trigger = allRequirements

  object autoImport {

    val enableNativeCompilation = settingKey[Boolean](
      "Determines if native compilation is enabled. If not enabled, only pre-compiled libraries in " +
          "`unmanagedNativeDirectories` will be packaged."
    )

    val unmanagedNativeDirectories = settingKey[Seq[File]](
      "Unmanaged directories containing native libraries. The libraries must be regular files " +
          "contained in a subdirectory corresponding to a platform. For example " +
          "`<unmanagedNativeDirectory>/x86_64-linux/libfoo.so` is an unmanaged library for machines having " +
          "the x86_64 architecture and running the Linux kernel."
    )

    val unmanagedNativeLibraries = taskKey[Seq[(File, String)]](
      "Reads `unmanagedNativeDirectories` and maps libraries to their locations on the classpath " +
          "(i.e. their path in a fat jar)."
    )

    val managedNativeLibraries = taskKey[Seq[(File, String)]](
      "Maps locally built, platform-dependant libraries to their locations on the classpath."
    )

    val nativeLibraries = taskKey[Seq[(File, String)]](
      "All native libraries, managed and unmanaged."
    )

  }

  import autoImport._
  import JniNative.autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(

    enableNativeCompilation := true,

    unmanagedNativeDirectories := Seq(baseDirectory.value / "lib_native"),

    unmanagedNativeLibraries := {
      val baseDirs: Seq[File] = unmanagedNativeDirectories.value
      val mappings: Seq[(File, String)] = unmanagedNativeDirectories.value.flatMap { dir =>
        val files: Seq[File] = (dir ** "*").get.filter(_.isFile)
        files pair rebase(dir, "/native")
      }
      mappings
    },

    managedNativeLibraries := Def.taskDyn[Seq[(File, String)]] {
      val enableManaged = (enableNativeCompilation).value
      if (enableManaged) Def.task {
        val library: File = nativeCompile.value
        val platform = nativePlatform.value

        Seq(library -> s"/native/$platform/${library.name}")
      }
      else Def.task {
        Seq.empty
      }
    }.value,

    nativeLibraries := unmanagedNativeLibraries.value ++ managedNativeLibraries.value,

    resourceGenerators += Def.task {
      val libraries: Seq[(File, String)] = nativeLibraries.value
      val resources: Seq[File] = for ((file, path) <- libraries) yield {

        // native library as a managed resource file
        val resource = resourceManaged.value / path

        // copy native library to a managed resource, so that it is always available
        // on the classpath, even when not packaged as a jar
        IO.copyFile(file, resource)
        resource
      }
      resources
    }.taskValue

  )

  override lazy val projectSettings = inConfig(Compile)(settings) ++ inConfig(Test)(settings) ++
      Seq(crossPaths := false) // don't add scala version to native jars

}
