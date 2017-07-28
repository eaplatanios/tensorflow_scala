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
  private[this] val currentNativePlatform: String = {
    try {
      // We first try to use "uname" and if that fails (e.g., on Windows), we fall back to the JVM information.
      val lines = Process("uname -sm").lines
      if (lines.isEmpty)
        sys.error("An error occured trying to run 'uname'.")
      // uname -sm returns "<kernel> <hardware name>"
      val parts = lines.head.split(" ")
      if (parts.length != 2) {
        sys.error("'uname -sm' returned unexpected string: " + lines.head)
      } else {
        val arch = parts(1).toLowerCase.replaceAll("\\s", "")
        val name = parts(0).toLowerCase.replaceAll("\\s", "")
        s"$arch-$name"
      }
    } catch {
      case _: Exception =>
        val arch = System.getProperty("os.arch").toLowerCase
        val name = System.getProperty("os.name").toLowerCase.split(' ')(0)
        s"$arch-$name"
    }
  }

  private[this] val jniHeadersLocations: Seq[String] = {
    val javaHome = System.getenv("JAVA_HOME")
    currentNativePlatform match {
      case p if p.endsWith("linux") => Seq(s"$javaHome/include", s"$javaHome/include/linux")
      case p if p.endsWith("darwin") => Seq(s"$javaHome/include", s"$javaHome/include/darwin")
      case p if p.endsWith("windows") => Seq(s"$javaHome/include", s"$javaHome/include/win32")
      case _ => sys.error(s"Unsupported platform: $currentNativePlatform.")
    }
  }

  object autoImport {
    val nativeCompile: TaskKey[Seq[File]] =
      taskKey[Seq[File]]("Builds a native library by calling the native build tool.")

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
    clean in nativeCompile := {
      streams.value.log.debug("Cleaning native build")
      try {
        nativeBuildToolInstance.value.clean()
      } catch {
        case exception: Exception =>
          streams.value.log.debug(s"Native build tool clean operation failed with exception: $exception.")
      }
    },
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
    // Make the SBT clean task also clean the native sources.
    clean := {
      (clean in nativeCompile).value
      clean.value
    }
  )

  override lazy val projectSettings: Seq[Setting[_]] = settings
}
