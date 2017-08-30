import java.io.File

import sbt._
//import sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
object BuildTool {
  sealed trait BuildTool {
    /** Name of this build tool. */
    val name: String

    /** Detects whether or not this build tool is configured in the provided directory.
      *
      * For example, for the Make build tool, this would return `true` if a Makefile is present in the directory.
      *
      * @param  baseDirectory Directory to check.
      * @return Boolean value indicating whether or not this build tool is configured in the provided directory.
      */
    def detect(baseDirectory: File): Boolean

    /** Represents an instance (i.e., a build configuration), for this build tool, that contains the actual tasks that
      * can be perfomed on a specific configuration (such as those configured in your Makefile). */
    trait Instance {
      /** Invokes the native build tool's clean task */
      def clean(): Unit

      /** Invokes the native build tool's main task, resulting in a single shared library file.
        *
        * @param  targetDirectory The directory into which the shared library is copied.
        * @return The shared library file.
        */
      def libraries(targetDirectory: File): Seq[File]
    }

    /** Gets an instance (i.e., a build configuration) of this tool for the specified directory. */
    def getInstance(baseDirectory: File, buildDirectory: File, logger: Logger): Instance
  }

  /** Trait that defines an API for native build tools that use a standard `configure && make && make install` process,
    * where the configure step is left abstract. */
  sealed trait ConfigureMakeInstall { self: BuildTool =>
    trait Instance extends self.Instance {
      val log           : Logger
      val baseDirectory : File
      val buildDirectory: File

      def clean(): Unit = Process("make clean", buildDirectory) ! log

      def configure(targetDirectory: File): ProcessBuilder

      def make(): ProcessBuilder = Process("make VERBOSE=1", buildDirectory)

      def install(): ProcessBuilder = Process("make install", buildDirectory)

      def libraries(targetDirectory: File): Seq[File] = {
        val exitCode: Int = (configure(targetDirectory) #&& make() #&& install()) ! log
        if (exitCode != 0)
          sys.error(s"Failed to build the native library. Exit code: $exitCode.")
        val products: List[File] = (targetDirectory ** ("*.so" | "*.dylib" | "*.dll")).get.filter(_.isFile).toList
        if (products == Nil)
          sys.error(s"No files were created during compilation, something went wrong with the $name configuration.")
        products
      }
    }
  }

  /** Make build tool. */
  object Make extends BuildTool with ConfigureMakeInstall {
    override val name: String = "Make"

    override def detect(baseDirectory: File): Boolean = baseDirectory.list().contains("Makefile")

    override def getInstance(baseDir: File, buildDir: File, logger: Logger) = new Instance {
      override val log           : Logger = logger
      override val baseDirectory : File   = baseDir
      override val buildDirectory: File   = buildDir

      override def configure(target: File): ProcessBuilder = Process(
        s"cp ${baseDirectory.getAbsolutePath}/Makefile $buildDirectory/Makefile", buildDirectory)
    }
  }

  /** Autotools build tool. */
  object Autotools extends BuildTool with ConfigureMakeInstall {
    val name: String = "Autotools"

    def detect(baseDirectory: File): Boolean = baseDirectory.list().contains("configure")

    override def getInstance(baseDir: File, buildDir: File, logger: Logger) = new Instance {
      override val log           : Logger = logger
      override val baseDirectory : File   = baseDir
      override val buildDirectory: File   = buildDir

      override def configure(target: File): ProcessBuilder = Process(
        // Disable producing versioned library files since that is not needed for fat JAR files.
        s"${baseDirectory.getAbsolutePath}/configure " +
            s"--prefix=${target.getAbsolutePath} " +
            s"--libdir=${target.getAbsolutePath}  " +
            "--disable-versioned-lib",
        buildDirectory)
    }
  }

  /** CMake build tool. */
  object CMake extends BuildTool with ConfigureMakeInstall {
    override val name: String = "CMake"

    override def detect(baseDirectory: File): Boolean = baseDirectory.list().contains("CMakeLists.txt")

    override def getInstance(baseDir: File, buildDir: File, logger: Logger) = new Instance {
      override val log           : Logger = logger
      override val baseDirectory : File   = baseDir
      override val buildDirectory: File   = buildDir

      override def configure(target: File): ProcessBuilder = Process(
        // Disable producing versioned library files since that is not needed for fat JAR files.
        "cmake " +
            s"-DCMAKE_INSTALL_PREFIX:PATH=${target.getAbsolutePath} " +
            "-DCMAKE_BUILD_TYPE=Release " +
            baseDirectory.getAbsolutePath,
        buildDirectory)
    }
  }
}
