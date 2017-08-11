package plugin.org.platanios.sbt.jni.build

import java.io.File
import sbt._

trait ConfigureMakeInstall { self: BuildTool =>

  /* API for native build tools that use a standard 'configure && make && make install' process,
   * where the configure step is left ab
   stract. */
  trait Instance extends self.Instance {

    def log: Logger
    def baseDirectory: File
    def buildDirectory: File

    def clean() = Process("make clean", buildDirectory) ! log

    def configure(targetDirectory: File): ProcessBuilder

    def make(): ProcessBuilder = Process("make VERBOSE=1", buildDirectory)

    def install(): ProcessBuilder = Process("make install", buildDirectory)

    def library(
        targetDirectory: File
    ): File = {

      val ev: Int = (
          configure(targetDirectory) #&& make() #&& install()
          ) ! log

      if (ev != 0) sys.error(s"Building native library failed. Exit code: ${ev}")

      val products: List[File] =
        (targetDirectory ** ("*.so" | "*.dylib")).get.filter(_.isFile).toList

      // only one produced library is expected
      products match {
        case Nil =>
          sys.error(s"No files were created during compilation, " +
                        s"something went wrong with the ${name} configuration.")
        case head :: Nil =>
          head
        case head :: tail =>
          log.warn("More than one file was created during compilation, " +
                       s"only the first one (${head.getAbsolutePath}) will be used.")
          head
      }
    }
  }

}
