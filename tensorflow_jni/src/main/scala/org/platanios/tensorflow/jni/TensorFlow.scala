package org.platanios.tensorflow.jni

import java.nio.file.{Files, Path}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlow {
  final class NativeException(message: String) extends RuntimeException(message)

  private[this] val nativeLibraryName = "tensorflow_jni"

  def loadPackaged(): Unit = {
    val lib: String = System.mapLibraryName(nativeLibraryName)
    val tmp: Path = Files.createTempDirectory("jni-")
    val plat: String = {
      val line = try {
        scala.sys.process.Process("uname -sm").lines.head
      } catch {
        case ex: Exception => sys.error("Error running `uname` command")
      }
      val parts = line.split(" ")
      if (parts.length != 2) {
        sys.error("Could not determine platform: 'uname -sm' returned unexpected string: " + line)
      } else {
        val arch = parts(1).toLowerCase.replaceAll("\\s", "")
        val kernel = parts(0).toLowerCase.replaceAll("\\s", "")
        arch + "-" + kernel
      }
    }

    val resourcePath: String = "/native/" + plat + "/" + lib
    val resourceStream = Option(TensorFlow.getClass.getResourceAsStream(resourcePath)) match {
      case Some(s) => s
      case None => throw new UnsatisfiedLinkError(
        "Native library " + lib + " (" + resourcePath + ") cannot be found on the classpath.")
    }
    val extractedPath = tmp.resolve(lib)
    try {
      Files.copy(resourceStream, extractedPath)
    } catch {
      case ex: Exception => throw new UnsatisfiedLinkError(
        "Error while extracting native library: " + ex)
    }
    System.load(extractedPath.toAbsolutePath.toString)
  }

  def load(): Unit = {
    try {
      System.loadLibrary(nativeLibraryName)
    } catch {
      case ex: UnsatisfiedLinkError => loadPackaged()
    }
  }

  load()

  @native def version: String
  @native def dataTypeSize(dataTypeCValue: Int): Int
}
