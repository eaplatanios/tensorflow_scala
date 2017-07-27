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
