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

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.{IOException, InputStream}
import java.nio.file.{Files, Path, StandardCopyOption}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlow {
  final class NativeException(message: String) extends RuntimeException(message)

  private[this] val logger      : Logger = Logger(LoggerFactory.getLogger("TensorFlow Native"))
  private[this] val LIB_NAME    : String = "tensorflow"
  private[this] val JNI_LIB_NAME: String = "tensorflow_jni"

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

  def load(lib: String = JNI_LIB_NAME): Unit = {
    val name = if (lib == LIB_NAME) "TensorFlow native library" else "TensorFlow JNI bindings"
    // If either:
    // (1) The native library has already been statically loaded, or
    // (2) The required native code has been statically linked (through a custom launcher), or
    // (3) The native code is part of another library (such as an application-level library) that has already been
    //     loaded (for example, tensorflow/examples/android and tensorflow/contrib/android include the required native
    //     code in differently named libraries).
    // then it seems that the native library has already been loaded and there is nothing else to do.
    if (!checkIfLoaded() && !tryLoadLibrary(lib)) {
      // Native code is not present, perhaps it has been packaged into the JAR file containing this code.
      val resourceName = makeResourceName(lib)
      Option(Thread.currentThread.getContextClassLoader.getResourceAsStream(resourceName)) match {
        case Some(s) =>
          try {
            val libPath = extractResource(lib, s).toAbsolutePath.toString
            if (lib == LIB_NAME)
              TensorFlow.loadGlobal(libPath)
            else
              System.load(libPath)
            logger.info(s"Loaded the $name as a resource.")
          } catch {
            case exception: IOException =>
              //if (lib != LIB_NAME)
              throw new UnsatisfiedLinkError(
                s"Unable to extract the $name into a temporary file (${exception.getMessage}).")
          }
        case None =>
          //if (lib != LIB_NAME)
          throw new UnsatisfiedLinkError(
            s"Cannot find the $name for OS: $os, and architecture: $architecture. See " +
                "https://github.com/eaplatanios/tensorflow_scala/tree/master/README.md for possible solutions " +
                "(such as building the library from source).")
      }
    }
  }

  private[this] def tryLoadLibrary(lib: String) = {
    val name = if (lib == LIB_NAME) "TensorFlow native library" else "TensorFlow JNI bindings"
    try {
      if (lib == LIB_NAME)
        TensorFlow.loadGlobal(System.mapLibraryName(lib))
      else
        System.loadLibrary(lib)
      true
    } catch {
      case exception: UnsatisfiedLinkError =>
        logger.info(
          s"Failed to load the $name with error: ${exception.getMessage}. Attempting to load it as a resource.")
        false
    }
  }

  private[this] def checkIfLoaded() = {
    try {
      TensorFlow.version
      true
    } catch {
      case _: UnsatisfiedLinkError => false
    }
  }

  private def makeResourceName(lib: String): String = {
    if (lib == LIB_NAME)
      System.mapLibraryName(lib)
    else
      s"native/$os-$architecture/${System.mapLibraryName(lib)}"
  }

  private def extractResource(lib: String, resourceStream: InputStream): Path = {
    val name = if (lib == LIB_NAME) "TensorFlow native library" else "TensorFlow JNI bindings"
    val sampleFilename = System.mapLibraryName(lib)
    val dot = sampleFilename.indexOf(".")
    val prefix = if (dot < 0) sampleFilename else sampleFilename.substring(0, dot)
    val suffix = if (dot < 0) null else sampleFilename.substring(dot)
    val tempFilePath = Files.createTempFile(prefix, suffix)
    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = Files.delete(tempFilePath)
    })
    logger.info(s"Extracting the $name to ${tempFilePath.toAbsolutePath}.")
    try {
      val numBytes = Files.copy(resourceStream, tempFilePath, StandardCopyOption.REPLACE_EXISTING)
      logger.info(String.format(s"Copied $numBytes bytes to ${tempFilePath.toAbsolutePath}."))
    } catch {
      case exception: Exception => throw new UnsatisfiedLinkError(s"Error while extracting the $name: $exception")
    }
    tempFilePath
  }

  load(JNI_LIB_NAME)
  load(LIB_NAME)

  @native def loadGlobal(libPath: String): Unit
  @native def version: String
  @native def dataTypeSize(dataTypeCValue: Int): Int

  // //region Internal API
  //
  // @native private[tensorflow] def updateInput(
  //     graphHandle: Long, inputOpHandle: Long, inputIndex: Int, outputOpHandle: Long, outputIndex: Int): Unit
  // @native private[tensorflow] def addControlInput(graphHandle: Long, opHandle: Long, inputOpHandle: Long): Int
  // @native private[tensorflow] def clearControlInputs(graphHandle: Long, opHandle: Long): Int
  //
  // //endregion Internal API
}
