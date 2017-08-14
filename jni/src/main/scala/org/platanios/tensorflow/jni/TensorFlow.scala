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

import java.io.{IOException, InputStream}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, StandardCopyOption}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlow {
  final class NativeException(message: String) extends RuntimeException(message)

  private[this] val logger          = Logger(LoggerFactory.getLogger("TensorFlow Native"))
  private[this] val LIBNAME: String = "tensorflow_jni"

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

  def load(): Unit = {
    // If either:
    // (1) The native library has already been statically loaded, or
    // (2) The required native code has been statically linked (through a custom launcher), or
    // (3) The native code is part of another library (such as an application-level library) that has already been
    //     loaded (for example, tensorflow/examples/android and tensorflow/contrib/android include the required native
    //     code in differently named libraries).
    // then it seems that the native library has already been loaded and there is nothing else to do.
    if (!checkIfLoaded() && !tryLoadLibrary()) {
      // Native code is not present, perhaps it has been packaged into the JAR file containing this code.
      val resourceName = makeResourceName()
      val resourceStream = Option(TensorFlow.getClass.getResourceAsStream(resourceName)) match {
        case Some(s) => s
        case None => throw new UnsatisfiedLinkError(
          s"Cannot find TensorFlow native library for OS: $os, and architecture: $architecture. See " +
              "https://github.com/eaplatanios/tensorflow_scala/tree/master/README.md for possible solutions (such as " +
              "building the library from source).")
      }
      try {
        val resourcePath = extractResource(resourceStream)
        System.load(resourcePath.toAbsolutePath.toString)
        logger.info("Loaded the TensorFlow native library as a resource.")
      } catch {
        case exception: IOException => throw new UnsatisfiedLinkError(
          s"Unable to extract the TensorFlow native library into a temporary file (${exception.getMessage}).")
      }
    }
  }

  private[this] def tryLoadLibrary() = {
    try {
      System.loadLibrary(LIBNAME)
      true
    } catch {
      case exception: UnsatisfiedLinkError =>
        logger.info(
          s"Failed to load the TensorFlow native library with error: ${exception.getMessage}. " +
              "Attempting to load it as a resource.")
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

  private def makeResourceName() = s"/native/$os-$architecture/${System.mapLibraryName(LIBNAME)}"

  private def extractResource(resourceStream: InputStream): Path = {
    val sampleFilename = System.mapLibraryName(LIBNAME)
    val dot = sampleFilename.indexOf(".")
    val prefix = if (dot < 0) sampleFilename else sampleFilename.substring(0, dot)
    val suffix = if (dot < 0) null else sampleFilename.substring(dot)
    val tempFilePath = Files.createTempFile(prefix, suffix)
    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = Files.delete(tempFilePath)
    })
    logger.info(s"Extracting TensorFlow native library to ${tempFilePath.toAbsolutePath}.")
    try {
      val numBytes = Files.copy(resourceStream, tempFilePath, StandardCopyOption.REPLACE_EXISTING)
      logger.info(String.format(s"Copied $numBytes bytes to ${tempFilePath.toAbsolutePath}."))
    } catch {
      case exception: Exception => throw new UnsatisfiedLinkError(
        "Error while extracting TensorFlow native library: " + exception)
    }
    tempFilePath
  }

  load()

  @native def version: String
  @native def dataTypeSize(dataTypeCValue: Int): Int

  //region Internal API

  @native private[tensorflow] def updateInput(
      graphHandle: Long, inputOpHandle: Long, inputIndex: Int, outputOpHandle: Long, outputIndex: Int): Unit
  @native private[tensorflow] def addControlInput(graphHandle: Long, opHandle: Long, inputOpHandle: Long): Int
  @native private[tensorflow] def clearControlInputs(graphHandle: Long, opHandle: Long): Int

  //endregion Internal API
}
