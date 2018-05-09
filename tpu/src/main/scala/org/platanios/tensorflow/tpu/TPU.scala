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

package org.platanios.tensorflow.tpu

import org.platanios.tensorflow.jni.TensorFlow

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.{IOException, InputStream}
import java.nio.file.{Files, Path, StandardCopyOption}

import scala.collection.JavaConverters._
import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
private[tpu] object TPU {
  private[this] val logger: Logger = Logger(LoggerFactory.getLogger("TensorFlow / TPU"))

  /** TensorFlow TPU ops library name. */
  private[this] val OPS_LIB_NAME: String = "tpu_ops"

  /** Current platform operating system. */
  private[this] val os = {
    val name = System.getProperty("os.name").toLowerCase
    if (name.contains("linux")) {
      // Hack to check if CUDA is installed in the system.
      val result = Process("nvidia-smi").lineStream
      if (result.isEmpty || result.exists(_.contains("command not found"))) {
        logger.info("Detected Linux x86-64 without CUDA support.")
        "linux"
      } else {
        logger.info("Detected Linux x86-64 with CUDA support.")
        "linux-gpu"
      }
    } else if (name.contains("os x") || name.contains("darwin")) {
      logger.info("Detected MacOS x86-64 without CUDA support.")
      "darwin"
    } else if (name.contains("windows")) {
      logger.info("Detected Windows x86-64 without CUDA support.")
      "windows"
    } else {
      name.replaceAll("\\s", "")
    }
  }

  /** Current platform architecture. */
  private[this] val architecture = {
    val arch = System.getProperty("os.arch").toLowerCase
    if (arch == "amd64") "x86_64"
    else arch
  }

  /** Loads the TensorFlow TPU ops library, if provided as a resource. */
  def load(): Unit = this synchronized {
    // Native code is not present, perhaps it has been packaged into the JAR file containing this code.
    val tempDirectory = Files.createTempDirectory("tensorflow_scala_tpu_native_library")
    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = {
        Files.walk(tempDirectory).iterator().asScala.toSeq.reverse.foreach(Files.deleteIfExists)
      }
    })
    val classLoader = Thread.currentThread.getContextClassLoader

    // Load the TensorFlow ops library from the appropriate resource.
    val opsResourceStream = Option(classLoader.getResourceAsStream(makeResourceName(OPS_LIB_NAME)))
    val opsPath = opsResourceStream.map(extractResource(OPS_LIB_NAME, _, tempDirectory))
    if (opsPath.isEmpty)
      throw new UnsatisfiedLinkError(
        s"Cannot find the TensorFlow TPU ops library for OS: $os, and architecture: $architecture. See " +
            "https://github.com/eaplatanios/tensorflow_scala/tree/master/README.md for possible solutions " +
            "(such as building the library from source).")
    opsPath.foreach(path => {
      try {
        System.load(path.toAbsolutePath.toString)
      } catch {
        case exception: IOException => throw new UnsatisfiedLinkError(
          "Unable to load the TensorFlow TPU ops library from the extracted file. This could be due to the " +
              s"TensorFlow native library not being available. Error: ${exception.getMessage}.")
      }
    })

    // Load the TPU ops library from the appropriate resource.
    // TODO: !!! For some reason this can be called twice.
    opsPath.foreach(path => TensorFlow.loadOpLibrary(path.toAbsolutePath.toString))
  }

  /** Maps the provided library name to a filename, similar to [[System.mapLibraryName]]. */
  private def mapLibraryName(lib: String): String = {
    var name = System.mapLibraryName(lib)
    if (os == "darwin" && name.endsWith(".dylib"))
      name = name.substring(0, name.lastIndexOf(".dylib")) + ".so"
    name
  }

  /** Generates the resource name (including the path) for the specified library. */
  private def makeResourceName(lib: String): String = {
    s"native/$os-$architecture/${mapLibraryName(lib)}"
  }

  /** Extracts the resource provided by `inputStream` to `directory`. The filename is the mapped library name for the
    * `lib` library. Returns a path pointing to the extracted file. */
  private def extractResource(lib: String, resourceStream: InputStream, directory: Path): Path = {
    val sampleFilename = mapLibraryName(lib)
    val filePath = directory.resolve(sampleFilename)
    logger.debug(s"Extracting the '$lib' native library to ${filePath.toAbsolutePath}.")
    try {
      val numBytes = Files.copy(resourceStream, filePath, StandardCopyOption.REPLACE_EXISTING)
      logger.debug(String.format(s"Copied $numBytes bytes to ${filePath.toAbsolutePath}."))
    } catch {
      case exception: Exception =>
        throw new UnsatisfiedLinkError(s"Error while extracting the '$lib' native library: $exception")
    }
    filePath
  }

  load()
}
