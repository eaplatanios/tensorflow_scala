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

package org.platanios.tensorflow.horovod

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
private[horovod] object Horovod {
  private[this] val logger: Logger = Logger(LoggerFactory.getLogger("TensorFlow / Horovod"))

  /** TensorFlow Horovod JNI bindings library name. */
  private[this] val JNI_LIB_NAME: String = "horovod_jni"

  /** Current platform operating system. */
  private[this] val os = {
    val name = System.getProperty("os.name").toLowerCase
    if (name.contains("linux")) {
      // Hack to check if CUDA is installed in the system.
      val result = Process("nvidia-smi").lineStream
      if (result.isEmpty || result.exists(_.contains("command not found")))
        "linux"
      else
        "linux-gpu"
    } else if (name.contains("os x") || name.contains("darwin")) {
      "darwin"
    } else if (name.contains("windows")) {
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

  /** Loads the TensorFlow Horovod JNI bindings library, if provided as a resource. */
  def load(): Unit = this synchronized {
    // If either:
    // (1) The native library has already been statically loaded, or
    // (2) The required native code has been statically linked (through a custom launcher), or
    // (3) The native code is part of another library (such as an application-level library) that has already been
    //     loaded.
    // then it seems that the native library has already been loaded and there is nothing else to do.
    if (!checkIfLoaded()) {
      // Native code is not present, perhaps it has been packaged into the JAR file containing this code.
      val tempDirectory = Files.createTempDirectory("tensorflow_scala_horovod_native_library")
      Runtime.getRuntime.addShutdownHook(new Thread() {
        override def run(): Unit = {
          Files.walk(tempDirectory).iterator().asScala.toSeq.reverse.foreach(Files.deleteIfExists)
        }
      })
      val classLoader = Thread.currentThread.getContextClassLoader

      // Load the TensorFlow JNI bindings from the appropriate resource.
      val jniResourceStream = Option(classLoader.getResourceAsStream(makeResourceName(JNI_LIB_NAME)))
      val jniPath = jniResourceStream.map(extractResource(JNI_LIB_NAME, _, tempDirectory))
      if (jniPath.isEmpty)
        throw new UnsatisfiedLinkError(
          s"Cannot find the TensorFlow Horovod JNI bindings for OS: $os, and architecture: $architecture. See " +
              "https://github.com/eaplatanios/tensorflow_scala/tree/master/README.md for possible solutions " +
              "(such as building the library from source).")
      jniPath.foreach(path => {
        try {
          System.load(path.toAbsolutePath.toString)
        } catch {
          case exception: IOException => throw new UnsatisfiedLinkError(
            "Unable to load the TensorFlow Horovod JNI bindings from the extracted file. This could be due to the " +
                s"TensorFlow native library not being available. Error: ${exception.getMessage}.")
        }
      })

      // Load the Horovod ops library from the appropriate resource.
      // TODO: !!! For some reason this can be called twice.
      jniPath.foreach(path => TensorFlow.loadOpLibrary(path.toAbsolutePath.toString))
    }
  }

  /** Checks if the TensorFlow JNI bindings library has been loaded. */
  private[this] def checkIfLoaded(): Boolean = {
    try {
      Horovod.rank()
      true
    } catch {
      case _: UnsatisfiedLinkError => false
    }
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

  @native def init(): Unit
  @native def rank(): Int
  @native def localRank(): Int
  @native def size(): Int
  @native def localSize(): Int
  @native def mpiThreadsSupported(): Int
}
