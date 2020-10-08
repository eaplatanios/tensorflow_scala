/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import scala.io.{Codec, Source}

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorFlow {
  private val logger: Logger = Logger(LoggerFactory.getLogger("TensorFlow Native"))

  /** TensorFlow native library name. */
  private val LIB_NAME: String = "tensorflow"

  /** TensorFlow native framework library name. */
  private val LIB_FRAMEWORK_NAME: String = "tensorflow_framework"

  /** TensorFlow JNI bindings library name. */
  private val JNI_LIB_NAME: String = "tensorflow_jni"

  /** TensorFlow ops library name. */
  private val OPS_LIB_NAME: String = "tensorflow_ops"

  /** Current platform. */
  private val platform = {
    val name = System.getProperty("os.name").toLowerCase
    if (name.contains("linux")) "linux"
    else if (name.contains("os x") || name.contains("darwin")) "darwin"
    else if (name.contains("windows")) "windows"
    else name.replaceAll("\\s", "")
  }

  /** Loads the TensorFlow JNI bindings library along with the TensorFlow native library, if provided as a resource. */
  def load(): Unit = this synchronized {
    // If either:
    // (1) The native library has already been statically loaded, or
    // (2) The required native code has been statically linked (through a custom launcher), or
    // (3) The native code is part of another library (such as an application-level library) that has already been
    //     loaded (for example, tensorflow/examples/android and tensorflow/contrib/android include the required native
    //     code in differently named libraries).
    // then it seems that the native library has already been loaded and there is nothing else to do.
    if (!checkIfLoaded()) {
      // Native code is not present, perhaps it has been packaged into the JAR file containing this code.
      val tempDirectory = Files.createTempDirectory("tensorflow_scala_native_libraries")
      tempDirectory.toFile.deleteOnExit()
      val classLoader = Thread.currentThread.getContextClassLoader

      // Check if a TensorFlow native framework library resources are provided and load them.
      (makeResourceNames(LIB_FRAMEWORK_NAME) ++ makeResourceNames(LIB_NAME)).foreach {
        case (name, path, preLoad) =>
          val resource = extractResource(name, classLoader.getResourceAsStream(path), tempDirectory)
          if (preLoad) {
            resource.foreach(r => System.load(r.toAbsolutePath.toString))
          }
      }

      // Load the TensorFlow JNI bindings from the appropriate resource.
      val jniPaths = makeResourceNames(JNI_LIB_NAME).flatMap {
        case (name, path, _) => extractResource(name, classLoader.getResourceAsStream(path), tempDirectory)
      }
      if (jniPaths.isEmpty) {
        throw new UnsatisfiedLinkError(
          s"Cannot find the TensorFlow JNI bindings for platform: $platform. See " +
              "https://github.com/eaplatanios/tensorflow_scala/tree/master/README.md for possible solutions " +
              "(such as building the library from source).")
      }
      jniPaths.foreach(path => {
        try {
          System.load(path.toAbsolutePath.toString)
        } catch {
          case exception: IOException => throw new UnsatisfiedLinkError(
            "Unable to load the TensorFlow JNI bindings from the extracted file. This could be due to the TensorFlow " +
                s"native library not being available. Error: ${exception.getMessage}.")
        }
      })

      // Load the TensorFlow ops library from the appropriate resource.
      val opsPaths = makeResourceNames(OPS_LIB_NAME).flatMap {
        case (name, path, _) => extractResource(name, classLoader.getResourceAsStream(path), tempDirectory)
      }
      opsPaths.foreach(path => loadOpLibrary(path.toAbsolutePath.toString))
    }
  }

  /** Checks if the TensorFlow JNI bindings library has been loaded. */
  private def checkIfLoaded(): Boolean = {
    try {
      TensorFlow.version
      true
    } catch {
      case _: UnsatisfiedLinkError => false
    }
  }

  /** Maps the provided library name to a set of filenames, similar to [[System.mapLibraryName]], but considering all
    * combinations of `dylib` and `so` extensions, along with versioning for TensorFlow 2.x. */
  private def mapLibraryName(lib: String): Seq[(String, Boolean)] = {
    if (platform == "windows") {
      Seq((s"$lib.dll", true), (s"$lib.lib", false))
    } else if (lib == JNI_LIB_NAME || lib == OPS_LIB_NAME) {
      Seq((s"lib$lib.so", false))
    } else {
      Seq(
        (s"lib$lib.so.link", false),
        (s"lib$lib.so.2.link", false),
        (s"lib$lib.so.2.3.1", false),
        (s"lib$lib.dylib.link", false),
        (s"lib$lib.2.dylib.link", false),
        (s"lib$lib.2.3.1.dylib", false),
      )
    }
  }

  /** Generates the resource names and paths for the specified library. */
  private def makeResourceNames(lib: String): Seq[(String, String, Boolean)] = {
    if (lib == LIB_NAME || lib == LIB_FRAMEWORK_NAME) {
      mapLibraryName(lib).map { case (name, preLoad) => (name, name, preLoad) }
    } else {
      mapLibraryName(lib).map { case (name, preLoad) => (name, s"native/$platform/$name", preLoad) }
    }
  }

  /** Extracts the resource provided by `inputStream` to `directory`. The filename is the mapped library name for the
    * `lib` library. Returns a path pointing to the extracted file. */
  private def extractResource(filename: String, resourceStream: InputStream, directory: Path): Option[Path] = {
    if (resourceStream != null) {
      val filePath = directory.resolve(filename)
      logger.debug(s"Extracting the '$filename' native library to ${filePath.toAbsolutePath}.")
      try {
        val (streamToCopy, path) = if (filename.endsWith(".link")) {
          val linkPath = Source.fromInputStream(resourceStream)(Codec.UTF8).mkString
          val stream = Thread.currentThread.getContextClassLoader.getResourceAsStream(linkPath)
          (stream, directory.resolve(filename.dropRight(5)))
        } else {
          (resourceStream, filePath)
        }
        val numBytes = Files.copy(streamToCopy, path, StandardCopyOption.REPLACE_EXISTING)
        logger.debug(String.format(s"Copied $numBytes bytes to ${filePath.toAbsolutePath}."))
      } catch {
        case exception: Exception =>
          throw new UnsatisfiedLinkError(s"Error while extracting the '$filename' native library: $exception")
      }
      Some(filePath)
    } else {
      None
    }
  }

  load()

  lazy val currentJvmPointer: Long = jvmPointer

  @native private[jni] def jvmPointer: Long
  @native def version: String
  @native def dataTypeSize(dataTypeCValue: Int): Int
  @native def loadOpLibrary(libraryPath: String): Array[Byte]
  @native def enableXLA(): Unit

  //region Internal API

  @native private[tensorflow] def updateInput(
      graphHandle: Long, inputOpHandle: Long, inputIndex: Int, outputOpHandle: Long, outputIndex: Int): Unit
  @native private[tensorflow] def addControlInput(graphHandle: Long, opHandle: Long, inputOpHandle: Long): Int
  @native private[tensorflow] def clearControlInputs(graphHandle: Long, opHandle: Long): Int
  @native private[tensorflow] def setRequestedDevice(graphHandle: Long, opHandle: Long, device: String): Int
  @native private[tensorflow] def setAttributeProto(
      graphHandle: Long, opHandle: Long, attributeName: String, attributeValue: Array[Byte]): Unit

  //endregion Internal API
}
