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

import OpGenerator._

import java.nio.file.{Files, Paths}

import scala.collection.JavaConverters._

import sbt._
import sbt.Keys._

/** Adds functionality for generating JNI header and implementation files for executing TensorFlow eager tensor ops.
  *
  * @author Emmanouil Antonios Platanios
  */
object TensorFlowGenerateTensorOps extends AutoPlugin {
  override def requires: Plugins = plugins.JvmPlugin

  object autoImport {
    val generateTensorOps: TaskKey[Unit] = taskKey[Unit](
      "Generates the TensorFlow tensor ops bindings. Returns the directory containing generated bindings.")

    val ops: SettingKey[Map[String, Seq[String]]] = settingKey[Map[String, Seq[String]]](
      "Grouped ops for which to generate bindings.")

    val scalaPackage: SettingKey[String] = settingKey[String](
      "Grouped ops for which to generate bindings.")
  }

  import autoImport._

  lazy val settings: Seq[Setting[_]] = Seq(
    scalaPackage in generateTensorOps := "tensors",
    target in generateTensorOps := target.value,
    ops in generateTensorOps := Map.empty,
    clean in generateTensorOps := {
      streams.value.log.info("Cleaning generated TensorFlow tensor op files.")
      val path = (target in generateTensorOps).value.toPath
      val scalaPackage = "org.platanios.tensorflow.jni.generated"
      val scalaPath = path.resolve(Paths.get("scala", scalaPackage.split('.'): _*))
      val nativePath = path.resolve(Paths.get("native", "generated"))
      if (Files.exists(scalaPath))
        Files.walk(scalaPath)
            .iterator()
            .asScala
            .toSeq
            .reverse
            .foreach(Files.deleteIfExists)
      if (Files.exists(nativePath))
        Files.walk(nativePath)
            .iterator()
            .asScala
            .toSeq
            .reverse
            .foreach(Files.deleteIfExists)
    },
    generateTensorOps := {
      val log = streams.value.log
      val opsPBFile = (target in generateTensorOps).value / "resources" / "ops.pbtxt"
      val cachedFunction = FileFunction.cached(streams.value.cacheDirectory)(opsFiles => {
        log.info("Generating TensorFlow tensor op files.")
        generateFiles(
          opsFiles.head,
          (target in generateTensorOps).value.toPath,
          (ops in generateTensorOps).value,
          s"org.platanios.tensorflow.jni.generated.${(scalaPackage in generateTensorOps).value}")
        Set.empty
      })
      cachedFunction(Set(opsPBFile))
    })

  override lazy val projectSettings: Seq[Setting[_]] = settings
}
