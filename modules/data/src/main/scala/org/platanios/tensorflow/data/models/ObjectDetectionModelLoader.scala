/*
 * Copyright 2017 Sören Brunk
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.platanios.tensorflow.data.models

import org.platanios.tensorflow.api.{Graph, Output}
import org.platanios.tensorflow.api.core.types.{FLOAT32, UINT8}
import org.platanios.tensorflow.data.Loader
import org.platanios.tensorflow.data.utilities.CompressedFiles

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.GraphDef

import java.io.{BufferedInputStream, FileInputStream}
import java.nio.file.{Files, Path}

/** Loader for object detection models compatible with the TensorFlow object detection API.
  *
  * The loader first looks if a model with the given name exists locally, enabling to use of custom models.
  * If no model is found locally, the loader tries to download it from the TensorFlow detection model zoo.
  *
  * See [[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]] for
  * details about the models available.
  *
  * Currently, it only supports loading the frozen inference graph for doing predictions.
  *
  * @author Sören Brunk
  */
class ObjectDetectionModelLoader(modelName: String) extends Loader {
  override protected val logger = Logger(LoggerFactory.getLogger("Object Detection Model Loader"))

  protected val baseUrl           : String = "http://download.tensorflow.org/models/object_detection/"
  protected val compressedFilename: String = modelName + ".tar.gz"
  protected val graphFilename     : String = "frozen_inference_graph.pb"

  /** Loads a serialized object detection model. Tries to download the model if not found locally.
    *
    * @param path Path where the model is stored.
    * @param bufferSize
    * @return
    */
  def load(path: Path, bufferSize: Int = 8192): ObjectDetectionModel = {
    if (!Files.exists(path.resolve(modelName))) {
      // Download the data, if necessary.
      maybeDownload(path.resolve(compressedFilename), baseUrl + compressedFilename, bufferSize)

      // Extract the model.
      logger.info(s"Extracting data from file '${path.resolve(compressedFilename)}'.")
      CompressedFiles.decompressTGZ(path.resolve(compressedFilename), path, bufferSize)
    }

    // Load the pretrained detection model as TensorFlow graph.
    logger.info(s"Loading TensorFlow graph from detection model '$modelName'.")
    val graphDef = GraphDef.parseFrom(
      new BufferedInputStream(new FileInputStream(path.resolve(modelName).resolve(graphFilename).toFile)))
    val graph = Graph.fromGraphDef(graphDef)
    new ObjectDetectionModel(graph)
  }
}

/** Convenience access to object detection model input and output placeholders. */
class ObjectDetectionModel(val graph: Graph) {
  /** Placeholder for the input image. dataType = [[UINT8]], shape = Shape(1, -1, -1, 3) */
  val inputImage: Output[Any] = graph.getOutputByName("image_tensor:0")
  /** Placeholder for the detected boxes. dataType = [[FLOAT32]], shape = Shape(1, numDetections, 4) */
  val detectionBoxes: Output[Any] = graph.getOutputByName("detection_boxes:0")
  /** Placeholder for the detected scores. dataType = [[FLOAT32]], shape = Shape(1, numDetections) */
  val detectionScores: Output[Any] = graph.getOutputByName("detection_scores:0")
  /** Placeholder for the detected labels. dataType = [[FLOAT32]], shape = Shape(1, numDetections) */
  val detectionClasses: Output[Any] = graph.getOutputByName("detection_classes:0")
  /** Placeholder for the number of detected objects. [[FLOAT32]], shape = Shape(1) */
  val numDetections: Output[Any] = graph.getOutputByName("num_detections:0")
}
