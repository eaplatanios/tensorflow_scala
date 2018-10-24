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

package org.platanios.tensorflow.examples

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.UByte
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.data.image.STL10Loader
import org.platanios.tensorflow.examples

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
object STL10 {
  private val logger = Logger(LoggerFactory.getLogger("Examples / STL10"))

  // Implicit helper for Scala 2.11
  implicit val evOutputStructureFloatLong : OutputStructure[(Output[Float], Output[Long])]  = examples.evOutputStructureFloatLong
  implicit val evOutputToDataTypeFloatLong: OutputToDataType[(Output[Float], Output[Long])] = examples.evOutputToDataTypeFloatLong
  implicit val evOutputToShapeFloatLong   : OutputToShape[(Output[Float], Output[Long])]    = examples.evOutputToShapeFloatLong

  def main(args: Array[String]): Unit = {
    val dataSet = STL10Loader.load(Paths.get("datasets/STL10"), loadUnlabeled = false)
    val trainImages = () => tf.data.datasetFromTensorSlices(dataSet.trainImages).map(_.toFloat)
    val trainLabels = () => tf.data.datasetFromTensorSlices(dataSet.trainLabels).map(_.toLong)
    val trainData = () =>
      trainImages().zip(trainLabels())
          .repeat()
          .shuffle(10000)
          .batch(256)
          .prefetch(10)

    logger.info("Building the logistic regression model.")
    val input = tf.learn.Input(FLOAT32, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2), dataSet.trainImages.shape(3)))
    val trainInput = tf.learn.Input(INT64, Shape(-1))
    val layer = tf.learn.Conv2D[Float]("Layer_0/Conv2D", Shape(5, 5, 3, 32), 1, 1, SameConvPadding) >>
        tf.learn.AddBias[Float]("Layer_0/Bias") >>
        tf.learn.ReLU[Float]("Layer_0/ReLU", 0.1f) >>
        tf.learn.MaxPool[Float]("Layer_0/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        tf.learn.Conv2D[Float]("Layer_1/Conv2D", Shape(5, 5, 32, 64), 1, 1, SameConvPadding) >>
        tf.learn.AddBias[Float]("Bias_1") >>
        tf.learn.ReLU[Float]("Layer_1/ReLU", 0.1f) >>
        tf.learn.MaxPool[Float]("Layer_1/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        tf.learn.Flatten[Float]("Layer_2/Flatten") >>
        tf.learn.Linear[Float]("Layer_2/Linear", 256) >> tf.learn.ReLU("Layer_2/ReLU", 0.1f) >>
        tf.learn.Linear[Float]("OutputLayer/Linear", 10)
    val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss/CrossEntropy") >>
        tf.learn.Mean[Float]("Loss/Mean") >>
        tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")
    val optimizer = tf.train.AdaGrad(0.1f)

    val model = tf.learn.Model.simpleSupervised(
      input = input,
      trainInput = trainInput,
      layer = layer,
      loss = loss,
      optimizer = optimizer)

    logger.info("Training the linear regression model.")
    val summariesDir = Paths.get("temp/cnn-stl10")
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))
    estimator.train(trainData, tf.learn.StopCriteria(maxSteps = Some(10000)))

    def accuracy(images: Tensor[UByte], labels: Tensor[UByte]): Float = {
      val imageBatches = images.toFloat.splitEvenly(100)
      val labelBatches = labels.toLong.splitEvenly(100)
      var accuracy = 0.0f
      (0 until 100).foreach(i => {
        val predictions = estimator.infer(() => imageBatches(i))
        accuracy += predictions
            .argmax(1).toLong
            .equal(labelBatches(i)).toFloat
            .sum().scalar
      })
      accuracy / images.shape(0)
    }

    logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}
