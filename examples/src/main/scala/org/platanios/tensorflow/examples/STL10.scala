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

package org.platanios.tensorflow.examples

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.data.image.STL10Loader

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
object STL10 {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / STL10"))

  def main(args: Array[String]): Unit = {
    val dataSet = STL10Loader.load(Paths.get("datasets/STL10"), loadUnlabeled = false)
    val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
    val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
    val trainData =
      trainImages.zip(trainLabels)
          .repeat()
          .shuffle(10000)
          .batch(256)
          .prefetch(10)

    logger.info("Building the logistic regression model.")
    val input = tf.learn.Input(UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2), dataSet.trainImages.shape(3)))
    val trainInput = tf.learn.Input(UINT8, Shape(-1))
    val layer = tf.learn.Cast("Input/Cast", FLOAT32) >>
        tf.learn.Conv2D("Layer_0/Conv2D", Shape(5, 5, 3, 32), 1, 1, SamePadding) >>
        tf.learn.AddBias("Layer_0/Bias") >>
        tf.learn.ReLU("Layer_0/ReLU", 0.1f) >>
        tf.learn.MaxPool("Layer_0/MaxPool", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
        tf.learn.Conv2D("Layer_1/Conv2D", Shape(5, 5, 32, 64), 1, 1, SamePadding) >>
        tf.learn.AddBias("Bias_1") >>
        tf.learn.ReLU("Layer_1/ReLU", 0.1f) >>
        tf.learn.MaxPool("Layer_1/MaxPool", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
        tf.learn.Flatten("Layer_2/Flatten") >>
        tf.learn.Linear("Layer_2/Linear", 256) >> tf.learn.ReLU("Layer_2/ReLU", 0.1f) >>
        tf.learn.Linear("OutputLayer/Linear", 10)
    val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)
    val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
        tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.AdaGrad(0.1)
    val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    logger.info("Training the linear regression model.")
    val summariesDir = Paths.get("temp/cnn-stl10")
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        // tf.learn.SummarySaverHook(summariesDir, tf.learn.StepHookTrigger(100)),
      tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))
    estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(10000)))

    def accuracy(images: Tensor, labels: Tensor): Float = {
      val imageBatches = images.splitEvenly(100)
      val labelBatches = labels.splitEvenly(100)
      var accuracy = 0.0f
      (0 until 100).foreach(i => {
        val predictions = estimator.infer(() => imageBatches(i))
        accuracy += predictions.argmax(1).cast(UINT8).equal(labelBatches(i)).cast(FLOAT32).sum().scalar.asInstanceOf[Float]
      })
      accuracy / images.shape(0)
    }

    logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}
