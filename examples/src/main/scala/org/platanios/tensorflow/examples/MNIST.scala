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
import org.platanios.tensorflow.api.learn.Model
import org.platanios.tensorflow.data.loaders.MNISTLoader

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
object MNIST {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / MNIST"))

  def mlpPredictionModel(architecture: Seq[Int]): tf.learn.Layer[tf.Output, tf.Output] = {
    architecture.zipWithIndex
        .map(a => tf.learn.withVariableScope(s"Layer_${a._2}")(tf.learn.linear(a._1) >> tf.learn.relu(0.1f)))
        .foldLeft(tf.learn.flatten() >> tf.learn.cast(FLOAT32))(_ >> _) >>
        tf.learn.withVariableScope("OutputLayer")(tf.learn.linear(10)) // >> tf.learn.logSoftmax())
  }

  def main(args: Array[String]): Unit = {
    val dataSet = MNISTLoader.load(Paths.get("temp/"))
    val trainImages = tf.datasetFromSlices(dataSet.trainImages).repeat().batch(1024)
    val trainLabels = tf.datasetFromSlices(dataSet.trainLabels).repeat().batch(1024)

    logger.info("Building the logistic regression model.")
    val input = tf.learn.input(UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2)))
    val layer = tf.learn.flatten() >>
        tf.learn.cast(FLOAT32) >>
        tf.learn.linear(10) // >> tf.learn.logSoftmax()
    val trainInput = tf.learn.input(UINT8, Shape(-1))
    val trainingInputLayer = tf.learn.cast(INT64) // tf.learn.oneHot(10) >> tf.learn.cast(FLOAT32)
    val loss = tf.learn.sparseSoftmaxCrossEntropy() >> tf.learn.mean()
    val optimizer = tf.learn.adaGrad()
    val model = tf.learn.model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

    logger.info("Training the linear regression model.")
    model.initialize(Model.DefaultInitialization)
    model.train(trainImages, trainLabels, maxIterations = 100000)

    // val inputs = tf.placeholder(tf.UINT8, tf.shape(-1, numberOfRows, numberOfColumns))
    // val labels = tf.placeholder(tf.UINT8, tf.shape(-1))

    // // val predictions = mlpPredictionModel(Seq(128))(inputs)
    // val predictions = linearPredictionModel(inputs)
    // val supervision = supervisionModel(labels)

    // val loss = lossFunction(predictions, supervision)
    // val trainOp = tfl.adaGrad().minimize(loss)

    // val correctPredictions = tf.equal(tf.argmax(predictions, 1), supervision) // tf.argmax(supervision, 1))
    // val accuracy = tf.mean(tf.cast(correctPredictions, tf.FLOAT32))

    // logger.info("Training the linear regression model.")
    // logger.info(f"${"Iteration"}%10s | ${"Train Loss"}%13s | ${"Test Accuracy"}%13s")
    // val session = tf.session()
    // session.run(targets = tf.globalVariablesInitializer())
    // for (i <- 0 to 1000) {
    //   val feeds = Map(inputs -> dataSet.trainImages, labels -> dataSet.trainLabels)
    //   val trainLoss = session.run(feeds = feeds, fetches = loss, targets = trainOp)
    //   if (i % 1 == 0) {
    //     val testFeeds = Map(inputs -> dataSet.testImages, labels -> dataSet.testLabels)
    //     val testAccuracy = session.run(feeds = testFeeds, fetches = accuracy)
    //     logger.info(f"$i%10d | " +
    //                     f"${trainLoss.scalar.asInstanceOf[Float]}%13.4e | " +
    //                     f"${testAccuracy.scalar.asInstanceOf[Float]}%13.4e")
    //   }
    // }
  }
}
