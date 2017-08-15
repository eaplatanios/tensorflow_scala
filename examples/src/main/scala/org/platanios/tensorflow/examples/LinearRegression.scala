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

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
object LinearRegression {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))
  private[this] val random = new Random()

  private[this] val weight = random.nextFloat()

  def main(args: Array[String]): Unit = {
    logger.info("Building linear regression model.")
    val inputs = tf.placeholder(tf.FLOAT32, tf.shape(-1, 1))
    val outputs = tf.placeholder(tf.FLOAT32, tf.shape(-1, 1))
    val weights = tf.variable("weights", tf.FLOAT32, tf.shape(1, 1), tf.zerosInitializer)
    val predictions = tf.matmul(inputs, weights)
    val loss = tf.sum(tf.square(predictions - outputs))
    val trainOp = tf.train.adaGrad(1.0).minimize(loss)

    logger.info("Training the linear regression model.")
    val session = tf.session()
    session.run(targets = tf.globalVariablesInitializer())
    for (i <- 0 to 50) {
      val trainBatch = batch(10000)
      val feeds = Map(inputs -> trainBatch._1, outputs -> trainBatch._2)
      val trainLoss = session.run(feeds = feeds, fetches = loss, targets = trainOp)
      if (i % 1 == 0)
        logger.info(s"Train loss at iteration $i = ${trainLoss.scalar} " +
                        s"(weight = ${session.run(fetches = weights.value).scalar})")
    }

    logger.info(s"Trained weight value: ${session.run(fetches = weights.value).scalar}")
    logger.info(s"True weight value: $weight")
  }

  def batch(batchSize: Int): (tf.Tensor, tf.Tensor) = {
    val inputs = ArrayBuffer.empty[Float]
    val outputs = ArrayBuffer.empty[Float]
    var i = 0
    while (i < batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
      i += 1
    }
    (tf.tensor(inputs.map(tf.tensor(_)): _*), tf.tensor(outputs.map(tf.tensor(_)): _*))
  }
}
