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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.{Shape, tf}
import org.platanios.tensorflow.api.learn.{TrainingMode, layers}

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  trait API {
    type Softmax = layers.Softmax
    type LogSoftmax = layers.LogSoftmax
    type Dropout = layers.Dropout

    def softmax(name: String = "Softmax"): Softmax = Softmax(name = name)

    def logSoftmax(name: String = "LogSoftmax"): LogSoftmax = LogSoftmax(name = name)

    def dropout(
        keepProbability: Float, noiseShape: Shape = null, seed: Option[Int] = None,
        name: String = "Dropout"): Dropout = {
      Dropout(keepProbability = keepProbability, noiseShape = noiseShape, seed = seed, name = name)
    }
  }

  object API extends API
}

case class Softmax private[layers](override val name: String = "Softmax") extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"Softmax"
  override val forward  : tf.Output => tf.Output = tf.softmax(_, name = name)
}

case class LogSoftmax private[layers](override val name: String = "LogSoftmax") extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"LogSoftmax"
  override val forward  : tf.Output => tf.Output = tf.logSoftmax(_, name = name)
}

case class Dropout private[layers](
    keepProbability: Float, noiseShape: Shape = null, seed: Option[Int] = None, name: String = "Dropout")
    extends NetworkLayer[tf.Output, tf.Output] with ModeConditionalNetworkLayer {
  override val layerType: String                 = s"Dropout[$keepProbability]"
  override val forward  : tf.Output => tf.Output = input => {
    val noise = if (noiseShape == null) null else noiseShape.toOutput
    val default: () => tf.Output = () => input
    val applyDropout: () => tf.Output = () => tf.dropout(input, keepProbability, noise, seed, name)
    applyDropout whenIn TrainingMode otherwise default
  }
}
