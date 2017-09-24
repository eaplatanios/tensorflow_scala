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

import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.learn.{TRAINING, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  trait API {
    val Softmax   : layers.Softmax.type    = layers.Softmax
    val LogSoftmax: layers.LogSoftmax.type = layers.LogSoftmax
    val Dropout   : layers.Dropout.type    = layers.Dropout
  }

  object API extends API
}

case class Softmax private[layers](override val name: String = "Softmax") extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"Softmax"
  override val forward  : Output => Output = ops.NN.softmax(_, name = name)
}

case class LogSoftmax private[layers](override val name: String = "LogSoftmax") extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"LogSoftmax"
  override val forward  : Output => Output = ops.NN.logSoftmax(_, name = name)
}

case class Dropout private[layers](
    keepProbability: Float, noiseShape: Shape = null, seed: Option[Int] = None, name: String = "Dropout")
    extends NetworkLayer[Output, Output] with ModeConditionalNetworkLayer {
  override val layerType: String           = s"Dropout[$keepProbability]"
  override val forward  : Output => Output = input => {
    val noise = if (noiseShape == null) null else noiseShape.toOutput()
    val default: () => Output = () => input
    val applyDropout: () => Output = () => ops.NN.dropout(input, keepProbability, noise, seed, name)
    applyDropout whenIn TRAINING otherwise default
  }
}
