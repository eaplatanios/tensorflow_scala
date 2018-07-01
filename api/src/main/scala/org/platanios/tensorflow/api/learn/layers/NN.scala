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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, TRAINING, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.NN.{CNNDataFormat, ConvPaddingMode}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  private[layers] trait API {
    type Softmax = layers.Softmax
    type LogSoftmax = layers.LogSoftmax
    type LRN = layers.LRN
    type Dropout = layers.Dropout
    type Conv2D = layers.Conv2D
    type MaxPool = layers.MaxPool

    val Softmax   : layers.Softmax.type    = layers.Softmax
    val LogSoftmax: layers.LogSoftmax.type = layers.LogSoftmax
    val LRN       : layers.LRN.type        = layers.LRN
    val Dropout   : layers.Dropout.type    = layers.Dropout
    val Conv2D    : layers.Conv2D.type     = layers.Conv2D
    val MaxPool   : layers.MaxPool.type    = layers.MaxPool
  }

  object API extends API
}

case class Softmax(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "Softmax"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.softmax(input, name = name)
  }
}

case class LogSoftmax(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "LogSoftmax"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.logSoftmax(input, name = name)
  }
}

case class LRN(
    override val name: String,
    depthRadius: Int = 5,
    bias: Float = 1.0f,
    alpha: Float = 1.0f,
    beta: Float = 0.5f
) extends Layer[Output, Output](name) {
  override val layerType: String = "LRN"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.lrn(input, depthRadius, bias, alpha, beta, name = name)
  }
}

case class Dropout(
    override val name: String,
    keepProbability: Float,
    scaleOutput: Boolean = true,
    noiseShape: Shape = null,
    seed: Option[Int] = None
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Dropout[$keepProbability]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    mode match {
      case TRAINING =>
        val noise = if (noiseShape == null) null else noiseShape.toOutput()
        ops.NN.dropout(input, keepProbability, scaleOutput, noise, seed, name)
      case _ => input
    }
  }
}

case class Conv2D(
    override val name: String,
    filterShape: Shape,
    stride1: Long,
    stride2: Long,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default,
    dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
    useCuDNNOnGPU: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Conv2D[${filterShape.asArray.mkString(",")}]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    val weights = getParameter("Weights", input.dataType, filterShape, weightsInitializer)
    ops.NN.conv2D(input, weights, stride1, stride2, padding, dataFormat, dilations, useCuDNNOnGPU)
  }
}

case class MaxPool(
    override val name: String,
    windowSize: Seq[Long],
    stride1: Long,
    stride2: Long,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default
) extends Layer[Output, Output](name) {
  override val layerType: String = s"MaxPool[${windowSize.mkString(",")}]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.maxPool(input, windowSize, stride1, stride2, padding, dataFormat, name)
  }
}
