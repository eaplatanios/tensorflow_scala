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
import org.platanios.tensorflow.api.learn.{Mode, TRAINING, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.NN.{CNNDataFormat, PaddingMode}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  trait API {
    type Softmax = layers.Softmax
    type LogSoftmax = layers.LogSoftmax
    type Dropout = layers.Dropout
    type Conv2D = layers.Conv2D
    type MaxPool = layers.MaxPool

    val Softmax   : layers.Softmax.type    = layers.Softmax
    val LogSoftmax: layers.LogSoftmax.type = layers.LogSoftmax
    val Dropout   : layers.Dropout.type    = layers.Dropout
    val Conv2D    : layers.Conv2D.type     = layers.Conv2D
    val MaxPool   : layers.MaxPool.type    = layers.MaxPool
  }

  object API extends API
}

case class Softmax(override protected val name: String = "Softmax")
    extends Layer[Output, Output](name) {
  override val layerType: String = "Softmax"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.softmax(input, name = uniquifiedName))
  }
}

case class LogSoftmax(override protected val name: String = "LogSoftmax")
    extends Layer[Output, Output](name) {
  override val layerType: String = "LogSoftmax"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.logSoftmax(input, name = uniquifiedName))
  }
}

case class Dropout(
    keepProbability: Float,
    noiseShape: Shape = null,
    seed: Option[Int] = None,
    override protected val name: String = "Dropout"
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Dropout[$keepProbability]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val output = mode match {
      case TRAINING =>
        val noise = if (noiseShape == null) null else noiseShape.toOutput()
        ops.NN.dropout(input, keepProbability, noise, seed, uniquifiedName)
      case _ => input
    }
    LayerInstance(input, output)
  }
}

case class Conv2D(
    filterShape: Shape,
    stride1: Long,
    stride2: Long,
    padding: PaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default,
    useCuDNNOnGPU: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer(),
    override protected val name: String = "Conv2D"
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Conv2D[${filterShape.asArray.mkString(",")}]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val weights = variable(s"$uniquifiedName/Weights", input.dataType, filterShape, weightsInitializer)
    val output = ops.NN.conv2D(input, weights, stride1, stride2, padding, dataFormat, useCuDNNOnGPU, s"$uniquifiedName/Conv2D")
    LayerInstance(input, output, Set(weights))
  }
}

case class MaxPool(
    windowSize: Seq[Long],
    stride1: Long,
    stride2: Long,
    padding: PaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default,
    override protected val name: String = "MaxPool"
) extends Layer[Output, Output](name) {
  override val layerType: String = s"MaxPool[${windowSize.mkString(",")}]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val output = ops.NN.maxPool(input, windowSize, stride1, stride2, padding, dataFormat, uniquifiedName)
    LayerInstance(input, output)
  }
}
