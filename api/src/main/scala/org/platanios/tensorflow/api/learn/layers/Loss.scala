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

import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Loss[T](override protected val name: String) extends Layer[T, Output](name)

object Loss {
  private[layers] trait API {
    type Loss[T] = layers.Loss[T]
    type L2Loss = layers.L2Loss
    type SoftmaxCrossEntropy = layers.SoftmaxCrossEntropy
    type SparseSoftmaxCrossEntropy = layers.SparseSoftmaxCrossEntropy
    type SigmoidCrossEntropy = layers.SigmoidCrossEntropy

    val L2Loss                   : layers.L2Loss.type                    = layers.L2Loss
    val SoftmaxCrossEntropy      : layers.SoftmaxCrossEntropy.type       = layers.SoftmaxCrossEntropy
    val SparseSoftmaxCrossEntropy: layers.SparseSoftmaxCrossEntropy.type = layers.SparseSoftmaxCrossEntropy
    val SigmoidCrossEntropy      : layers.SigmoidCrossEntropy.type       = layers.SigmoidCrossEntropy
  }

  object API extends API
}

case class L2Loss(override protected val name: String = "L2Loss")
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "L2Loss"

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), Output] = {
    LayerInstance(input, ops.NN.l2Loss(input._1 - input._2, name = uniquifiedName))
  }
}

case class SoftmaxCrossEntropy(override protected val name: String = "SoftmaxCrossEntropy")
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SoftmaxCrossEntropy"

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), Output] = {
    LayerInstance(input, ops.NN.softmaxCrossEntropy(input._1, input._2, name = uniquifiedName))
  }
}

case class SparseSoftmaxCrossEntropy(override protected val name: String = "SparseSoftmaxCrossEntropy")
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SparseSoftmaxCrossEntropy"

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), Output] = {
    LayerInstance(input, ops.NN.sparseSoftmaxCrossEntropy(input._1, input._2, name = uniquifiedName))
  }
}

case class SigmoidCrossEntropy(override protected val name: String = "SigmoidCrossEntropy")
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SigmoidCrossEntropy"

  override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), Output] = {
    LayerInstance(input, ops.NN.sigmoidCrossEntropy(input._1, input._2, name = uniquifiedName))
  }
}
