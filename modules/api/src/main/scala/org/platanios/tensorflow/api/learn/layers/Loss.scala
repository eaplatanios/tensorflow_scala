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

import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Loss[T](override val name: String) extends Layer[T, Output](name)

object Loss {
  private[layers] trait API {
    type Loss[T] = layers.Loss[T]
    type L2Loss = layers.L2Loss
    type SoftmaxCrossEntropy = layers.SoftmaxCrossEntropy
    type SparseSoftmaxCrossEntropy = layers.SparseSoftmaxCrossEntropy
    type SigmoidCrossEntropy = layers.SigmoidCrossEntropy
    type LogPoissonLoss = layers.LogPoissonLoss
    type SequenceLoss = layers.SequenceLoss

    val L2Loss                   : layers.L2Loss.type                    = layers.L2Loss
    val SoftmaxCrossEntropy      : layers.SoftmaxCrossEntropy.type       = layers.SoftmaxCrossEntropy
    val SparseSoftmaxCrossEntropy: layers.SparseSoftmaxCrossEntropy.type = layers.SparseSoftmaxCrossEntropy
    val SigmoidCrossEntropy      : layers.SigmoidCrossEntropy.type       = layers.SigmoidCrossEntropy
    val LogPoissonLoss           : layers.LogPoissonLoss.type            = layers.LogPoissonLoss
    val SequenceLoss             : layers.SequenceLoss.type              = layers.SequenceLoss
  }

  object API extends API
}

case class L2Loss(override val name: String)
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "L2Loss"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.l2Loss(input._1 - input._2, name = name)
  }
}

case class SoftmaxCrossEntropy(override val name: String)
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SoftmaxCrossEntropy"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.softmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SparseSoftmaxCrossEntropy(override val name: String)
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SparseSoftmaxCrossEntropy"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.sparseSoftmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SigmoidCrossEntropy(override val name: String)
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "SigmoidCrossEntropy"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.sigmoidCrossEntropy(input._1, input._2, name = name)
  }
}

case class LogPoissonLoss(override val name: String, computeFullLoss: Boolean = false)
    extends Loss[(Output, Output)](name) {
  override val layerType: String = "LogPoissonLoss"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.logPoissonLoss(input._1, input._2, computeFullLoss, name = name)
  }
}

case class SequenceLoss(
    override val name: String,
    averageAcrossTimeSteps: Boolean = true,
    averageAcrossBatch: Boolean = true,
    weights: Tensor[DataType] = null,
    lossFn: (Output, Output) => Output = ops.NN.sparseSoftmaxCrossEntropy(_, _)
) extends Loss[(Output, Output)](name) {
  override val layerType: String = "SequenceLoss"

  override def forwardWithoutContext(input: (Output, Output))(implicit mode: Mode): Output = {
    ops.NN.sequenceLoss(
      input._1, input._2,
      if (weights == null) null else ops.Basic.constant(weights),
      averageAcrossTimeSteps, averageAcrossBatch, lossFn, name = name)
  }
}
