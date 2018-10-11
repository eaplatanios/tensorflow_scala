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

import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Loss[T, L: IsFloat32OrFloat64](
    override val name: String
) extends Layer[T, Output[L]](name)

object Loss {
  private[layers] trait API {
    type Loss[Predictions, L] = layers.Loss[Predictions, L]
    type L2Loss[Predictions, L] = layers.L2Loss[Predictions, L]
    type SoftmaxCrossEntropy[Predictions, L] = layers.SoftmaxCrossEntropy[Predictions, L]
    type SparseSoftmaxCrossEntropy[Predictions, I, L] = layers.SparseSoftmaxCrossEntropy[Predictions, I, L]
    type SigmoidCrossEntropy[Predictions, L] = layers.SigmoidCrossEntropy[Predictions, L]
    type LogPoissonLoss[Predictions, L] = layers.LogPoissonLoss[Predictions, L]
    type SequenceLoss[Predictions, Labels, L] = layers.SequenceLoss[Predictions, Labels, L]

    val L2Loss                   : layers.L2Loss.type                    = layers.L2Loss
    val SoftmaxCrossEntropy      : layers.SoftmaxCrossEntropy.type       = layers.SoftmaxCrossEntropy
    val SparseSoftmaxCrossEntropy: layers.SparseSoftmaxCrossEntropy.type = layers.SparseSoftmaxCrossEntropy
    val SigmoidCrossEntropy      : layers.SigmoidCrossEntropy.type       = layers.SigmoidCrossEntropy
    val LogPoissonLoss           : layers.LogPoissonLoss.type            = layers.LogPoissonLoss
    val SequenceLoss             : layers.SequenceLoss.type              = layers.SequenceLoss
  }

  object API extends API
}

case class L2Loss[Predictions: TF : IsDecimal : IsNotQuantized, L: TF : IsFloat32OrFloat64](
    override val name: String
) extends Loss[(Output[Predictions], Output[Predictions]), L](name) {
  override val layerType: String = "L2Loss"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Predictions])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.l2Loss(input._1 - input._2, name = name).castTo[L]
  }
}

case class SoftmaxCrossEntropy[Predictions: TF : IsDecimal, L: TF : IsFloat32OrFloat64](
    override val name: String
) extends Loss[(Output[Predictions], Output[Predictions]), L](name) {
  override val layerType: String = "SoftmaxCrossEntropy"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Predictions])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.softmaxCrossEntropy(input._1, input._2, name = name).castTo[L]
  }
}

case class SparseSoftmaxCrossEntropy[Predictions: TF : IsDecimal, Labels: TF : IsInt32OrInt64, L: TF : IsFloat32OrFloat64](
    override val name: String
) extends Loss[(Output[Predictions], Output[Labels]), L](name) {
  override val layerType: String = "SparseSoftmaxCrossEntropy"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Labels])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.sparseSoftmaxCrossEntropy(input._1, input._2, name = name).castTo[L]
  }
}

case class SigmoidCrossEntropy[Predictions: TF : IsDecimal, L: TF : IsFloat32OrFloat64](
    override val name: String
) extends Loss[(Output[Predictions], Output[Predictions]), L](name) {
  override val layerType: String = "SigmoidCrossEntropy"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Predictions])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.sigmoidCrossEntropy(input._1, input._2, name = name).castTo[L]
  }
}

case class LogPoissonLoss[Predictions: TF : IsDecimal, L: TF : IsFloat32OrFloat64](
    override val name: String,
    computeFullLoss: Boolean = false
) extends Loss[(Output[Predictions], Output[Predictions]), L](name) {
  override val layerType: String = "LogPoissonLoss"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Predictions])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.logPoissonLoss(input._1, input._2, computeFullLoss, name = name).castTo[L]
  }
}

case class SequenceLoss[Predictions: TF : IsDecimal, Labels: TF, L: TF : IsFloat32OrFloat64](
    override val name: String,
    lossFn: (Output[Predictions], Output[Labels]) => Output[Predictions],
    averageAcrossTimeSteps: Boolean = true,
    averageAcrossBatch: Boolean = true,
    weights: Tensor[Predictions] = null,
) extends Loss[(Output[Predictions], Output[Labels]), L](name) {
  override val layerType: String = "SequenceLoss"

  override def forwardWithoutContext(
      input: (Output[Predictions], Output[Labels])
  )(implicit mode: Mode): Output[L] = {
    ops.NN.sequenceLoss(
      input._1, input._2,
      lossFn = lossFn,
      weights = if (weights == null) null else ops.Basic.constant(weights),
      averageAcrossTimeSteps = averageAcrossTimeSteps,
      averageAcrossBatch = averageAcrossBatch,
      name = name
    ).castTo[L]
  }
}
