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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types._

/** Optimizer that implements the gradient descent algorithm and includes support for learning rate decay, momentum, and
  * Nesterov acceleration.
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument specifies the
  *                                initial value of the learning rate.
  * @param  decay                  Learning rate decay method to use for each update.
  * @param  momentum               Momentum. Must be `>= 0`.
  * @param  useNesterov            Boolean value indicating whether to use Nesterov acceleration or not. For details,
  *                                refer to [Sutskever et. al., 2013](http://proceedings.mlr.press/v28/sutskever13.pdf).
  * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can be
  *                                monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class GradientDescent protected (
    val learningRate: Float,   // TODO: [TYPES] Should be allowed to also be `Double`.
    val decay: Schedule[Float] = FixedSchedule,
    val momentum: Float = 0.0f,
    val useNesterov: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "GradientDescent"
) extends Optimizer {
  override val ignoreDuplicateSparseIndices: Boolean = true

  protected var learningRateTensor: Output[Float] = _
  protected var momentumTensor    : Output[Float] = _

  protected def getLearningRate[V, I: IsInt32OrInt64](
      variable: Variable[V],
      iteration: Option[Variable[I]]
  ): Output[V] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.castTo(variable.dataType).toOutput
  }

  protected def getMomentum[V](
      variable: Variable[V]
  ): Output[V] = {
    if (momentumTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    momentumTensor.castTo(variable.dataType).toOutput
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    if (momentum > 0.0f)
      variables.foreach(v => zerosSlot("Momentum", v, name))
  }

  override def prepare[I: IsInt32OrInt64](
      iteration: Option[Variable[I]]
  ): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    if (momentum > 0.0f)
      momentumTensor = Basic.constant(momentum, name = "Momentum")
  }

  override def applyDense[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    if (momentum > 0.0f) {
      Op.Builder[(Output[Long], Output[Long], Output[T], Output[T], Output[T]), Unit](
        opType = "ResourceApplyMomentum",
        name = s"$name/ApplyDense",
        input = (variable.handle,
            getSlot("Momentum", variable).handle,
            getLearningRate(variable, iteration),
            gradient,
            getMomentum(variable))
      ).setAttribute("use_locking", useLocking)
          .setAttribute("use_nesterov", useNesterov)
          .build().asUntyped
    } else {
      Op.Builder[(Output[Long], Output[T], Output[T]), Unit](
        opType = "ResourceApplyGradientDescent",
        name = s"$name/ApplyDense",
        input = (variable.handle,
            getLearningRate(variable, iteration),
            gradient)
      ).setAttribute("use_locking", useLocking)
          .build().asUntyped
    }
  }

  override def applySparse[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    if (momentum > 0.0f) {
      Op.Builder[(Output[Long], Output[Long], Output[T], Output[T], Output[Long], Output[T]), Unit](
        opType = "ResourceSparseApplyMomentum",
        name = s"$name/ApplySparse",
        input = (variable.handle,
            getSlot("Momentum", variable).handle,
            getLearningRate(variable, iteration),
            gradient.values,
            gradient.indices,
            getMomentum(variable))
      ).setAttribute("use_locking", useLocking)
          .setAttribute("use_nesterov", useNesterov)
          .build().asUntyped
    } else {
      variable.assignScatterSub(
        gradient.indices,
        gradient.values * getLearningRate(variable, iteration),
        name = s"$name/ApplySparse").op
    }
  }

  override def applySparseDuplicateIndices[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    applySparse(gradient, variable, iteration)
  }
}

object GradientDescent {
  def apply(
      learningRate: Float,
      decay: Schedule[Float] = FixedSchedule,
      momentum: Float = 0.0f,
      useNesterov: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "GradientDescent"
  ): GradientDescent = {
    new GradientDescent(
      learningRate, decay, momentum, useNesterov,
      useLocking, learningRateSummaryTag, name)
  }
}
