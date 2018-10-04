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
import org.platanios.tensorflow.api.types.{IsInt32OrInt64, IsNotQuantized}

/** Optimizer that implements the AdaDelta optimization algorithm.
  *
  * The AdaDelta update is as follows:
  * {{{
  *   accumulator = rho * accumulator + (1 - rho) * gradient
  *   update = sqrt(accumulatorUpdate + epsilon) * rsqrt(accumulator + epsilon) * gradient
  *   accumulatorUpdate = rho * accumulatorUpdate + (1 - rho) * square(update)
  *   variable -= update
  * }}}
  *
  * For more information on this algorithm, please refer to this [paper](http://arxiv.org/abs/1212.5701)
  * ([PDF](http://arxiv.org/pdf/1212.5701v1.pdf)).
  *
  * @param  learningRate                 Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                      specifies the initial value of the learning rate.
  * @param  decay                        Learning rate decay method to use for each update.
  * @param  rho                          AdaDelta decay factor.
  * @param  epsilon                      AdaDelta constant factor.
  * @param  ignoreDuplicateSparseIndices If `true`, duplicate indices will be ignored in sparse updates (i.e., they will
  *                                      not be uniquified before applying the update).
  * @param  useLocking                   If `true`, the gradient descent updates will be protected by a lock. Otherwise,
  *                                      the behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag       Optional summary tag name to use for the learning rate value. If `null`, no
  *                                      summary is created for the learning rate. Otherwise, a scalar summary is
  *                                      created which can be monitored using TensorBoard.
  * @param  name                         Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class AdaDelta protected (
    val learningRate: Float = 0.01f,
    val decay: Schedule[Float] = FixedSchedule,
    val rho: Float = 0.95f,
    val epsilon: Float = 1e-8f,
    override val ignoreDuplicateSparseIndices: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "AdaDelta"
) extends Optimizer {
  protected var learningRateTensor: Output[Float] = _
  protected var rhoTensor         : Output[Float] = _
  protected var epsilonTensor     : Output[Float] = _

  protected def getLearningRate[V, I: IsInt32OrInt64](
      variable: Variable[V],
      iteration: Option[Variable[I]]
  ): Output[V] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.castTo(variable.dataType).toOutput
  }

  protected def getRho[V](
      variable: Variable[V]
  ): Output[V] = {
    if (rhoTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    rhoTensor.castTo(variable.dataType).toOutput
  }

  protected def getEpsilon[V](
      variable: Variable[V]
  ): Output[V] = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilonTensor.castTo(variable.dataType).toOutput
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    variables.foreach(v => {
      zerosSlot("Accumulator", v, name)
      zerosSlot("AccumulatorUpdate", v, name)
    })
  }

  override def prepare[I: IsInt32OrInt64](
      iteration: Option[Variable[I]]
  ): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    rhoTensor = Basic.constant(rho, name = "Rho")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulator = getSlot("Accumulator", variable)
    val accumulatorUpdate = getSlot("AccumulatorUpdate", variable)
    Op.Builder[(Output[Long], Output[Long], Output[Long], Output[T], Output[T], Output[T], Output[T]), Unit](
      opType = "ResourceApplyAdadelta",
      name = s"$name/ApplyDense",
      input = (variable.handle,
          accumulator.handle,
          accumulatorUpdate.handle,
          getLearningRate(variable, iteration),
          getRho(variable),
          getEpsilon(variable),
          gradient)
    ).setAttribute("use_locking", useLocking)
        .build().asUntyped
  }

  override def applySparse[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulator = getSlot("Accumulator", variable)
    val accumulatorUpdate = getSlot("AccumulatorUpdate", variable)
    Op.Builder[(Output[Long], Output[Long], Output[Long], Output[T], Output[T], Output[T], Output[T], Output[Long]), Unit](
      opType = "ResourceSparseApplyAdadelta",
      name = s"$name/ApplyDense",
      input = (variable.handle,
          accumulator.handle,
          accumulatorUpdate.handle,
          getLearningRate(variable, iteration),
          getRho(variable),
          getEpsilon(variable),
          gradient.values,
          gradient.indices)
    ).setAttribute("use_locking", useLocking)
        .build().asUntyped
  }
}

object AdaDelta {
  def apply(
      learningRate: Float = 0.01f,
      decay: Schedule[Float] = FixedSchedule,
      rho: Float = 0.95f,
      epsilon: Float = 1e-8f,
      ignoreDuplicateSparseIndices: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "AdaDelta"
  ): AdaDelta = {
    new AdaDelta(
      learningRate, decay, rho, epsilon,
      ignoreDuplicateSparseIndices, useLocking,
      learningRateSummaryTag, name)
  }
}
