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
import org.platanios.tensorflow.api.ops.{Basic, Cast, Op, Output, OutputIndexedSlices, Summary}
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{Schedule, FixedSchedule}
import org.platanios.tensorflow.api.ops.variables.Variable

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
    val decay: Schedule = FixedSchedule,
    val rho: Float = 0.95f,
    val epsilon: Float = 1e-8f,
    override val ignoreDuplicateSparseIndices: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "AdaDelta"
) extends Optimizer {
  protected var learningRateTensor: Output = _
  protected var rhoTensor         : Output = _
  protected var epsilonTensor     : Output = _

  protected def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.cast(variable.dataType).toOutput
  }

  protected def getRho(variable: Variable): Output = {
    if (rhoTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    rhoTensor.cast(variable.dataType).toOutput
  }

  protected def getEpsilon(variable: Variable): Output = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilonTensor.cast(variable.dataType).toOutput
  }

  override def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => {
      zerosSlot("Accumulator", v, name)
      zerosSlot("AccumulatorUpdate", v, name)
    })
  }

  override def prepare(iteration: Option[Variable]): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    rhoTensor = Basic.constant(rho, name = "Rho")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("Accumulator", variable)
    val accumulatorUpdate = getSlot("AccumulatorUpdate", variable)
    AdaDelta.resourceApplyDense(
      variable = variable,
      accumulator = accumulator,
      accumulatorUpdate = accumulatorUpdate,
      stepSize = getLearningRate(variable, iteration),
      rho = getRho(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient,
      useLocking = useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("Accumulator", variable)
    val accumulatorUpdate = getSlot("AccumulatorUpdate", variable)
    AdaDelta.resourceApplySparse(
      variable = variable,
      accumulator = accumulator,
      accumulatorUpdate = accumulatorUpdate,
      stepSize = getLearningRate(variable, iteration),
      rho = getRho(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient.values,
      indices = gradient.indices,
      useLocking = useLocking)
  }
}

object AdaDelta {
  def apply(
      learningRate: Float = 0.01f,
      decay: Schedule = FixedSchedule,
      rho: Float = 0.95f,
      epsilon: Float = 1e-8f,
      ignoreDuplicateSparseIndices: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "AdaDelta"
  ): AdaDelta = {
    new AdaDelta(
      learningRate, decay, rho, epsilon, ignoreDuplicateSparseIndices, useLocking, learningRateSummaryTag, name)
  }

  /** Creates an op that updates `variable` by applying the AdaDelta algorithm update to it.
    *
    * The AdaDelta update is as follows:
    * {{{
    *   accumulator = rho * accumulator + (1 - rho) * gradient
    *   update = sqrt(accumulatorUpdate + epsilon) * rsqrt(accumulator + epsilon) * gradient
    *   accumulatorUpdate = rho * accumulatorUpdate + (1 - rho) * square(update)
    *   variable -= update
    * }}}
    *
    * @param  variable          Variable whose value to update.
    * @param  accumulator       AdaDelta accumulator variable.
    * @param  accumulatorUpdate AdaDelta accumulator update variable.
    * @param  stepSize          Step size to use for the AdaDelta update.
    * @param  rho               AdaDelta decay factor.
    * @param  epsilon           AdaDelta constant factor.
    * @param  gradient          Gradient to apply.
    * @param  useLocking        If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                           undefined, but may exhibit less contention.
    * @param  name              Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplyDense(
      variable: Variable,
      accumulator: Variable,
      accumulatorUpdate: Variable,
      stepSize: Output,
      rho: Output,
      epsilon: Output,
      gradient: Output,
      useLocking: Boolean = false,
      name: String = "ResourceApplyAdaDelta"
  ): Op = {
    Op.Builder(opType = "ResourceApplyAdadelta", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(accumulatorUpdate.handle)
        .addInput(stepSize)
        .addInput(rho)
        .addInput(epsilon)
        .addInput(gradient)
        .setAttribute("use_locking", useLocking)
        .build()
  }

  /** Creates an op that applies sparse updates to `variable` by applying the AdaDelta algorithm update to it.
    *
    * That is for rows that we have a gradient for, the AdaDelta update is as follows:
    * {{{
    *   accumulator = rho * accumulator + (1 - rho) * gradient
    *   update = sqrt(accumulatorUpdate + epsilon) * rsqrt(accumulator + epsilon) * gradient
    *   accumulatorUpdate = rho * accumulatorUpdate + (1 - rho) * square(update)
    *   variable -= update
    * }}}
    *
    * @param  variable          Variable whose value to update.
    * @param  accumulator       AdaDelta accumulator variable.
    * @param  accumulatorUpdate AdaDelta accumulator update variable.
    * @param  stepSize          Step size to use for the AdaDelta update.
    * @param  rho               AdaDelta decay factor.
    * @param  epsilon           AdaDelta constant factor.
    * @param  gradient          Gradient to apply.
    * @param  indices           Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  useLocking        If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                           undefined, but may exhibit less contention.
    * @param  name              Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplySparse(
      variable: Variable,
      accumulator: Variable,
      accumulatorUpdate: Variable,
      stepSize: Output,
      rho: Output,
      epsilon: Output,
      gradient: Output,
      indices: Output,
      useLocking: Boolean = false,
      name: String = "ResourceSparseApplyAdaDelta"
  ): Op = {
    Op.Builder(opType = "ResourceSparseApplyAdadelta", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(accumulatorUpdate.handle)
        .addInput(stepSize)
        .addInput(rho)
        .addInput(epsilon)
        .addInput(gradient)
        .addInput(indices)
        .setAttribute("use_locking", useLocking)
        .build()
  }
}
