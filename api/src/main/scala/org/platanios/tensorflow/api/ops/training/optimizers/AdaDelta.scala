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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices}
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
  * @param  learningRate Learning rate to use for the AdaDelta update.
  * @param  decay        Learning rate decay method to use for each update.
  * @param  rho          AdaDelta decay factor.
  * @param  epsilon      AdaDelta constant factor.
  * @param  useLocking   If `true`, the gradient descent updates will be protected by a lock. Otherwise, the behavior is
  *                      undefined, but may exhibit less contention.
  * @param  name         Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
case class AdaDelta private[optimizers](
    learningRate: Double = 0.01, decay: Decay = NoDecay, rho: Double = 0.95, epsilon: Double = 1e-8,
    useLocking: Boolean = false, name: String = "AdaDeltaOptimizer") extends Optimizer {
  private[this] var learningRateTensor: Output = _
  private[this] var rhoTensor         : Output = _
  private[this] var epsilonTensor     : Output = _

  private[this] def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    val lr = Math.cast(learningRateTensor, variable.dataType)
    decay(lr, iteration)
  }

  private[this] def getRho(variable: Variable): Output = {
    if (rhoTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(rhoTensor, variable.dataType)
  }

  private[this] def getEpsilon(variable: Variable): Output = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(epsilonTensor, variable.dataType)
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => {
      zerosSlot("accumulator", v, "Accumulator")
      zerosSlot("accumulator_update", v, "AccumulatorUpdate")
    })
  }

  override def prepare(): Unit = {
    learningRateTensor = Basic.constant(learningRate, name = "LearningRate")
    rhoTensor = Basic.constant(rho, name = "Rho")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("accumulator", variable)
    val accumulatorUpdate = getSlot("accumulator_update", variable)
    AdaDelta.resourceApplyDense(
      variable = variable,
      accumulator = accumulator,
      accumulatorUpdate = accumulatorUpdate,
      learningRate = getLearningRate(variable, iteration),
      rho = getRho(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient,
      useLocking = useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("accumulator", variable)
    val accumulatorUpdate = getSlot("accumulator_update", variable)
    AdaDelta.resourceApplySparse(
      variable = variable,
      accumulator = accumulator,
      accumulatorUpdate = accumulatorUpdate,
      learningRate = getLearningRate(variable, iteration),
      rho = getRho(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient.values,
      indices = gradient.indices,
      useLocking = useLocking)
  }
}

object AdaDelta {
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
    * @param  learningRate      Learning rate to use for the AdaDelta update.
    * @param  rho               AdaDelta decay factor.
    * @param  epsilon           AdaDelta constant factor.
    * @param  gradient          Gradient to apply.
    * @param  useLocking        If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                           undefined, but may exhibit less contention.
    * @param  name              Name for the created op.
    * @return Created op.
    */
  private[AdaDelta] def resourceApplyDense(
      variable: Variable, accumulator: Variable, accumulatorUpdate: Variable, learningRate: Output, rho: Output,
      epsilon: Output, gradient: Output, useLocking: Boolean = false, name: String = "ResourceApplyAdaDelta"): Op = {
    Op.Builder(opType = "ResourceApplyAdadelta", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(accumulatorUpdate.handle)
        .addInput(learningRate)
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
    * @param  learningRate      Learning rate to use for the AdaDelta update.
    * @param  rho               AdaDelta decay factor.
    * @param  epsilon           AdaDelta constant factor.
    * @param  gradient          Gradient to apply.
    * @param  indices           Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  useLocking        If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                           undefined, but may exhibit less contention.
    * @param  name              Name for the created op.
    * @return Created op.
    */
  private[AdaDelta] def resourceApplySparse(
      variable: Variable, accumulator: Variable, accumulatorUpdate: Variable, learningRate: Output, rho: Output,
      epsilon: Output, gradient: Output, indices: Output, useLocking: Boolean = false,
      name: String = "ResourceSparseApplyAdaDelta"): Op = {
    Op.Builder(opType = "ResourceSparseApplyAdadelta", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(accumulatorUpdate.handle)
        .addInput(learningRate)
        .addInput(rho)
        .addInput(epsilon)
        .addInput(gradient)
        .addInput(indices)
        .setAttribute("use_locking", useLocking)
        .build()
  }
}
