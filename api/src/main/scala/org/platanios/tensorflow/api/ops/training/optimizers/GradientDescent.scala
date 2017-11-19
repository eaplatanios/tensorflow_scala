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

import org.platanios.tensorflow.api.ops.training.optimizers.decay.{Decay, NoDecay}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices, Summary}
import org.platanios.tensorflow.api.ops.variables.Variable

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
case class GradientDescent(
    learningRate: Double, decay: Decay = NoDecay, momentum: Double = 0.0, useNesterov: Boolean = false,
    useLocking: Boolean = false, learningRateSummaryTag: String = null,
    name: String = "GradientDescent"
) extends Optimizer {
  private[this] var learningRateTensor: Output = _
  private[this] var momentumTensor    : Output = _

  private[this] def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    var lr = Math.cast(learningRateTensor, variable.dataType)
    lr = decay(lr, iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, lr)
    lr
  }

  private[this] def getMomentum(variable: Variable): Output = {
    if (momentumTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(momentumTensor, variable.dataType)
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    if (momentum > 0.0f)
      variables.foreach(v => zerosSlot("momentum", v, "Momentum"))
  }

  override def prepare(): Unit = {
    learningRateTensor = Basic.constant(learningRate, name = "LearningRate")
    if (momentum > 0.0f)
      momentumTensor = Basic.constant(momentum, name = "Momentum")
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    if (momentum > 0.0f)
      GradientDescent.resourceApplyMomentumDense(
        variable = variable,
        accumulator = getSlot("momentum", variable),
        stepSize = getLearningRate(variable, iteration),
        gradient = gradient,
        momentum = getMomentum(variable),
        useLocking = useLocking,
        useNesterov = useNesterov)
    else
      GradientDescent.resourceApplyDense(variable, getLearningRate(variable, iteration), gradient, useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    if (momentum > 0.0f)
      GradientDescent.resourceApplyMomentumSparse(
        variable = variable,
        accumulator = getSlot("momentum", variable),
        stepSize = getLearningRate(variable, iteration),
        gradient = gradient.values,
        indices = gradient.indices,
        momentum = getMomentum(variable),
        useLocking = useLocking,
        useNesterov = useNesterov)
    else
      variable.assignScatterSub(gradient.indices, gradient.values * getLearningRate(variable, iteration)).op
  }

  override def applySparseDuplicateIndices(
      gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    applySparse(gradient, variable, iteration)
  }
}

private[api] object GradientDescent {
  /** Creates an op that updates the value of `variable` by subtracting `stepSize * gradient` from it.
    *
    * @param  variable   Variable whose value to update.
    * @param  stepSize   Step size to use for the gradient descent update.
    * @param  gradient   Gradient to apply.
    * @param  useLocking If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                    but may exhibit less contention.
    * @param  name       Name for the created op.
    * @return Created op.
    */
  private[GradientDescent] def resourceApplyDense(
      variable: Variable, stepSize: Output, gradient: Output, useLocking: Boolean = false,
      name: String = "ResourceApplyGradientDescent"): Op = {
    Op.Builder(opType = "ResourceApplyGradientDescent", name = name)
        .addInput(variable.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .setAttribute("use_locking", useLocking)
        .build()
  }

  /** Creates an op that applies updates to the value of `variable` according to the momentum scheme.
    *
    * If `useNesterov = false`, the op computes:
    * {{{
    *   accumulator = momentum * accumulator + gradient
    *   variable -= stepSize * accumulator
    * }}}
    * Otherwise, if `useNesterov = true`, the op uses Nesterov acceleration as described in
    * [Sutskever et. al., 2013](http://proceedings.mlr.press/v28/sutskever13.pdf).
    *
    * @param  variable    Variable whose value to update.
    * @param  accumulator Momentum accumulator variable.
    * @param  stepSize    Step size to use for the gradient descent update.
    * @param  gradient    Gradient to apply.
    * @param  momentum    Momentum value to use.
    * @param  useNesterov If `true`, Nesterov acceleration is used for the update.
    * @param  useLocking  If `true`, the updates will be protected by a lock. Otherwise, the behavior is undefined, but
    *                     may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[GradientDescent] def resourceApplyMomentumDense(
      variable: Variable, accumulator: Variable, stepSize: Output, gradient: Output, momentum: Output,
      useNesterov: Boolean = false, useLocking: Boolean = false, name: String = "ResourceApplyMomentum"): Op = {
    Op.Builder(opType = "ResourceApplyMomentum", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .addInput(momentum)
        .setAttribute("use_locking", useLocking)
        .setAttribute("use_nesterov", useNesterov)
        .build()
  }

  /** Creates an op that applies sparse updates to the value of `variable` according to the momentum scheme.
    *
    * If `useNesterov = false`, for rows that we have a gradient for, the op computes:
    * {{{
    *   accumulator = momentum * accumulator + gradient
    *   variable -= stepSize * accumulator
    * }}}
    * Otherwise, if `useNesterov = true`, the op uses Nesterov acceleration as described in
    * [Sutskever et. al., 2013](http://proceedings.mlr.press/v28/sutskever13.pdf).
    *
    * @param  variable    Variable whose value to update.
    * @param  accumulator Momentum accumulator variable.
    * @param  stepSize    Step size to use for the gradient descent update.
    * @param  gradient    Gradient to apply.
    * @param  indices     Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  momentum    Momentum value to use.
    * @param  useNesterov If `true`, the tensor
    * @param  useLocking  If `true`, the updates will be protected by a lock. Otherwise, the behavior is undefined, but
    *                     may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[GradientDescent] def resourceApplyMomentumSparse(
      variable: Variable, accumulator: Variable, stepSize: Output, gradient: Output, indices: Output, momentum: Output,
      useNesterov: Boolean = false, useLocking: Boolean = false, name: String = "ResourceSparseApplyMomentum"): Op = {
    Op.Builder(opType = "ResourceSparseApplyMomentum", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .addInput(indices)
        .addInput(momentum)
        .setAttribute("use_locking", useLocking)
        .setAttribute("use_nesterov", useNesterov)
        .build()
  }
}
