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

import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices, Summary}
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{Schedule, FixedSchedule}
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, Variable}

/** Optimizer that implements the AdaGrad optimization algorithm.
  *
  * The AdaGrad update is as follows:
  * {{{
  *   accumulator += gradient * gradient
  *   variable -= stepSize * gradient * (1 / sqrt(accumulator))
  * }}}
  *
  * For more information on this algorithm, please refer to this
  * [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument specifies the
  *                                initial value of the learning rate.
  * @param  decay                  Learning rate decay method to use for each update.
  * @param  epsilon                Initial value to use for the accumulator (i.e., to avoid dividing by zero, or a very
  *                                small value).
  * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can be
  *                                monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class AdaGrad protected (
    val learningRate: Double = 0.01,
    val decay: Schedule = FixedSchedule,
    val epsilon: Double = 1e-8,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "AdaGrad"
) extends Optimizer {
  protected var learningRateTensor: Output = _

  protected def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => {
      val initializer = ConstantInitializer(epsilon)
      getSlot("Accumulator", v, initializer, v.shape, v.dataType, name)
    })
  }

  override def prepare(iteration: Option[Variable]): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("Accumulator", variable)
    AdaGrad.resourceApplyDense(variable, accumulator, getLearningRate(variable, iteration), gradient, useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulator = getSlot("Accumulator", variable)
    AdaGrad.resourceApplySparse(
      variable, accumulator, getLearningRate(variable, iteration), gradient.values, gradient.indices, useLocking)
  }
}

object AdaGrad {
  def apply(
      learningRate: Double = 0.01,
      decay: Schedule = FixedSchedule,
      epsilon: Double = 1e-8,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "AdaGrad"
  ): AdaGrad = {
    new AdaGrad(learningRate, decay, epsilon, useLocking, learningRateSummaryTag, name)
  }

  /** Creates an op that updates `variable` by applying the AdaGrad algorithm update to it.
    *
    * The AdaGrad update is as follows:
    * {{{
    *   accumulator += gradient * gradient
    *   variable -= stepSize * gradient * (1 / sqrt(accumulator))
    * }}}
    *
    * @param  variable    Variable whose value to update.
    * @param  accumulator AdaGrad accumulator variable.
    * @param  stepSize    Step size to use for the AdaGrad update.
    * @param  gradient    Gradient to apply.
    * @param  useLocking  If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                     but may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplyDense(
      variable: Variable,
      accumulator: Variable,
      stepSize: Output,
      gradient: Output,
      useLocking: Boolean = false,
      name: String = "ResourceApplyAdaGrad"
  ): Op = {
    Op.Builder(opType = "ResourceApplyAdagrad", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .setAttribute("use_locking", useLocking)
        .build()
  }

  /** Creates an op that applies sparse updates to `variable` by applying the AdaGrad algorithm update to it.
    *
    * That is for rows that we have a gradient for, the AdaGrad update is as follows:
    * {{{
    *   accumulator += gradient * gradient
    *   variable -= stepSize * gradient * (1 / sqrt(accumulator))
    * }}}
    *
    * @param  variable    Variable whose value to update.
    * @param  accumulator AdaGrad accumulator variable.
    * @param  stepSize    Step size to use for the AdaGrad update.
    * @param  gradient    Gradient to apply.
    * @param  indices     Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  useLocking  If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                     but may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplySparse(
      variable: Variable,
      accumulator: Variable,
      stepSize: Output,
      gradient: Output,
      indices: Output,
      useLocking: Boolean = false,
      name: String = "ResourceSparseApplyAdaGrad"
  ): Op = {
    Op.Builder(opType = "ResourceSparseApplyAdagrad", name = name)
        .addInput(variable.handle)
        .addInput(accumulator.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .addInput(indices)
        .setAttribute("use_locking", useLocking)
        .build()
  }
}
