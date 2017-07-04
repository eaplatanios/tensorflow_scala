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
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, Variable}
import org.platanios.tensorflow.api.types.DataType

/** Optimizer that implements the AdaGrad optimization algorithm.
  *
  * For more information on this algorithm, please refer to this
  * [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  *
  * @author Emmanouil Antonios Platanios
  */
case class AdaGrad(
    learningRate: Double = 1.0, initialAccumulatorValue: Double = 0.1, useLocking: Boolean = false,
    name: String = "AdaGradOptimizer") extends Optimizer {
  private[this] var learningRateTensor: Output[DataType] = _

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => {
      val initializer = ConstantInitializer(initialAccumulatorValue)
      getSlot("accumulator", v, initializer, v.shape, v.dataType, s"${this.name}/Accumulator")
    })
  }

  private[this] def getLearningRate(variable: Variable): Output[DataType] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  override def prepare(): Unit = {
    learningRateTensor = Basic.constant(learningRate, name = "LearningRate")()
  }

  override def applyDense(gradient: Output[DataType], variable: Variable): Op = {
    val accumulator = getSlot("accumulator", variable)
    AdaGrad.resourceApplyDense(variable, accumulator, getLearningRate(variable), gradient, useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices[DataType], variable: Variable): Op = {
    val accumulator = getSlot("accumulator", variable)
    AdaGrad.resourceApplySparse(
      variable, accumulator, getLearningRate(variable), gradient.values, gradient.indices, useLocking)
  }
}

object AdaGrad {
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
    * @param  stepSize    Step size to use for the gradient descent update.
    * @param  gradient    Gradient to apply.
    * @param  useLocking  If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                     but may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[AdaGrad] def resourceApplyDense(
      variable: Variable, accumulator: Variable, stepSize: Output[DataType], gradient: Output[DataType], useLocking: Boolean = false,
      name: String = "ResourceApplyAdaGrad"): Op = {
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
    * @param  stepSize    Step size to use for the gradient descent update.
    * @param  gradient    Gradient to apply.
    * @param  indices     Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  useLocking  If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                     but may exhibit less contention.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[AdaGrad] def resourceApplySparse(
      variable: Variable, accumulator: Variable, stepSize: Output[DataType], gradient: Output[DataType], indices: Output[DataType],
      useLocking: Boolean = false, name: String = "ResourceSparseApplyAdaGrad"): Op = {
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
