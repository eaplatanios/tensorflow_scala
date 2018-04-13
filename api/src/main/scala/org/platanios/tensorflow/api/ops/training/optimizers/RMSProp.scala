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

import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.{DynamicConstantInitializer, OnesInitializer, Variable}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices, Summary}

/** Optimizer that implements the RMSProp optimization algorithm.
  *
  * The RMSProp update is as follows:
  * {{{
  *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
  *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc + epsilon)
  *   variable -= momAcc
  * }}}
  *
  * This implementation of RMSProp uses plain momentum, not Nesterov momentum.
  *
  * If the centered version is used, the algorithm additionally maintains a moving (discounted) average of the
  * gradients, and uses that average to estimate the variance:
  * {{{
  *   meanGradAcc = decay * rmsAcc + (1 - decay) * gradient
  *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
  *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc - (meanGradAcc ^ 2) + epsilon)
  *   variable -= momAcc
  * }}}
  *
  * @param  learningRate                 Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                      specifies the initial value of the learning rate.
  * @param  decay                        Learning rate decay method to use for each update.
  * @param  rho                          RMSProp decay factor.
  * @param  momentum                     RMSProp momentum factor.
  * @param  epsilon                      RMSProp constant factor.
  * @param  centered                     Boolean value indicating whether or not to use the centered version of RMSProp.
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
class RMSProp protected (
    val learningRate: Double = 0.01,
    val decay: Schedule = FixedSchedule,
    val rho: Double = 0.9,
    val momentum: Double = 0.0,
    val epsilon: Double = 1e-10,
    val centered: Boolean = false,
    override val ignoreDuplicateSparseIndices: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "RMSProp"
) extends Optimizer {
  protected var learningRateTensor: Output = _
  protected var rhoTensor         : Output = _
  protected var momentumTensor    : Output = _
  protected var epsilonTensor     : Output = _

  protected def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  protected def getRho(variable: Variable): Output = {
    if (rhoTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(rhoTensor, variable.dataType)
  }

  protected def getMomentum(variable: Variable): Output = {
    if (momentumTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(momentumTensor, variable.dataType)
  }

  protected def getEpsilon(variable: Variable): Output = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(epsilonTensor, variable.dataType)
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => {
      val rmsInit = if (v.shape.isFullyDefined) OnesInitializer else DynamicConstantInitializer(Basic.onesLike(v))
      getSlot("AccumulatorRMS", v, rmsInit, v.shape, v.dataType, name)
      if (centered)
        zerosSlot("AccumulatorMeanGradient", v, name)
      zerosSlot("AccumulatorMomentum", v, name)
    })
  }

  override def prepare(iteration: Option[Variable]): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    rhoTensor = Basic.constant(rho, name = "Decay")
    momentumTensor = Basic.constant(momentum, name = "Momentum")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulatorMeanGradient = if (centered) getSlot("AccumulatorMeanGradient", variable) else null
    val accumulatorRMS = getSlot("AccumulatorRMS", variable)
    val accumulatorMomentum = getSlot("AccumulatorMomentum", variable)
    RMSProp.resourceApplyDense(
      variable = variable,
      accumulatorMeanGradient = accumulatorMeanGradient,
      accumulatorRMS = accumulatorRMS,
      accumulatorMomentum = accumulatorMomentum,
      learningRate = getLearningRate(variable, iteration),
      decay = getRho(variable),
      momentum = getMomentum(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient,
      useLocking = useLocking)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val accumulatorMeanGradient = if (centered) getSlot("AccumulatorMeanGradient", variable) else null
    val accumulatorRMS = getSlot("AccumulatorRMS", variable)
    val accumulatorMomentum = getSlot("AccumulatorMomentum", variable)
    RMSProp.resourceApplySparse(
      variable = variable,
      accumulatorMeanGradient = accumulatorMeanGradient,
      accumulatorRMS = accumulatorRMS,
      accumulatorMomentum = accumulatorMomentum,
      learningRate = getLearningRate(variable, iteration),
      decay = getRho(variable),
      momentum = getMomentum(variable),
      epsilon = getEpsilon(variable),
      gradient = gradient.values,
      indices = gradient.indices,
      useLocking = useLocking)
  }
}

object RMSProp {
  def apply(
      learningRate: Double = 0.01,
      decay: Schedule = FixedSchedule,
      rho: Double = 0.9,
      momentum: Double = 0.0,
      epsilon: Double = 1e-10,
      centered: Boolean = false,
      ignoreDuplicateSparseIndices: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "RMSProp"
  ): RMSProp = {
    new RMSProp(
      learningRate, decay, rho, momentum, epsilon, centered, ignoreDuplicateSparseIndices, useLocking,
      learningRateSummaryTag, name)
  }

  /** Creates an op that updates `variable` by applying the RMSProp algorithm update to it.
    *
    * The RMSProp update is as follows:
    * {{{
    *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
    *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc + epsilon)
    *   variable -= momAcc
    * }}}
    *
    * This implementation of RMSProp uses plain momentum, not Nesterov momentum.
    *
    * If `accumulatorMeanGradient` is provided, then the centered version is used, which additionally maintains a moving
    * (discounted) average of the gradients, and uses that average to estimate the variance:
    * {{{
    *   meanGradAcc = decay * rmsAcc + (1 - decay) * gradient
    *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
    *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc - (meanGradAcc ^ 2) + epsilon)
    *   variable -= momAcc
    * }}}
    *
    * @param  variable                Variable whose value to update.
    * @param  accumulatorMeanGradient RMSProp mean gradient accumulator variable.
    * @param  accumulatorRMS          RMSProp RMS accumulator variable.
    * @param  accumulatorMomentum     RMSProp momentum accumulator variable.
    * @param  learningRate            Step size to use for the RMSProp update.
    * @param  decay                   RMSProp decay factor.
    * @param  momentum                RMSProp momentum factor.
    * @param  epsilon                 RMSProp constant factor.
    * @param  gradient                Gradient to apply.
    * @param  useLocking              If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                                 undefined, but may exhibit less contention.
    * @param  name                    Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplyDense(
      variable: Variable,
      accumulatorMeanGradient: Variable,
      accumulatorRMS: Variable,
      accumulatorMomentum: Variable,
      learningRate: Output,
      decay: Output,
      momentum: Output,
      epsilon: Output,
      gradient: Output,
      useLocking: Boolean = false,
      name: String = "ResourceApplyRMSProp"
  ): Op = {
    if (accumulatorMeanGradient != null) {
      Op.Builder(opType = "ResourceApplyCenteredRMSProp", name = name)
          .addInput(variable.handle)
          .addInput(accumulatorMeanGradient.handle)
          .addInput(accumulatorRMS.handle)
          .addInput(accumulatorMomentum.handle)
          .addInput(learningRate)
          .addInput(decay)
          .addInput(momentum)
          .addInput(epsilon)
          .addInput(gradient)
          .setAttribute("use_locking", useLocking)
          .build()
    } else {
      Op.Builder(opType = "ResourceApplyRMSProp", name = name)
          .addInput(variable.handle)
          .addInput(accumulatorRMS.handle)
          .addInput(accumulatorMomentum.handle)
          .addInput(learningRate)
          .addInput(decay)
          .addInput(momentum)
          .addInput(epsilon)
          .addInput(gradient)
          .setAttribute("use_locking", useLocking)
          .build()
    }
  }

  /** Creates an op that applies sparse updates to `variable` by applying the RMSProp algorithm update to it.
    *
    * That is for rows that we have a gradient for, the RMSProp update is as follows:
    * {{{
    *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
    *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc + epsilon)
    *   variable -= momAcc
    * }}}
    *
    * This implementation of RMSProp uses plain momentum, not Nesterov momentum.
    *
    * If `accumulatorMeanGradient` is provided, then the centered version is used, which additionally maintains a moving
    * (discounted) average of the gradients, and uses that average to estimate the variance:
    * {{{
    *   meanGradAcc = decay * rmsAcc + (1 - decay) * gradient
    *   rmsAcc = decay * rmsAcc + (1 - decay) * (gradient ^ 2)
    *   momAcc = momentum * momAcc + learningRate * gradient / sqrt(rmsAcc - (meanGradAcc ^ 2) + epsilon)
    *   variable -= momAcc
    * }}}
    *
    * @param  variable                Variable whose value to update.
    * @param  accumulatorMeanGradient RMSProp mean gradient accumulator variable.
    * @param  accumulatorRMS          RMSProp RMS accumulator variable.
    * @param  accumulatorMomentum     RMSProp momentum accumulator variable.
    * @param  learningRate            Step size to use for the RMSProp update.
    * @param  decay                   RMSProp decay factor.
    * @param  momentum                RMSProp momentum factor.
    * @param  epsilon                 RMSProp constant factor.
    * @param  gradient                Gradient to apply.
    * @param  indices                 Vector of indices into the first dimension of `variable` and `accumulator`.
    * @param  useLocking              If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is
    *                                 undefined, but may exhibit less contention.
    * @param  name                    Name for the created op.
    * @return Created op.
    */
  private[optimizers] def resourceApplySparse(
      variable: Variable,
      accumulatorMeanGradient: Variable,
      accumulatorRMS: Variable,
      accumulatorMomentum: Variable,
      learningRate: Output,
      decay: Output,
      momentum: Output,
      epsilon: Output,
      gradient: Output,
      indices: Output,
      useLocking: Boolean = false,
      name: String = "ResourceSparseApplyRMSProp"
  ): Op = {
    if (accumulatorMeanGradient != null) {
      Op.Builder(opType = "ResourceSparseApplyCenteredRMSProp", name = name)
          .addInput(variable.handle)
          .addInput(accumulatorMeanGradient.handle)
          .addInput(accumulatorRMS.handle)
          .addInput(accumulatorMomentum.handle)
          .addInput(learningRate)
          .addInput(decay)
          .addInput(momentum)
          .addInput(epsilon)
          .addInput(gradient)
          .addInput(indices)
          .setAttribute("use_locking", useLocking)
          .build()
    } else {
      Op.Builder(opType = "ResourceSparseApplyRMSProp", name = name)
          .addInput(variable.handle)
          .addInput(accumulatorRMS.handle)
          .addInput(accumulatorMomentum.handle)
          .addInput(learningRate)
          .addInput(decay)
          .addInput(momentum)
          .addInput(epsilon)
          .addInput(gradient)
          .addInput(indices)
          .setAttribute("use_locking", useLocking)
          .build()
    }
  }
}
