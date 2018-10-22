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

import org.platanios.tensorflow.api.core.types.{Resource, TF, IsIntOrLong, IsNotQuantized}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.{DynamicConstantInitializer, OnesInitializer, Variable}

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
    val learningRate: Float = 0.01f,
    val decay: Schedule[Float] = FixedSchedule[Float](),
    val rho: Float = 0.9f,
    val momentum: Float = 0.0f,
    val epsilon: Float = 1e-10f,
    val centered: Boolean = false,
    override val ignoreDuplicateSparseIndices: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "RMSProp"
) extends Optimizer {
  protected var learningRateTensor: Output[Float] = _
  protected var rhoTensor         : Output[Float] = _
  protected var momentumTensor    : Output[Float] = _
  protected var epsilonTensor     : Output[Float] = _

  protected def getLearningRate[V: TF, I: TF : IsIntOrLong](
      variable: Variable[V],
      iteration: Option[Variable[I]]
  ): Output[V] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.castTo[V].toOutput
  }

  protected def getRho[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (rhoTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    rhoTensor.castTo[V].toOutput
  }

  protected def getMomentum[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (momentumTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    momentumTensor.castTo[V].toOutput
  }

  protected def getEpsilon[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilonTensor.castTo[V].toOutput
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    variables.foreach(v => {
      val evTF = TF.fromDataType(v.dataType)
      val rmsInit = {
        if (v.shape.isFullyDefined)
          OnesInitializer
        else
          DynamicConstantInitializer(Basic.onesLike(v.value))(evTF)
      }
      getSlot("AccumulatorRMS", v, v.dataType, rmsInit, v.shape, name)(evTF, evTF)
      if (centered)
        zerosSlot("AccumulatorMeanGradient", v, name)(evTF)
      zerosSlot("AccumulatorMomentum", v, name)(evTF)
    })
  }

  override def prepare[I: TF : IsIntOrLong](
      iteration: Option[Variable[I]]
  ): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    rhoTensor = Basic.constant(rho, name = "Decay")
    momentumTensor = Basic.constant(momentum, name = "Momentum")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulatorMeanGradient = if (centered) getSlot[T, T]("AccumulatorMeanGradient", variable) else null
    val accumulatorRMS = getSlot[T, T]("AccumulatorRMS", variable)
    val accumulatorMomentum = getSlot[T, T]("AccumulatorMomentum", variable)
    if (accumulatorMeanGradient != null) {
      Op.Builder[(Output[Resource], Output[Resource], Output[Resource], Output[Resource], Output[T], Output[T], Output[T], Output[T], Output[T]), Unit](
        opType = "ResourceApplyCenteredRMSProp",
        name = s"$name/ApplyDense",
        input = (variable.handle,
            accumulatorMeanGradient.handle,
            accumulatorRMS.handle,
            accumulatorMomentum.handle,
            getLearningRate(variable, iteration),
            getRho(variable),
            getMomentum(variable),
            getEpsilon(variable),
            gradient)
      ).setAttribute("use_locking", useLocking)
          .build()
    } else {
      Op.Builder[(Output[Resource], Output[Resource], Output[Resource], Output[T], Output[T], Output[T], Output[T], Output[T]), Unit](
        opType = "ResourceApplyRMSProp",
        name = s"$name/ApplyDense",
        input = (variable.handle,
            accumulatorRMS.handle,
            accumulatorMomentum.handle,
            getLearningRate(variable, iteration),
            getRho(variable),
            getMomentum(variable),
            getEpsilon(variable),
            gradient)
      ).setAttribute("use_locking", useLocking)
          .build()
    }
  }

  override def applySparse[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulatorMeanGradient = if (centered) getSlot[T, T]("AccumulatorMeanGradient", variable) else null
    val accumulatorRMS = getSlot[T, T]("AccumulatorRMS", variable)
    val accumulatorMomentum = getSlot[T, T]("AccumulatorMomentum", variable)
    if (accumulatorMeanGradient != null) {
      Op.Builder[(Output[Resource], Output[Resource], Output[Resource], Output[Resource], Output[T], Output[T], Output[T], Output[T], Output[T], Output[Long]), Unit](
        opType = "ResourceSparseApplyCenteredRMSProp",
        name = s"$name/ApplySparse",
        input = (variable.handle,
            accumulatorMeanGradient.handle,
            accumulatorRMS.handle,
            accumulatorMomentum.handle,
            getLearningRate(variable, iteration),
            getRho(variable),
            getMomentum(variable),
            getEpsilon(variable),
            gradient.values,
            gradient.indices)
      ).setAttribute("use_locking", useLocking)
          .build()
    } else {
      Op.Builder[(Output[Resource], Output[Resource], Output[Resource], Output[T], Output[T], Output[T], Output[T], Output[T], Output[Long]), Unit](
        opType = "ResourceSparseApplyRMSProp",
        name = s"$name/ApplySparse",
        input = (variable.handle,
            accumulatorRMS.handle,
            accumulatorMomentum.handle,
            getLearningRate(variable, iteration),
            getRho(variable),
            getMomentum(variable),
            getEpsilon(variable),
            gradient.values,
            gradient.indices)
      ).setAttribute("use_locking", useLocking)
          .build()
    }
  }
}

object RMSProp {
  def apply(
      learningRate: Float = 0.01f,
      decay: Schedule[Float] = FixedSchedule[Float](),
      rho: Float = 0.9f,
      momentum: Float = 0.0f,
      epsilon: Float = 1e-10f,
      centered: Boolean = false,
      ignoreDuplicateSparseIndices: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "RMSProp"
  ): RMSProp = {
    new RMSProp(
      learningRate, decay, rho, momentum, epsilon, centered,
      ignoreDuplicateSparseIndices, useLocking,
      learningRateSummaryTag, name)
  }
}
