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

import org.platanios.tensorflow.api.core.types.{Resource, TF, IsInt32OrInt64, IsNotQuantized}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
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
  * @param  learningRate                 Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                      specifies the initial value of the learning rate.
  * @param  decay                        Learning rate decay method to use for each update.
  * @param  epsilon                      Initial value to use for the accumulator (i.e., to avoid dividing by zero, or a
  *                                      very small value).
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
class AdaGrad protected (
    val learningRate: Float = 0.01f,
    val decay: Schedule[Float] = FixedSchedule[Float](),
    val epsilon: Float = 1e-8f,
    override val ignoreDuplicateSparseIndices: Boolean = false,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "AdaGrad"
) extends Optimizer {
  protected var learningRateTensor: Output[Float] = _

  protected def getLearningRate[V: TF, I: TF : IsInt32OrInt64](
      variable: Variable[V],
      iteration: Option[Variable[I]]
  ): Output[V] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.castTo[V].toOutput
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    variables.foreach(v => {
      val initializer = ConstantInitializer(epsilon)
      val evTF = TF.fromDataType(v.dataType)
      getSlot("Accumulator", v, v.dataType, initializer, v.shape, name)(evTF, evTF)
    })
  }

  override def prepare[I: TF : IsInt32OrInt64](
      iteration: Option[Variable[I]]
  ): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
  }

  override def applyDense[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulator = getSlot[T, T]("Accumulator", variable)
    Op.Builder[(Output[Resource], Output[Resource], Output[T], Output[T]), Unit](
      opType = "ResourceApplyAdagrad",
      name = s"$name/ApplyDense",
      input = (variable.handle,
          accumulator.handle,
          getLearningRate(variable, iteration),
          gradient)
    ).setAttribute("use_locking", useLocking)
        .build()
  }

  override def applySparse[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    val accumulator = getSlot[T, T]("Accumulator", variable)
    Op.Builder[(Output[Resource], Output[Resource], Output[T], Output[T], Output[Long]), Unit](
      opType = "ResourceSparseApplyAdagrad",
      name = s"$name/ApplySparse",
      input = (variable.handle,
          accumulator.handle,
          getLearningRate(variable, iteration),
          gradient.values,
          gradient.indices)
    ).setAttribute("use_locking", useLocking)
        .build()
  }
}

object AdaGrad {
  def apply(
      learningRate: Float = 0.01f,
      decay: Schedule[Float] = FixedSchedule[Float](),
      epsilon: Float = 1e-8f,
      ignoreDuplicateSparseIndices: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "AdaGrad"
  ): AdaGrad = {
    new AdaGrad(
      learningRate, decay, epsilon, ignoreDuplicateSparseIndices,
      useLocking, learningRateSummaryTag, name)
  }
}
