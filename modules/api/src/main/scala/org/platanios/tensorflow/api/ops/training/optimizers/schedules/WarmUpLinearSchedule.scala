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

package org.platanios.tensorflow.api.ops.training.optimizers.schedules

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.IsInt32OrInt64

/** Learning rate schedule that implements a warm-up scheme, similar to the one proposed in
  * [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
  *
  * For the first `warmUpSteps` steps the learning rate is multiplied by:
  * `start + ((1.0f - start) / warmUpSteps) * step`.
  *
  * @param  warmUpSteps  Number of warm-up steps.
  * @param  warmUpOffset Linear schedule offset.
  *
  * @author Emmanouil Antonios Platanios
  */
class WarmUpLinearSchedule protected (
    val warmUpSteps: Int,
    val warmUpOffset: Float = 0.35f,
    val name: String = "WarmUpLinearSchedule"
) extends Schedule[Float] {
  /** Applies the scheduling method to `value`, the current iteration in the optimization loop is `step` and returns the
    * result.
    *
    * @param  value Value to change based on this schedule.
    * @param  step  Option containing current iteration in the optimization loop, if one has been provided.
    * @return Potentially modified value.
    * @throws IllegalArgumentException If the scheduling method requires a value for `step` but the provided option is
    *                                  empty.
    */
  @throws[IllegalArgumentException]
  override def apply[V <: Float, I: IsInt32OrInt64](
      value: Output[V],
      step: Option[Variable[I]]
  ): Output[V] = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for warm-up schedule.")
    Op.nameScope(name) {
      val stepValue = step.get.value.toFloat32
      val warmUpStepsValue = Basic.constant(warmUpSteps).toFloat32
      val warmUpOffsetValue = Basic.constant(warmUpOffset).toFloat32
      ControlFlow.cond(
        stepValue < warmUpStepsValue,
        () => value.toFloat32 * schedule(stepValue, warmUpStepsValue, warmUpOffsetValue),
        () => value
      ).cast(value.dataType)
    }
  }

  private def schedule(
      step: Output[Float],
      warmUpSteps: Output[Float],
      offset: Output[Float]
  ): Output[Float] = {
    offset + ((1.0f - offset) / warmUpSteps) * step
  }
}

object WarmUpLinearSchedule {
  def apply(
      warmUpSteps: Int,
      warmUpOffset: Float = 0.35f,
      name: String = "WarmUpLinearSchedule"
  ): WarmUpLinearSchedule = {
    new WarmUpLinearSchedule(warmUpSteps, warmUpOffset, name)
  }
}
