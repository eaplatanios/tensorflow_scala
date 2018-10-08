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

import org.platanios.tensorflow.api.core.types.{TF, IsInt32OrInt64}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.Variable

/** Cycle-linear 10x decay method.
  *
  * This method applies a cycle-linear decay function to a provided initial learning rate (i.e., `value`). It requires a
  * step value to be provided in it's application function, in order to compute the decayed learning rate. You may
  * simply pass a TensorFlow variable that you increment at each training step.
  *
  * The decayed value is computed as follows:
  * {{{
  *    cyclePosition = 1 - abs(((step % (2 * cycleSteps)) - cycleSteps) / cycleSteps)
  *    decayed = value * (0.1 + cyclePosition) * 3
  * }}}
  *
  * @param  cycleSteps Cycle linear decay cycle in terms of number of steps.
  * @param  startStep  Step after which to start decaying the learning rate.
  *
  * @author Emmanouil Antonios Platanios
  */
class CycleLinear10xDecay protected (
    val cycleSteps: Int,
    val startStep: Long = 0L,
    val name: String = "CycleLinear10xDecay"
) extends Schedule[Float] {
  /** Applies the decay method to `value`, the current iteration in the optimization loop is `step` and returns the
    * result.
    *
    * @param  value Value to decay.
    * @param  step  Option containing current iteration in the optimization loop, if one has been provided.
    * @return Decayed value.
    * @throws IllegalArgumentException If the decay method requires a value for `step` but the provided option is empty.
    */
  @throws[IllegalArgumentException]
  override def apply[I: TF : IsInt32OrInt64](
      value: Output[Float],
      step: Option[Variable[I]]
  ): Output[Float] = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for cycle-linear 10x decay.")
    Op.nameScope(name) {
      val stepValue = step.get.value.castTo[Float]
      val cycleStepsValue = Basic.constant(cycleSteps).castTo[Float]
      if (startStep == 0L) {
        decay(value, stepValue, cycleStepsValue)
      } else {
        val startStepValue = Basic.constant(startStep).castTo[Float]
        ControlFlow.cond(
          stepValue < startStepValue,
          () => value,
          () => decay(value, stepValue - startStepValue, cycleStepsValue))
      }
    }
  }

  private def decay(
      initialValue: Output[Float],
      step: Output[Float],
      cycleSteps: Output[Float]
  ): Output[Float] = {
    // Cycle the rate linearly by 10x every `cycleSteps`, up and down.
    val cyclePosition = 1.0f - Math.abs(((step % (2 * cycleSteps)) - cycleSteps).castTo[Float] / cycleSteps)
    (0.1f + cyclePosition) * 3.0f // 10x difference in each cycle (0.3 - 3).
  }
}

object CycleLinear10xDecay {
  def apply(cycleSteps: Int, startStep: Long = 0L, name: String = "CycleLinear10xDecay"): CycleLinear10xDecay = {
    new CycleLinear10xDecay(cycleSteps, startStep, name)
  }
}
