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

package org.platanios.tensorflow.api.ops.training.optimizers.decay

import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.FLOAT32

/** Learning rate decay wrapper that implements a warm-up scheme, similar to the one proposed in
  * [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
  *
  * Warm-up decay methods compute a factor (possible step-dependent) and multiply the learning rate by that factor,
  * for the first `warmUpSteps` steps of training.
  *
  * @param  warmUpSteps Number of warm-up steps.
  * @param  schedule    Warm-up decay schedule.
  *
  * @author Emmanouil Antonios Platanios
  */
class WarmUpDecay(
    val warmUpSteps: Int,
    val schedule: WarmUpDecay.Schedule,
    val name: String = "WarmUpDecay"
) extends Decay {
  /** Applies the decay method to `value`, the current iteration in the optimization loop is `step` and returns the
    * result.
    *
    * @param  value Value to decay.
    * @param  step  Option containing current iteration in the optimization loop, if one has been provided.
    * @return Decayed value.
    * @throws IllegalArgumentException If the decay method requires a value for `step` but the provided option is empty.
    */
  @throws[IllegalArgumentException]
  override def apply(value: Output, step: Option[Variable]): Output = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for exponential decay.")
    Op.createWithNameScope(name, Set(value.op, step.get.op)) {
      val stepValue = Math.cast(step.get.value, value.dataType)
      val warmUpStepsValue = Basic.constant(warmUpSteps, value.dataType)
      ControlFlow.cond(
        stepValue < warmUpStepsValue,
        () => value * schedule(stepValue, warmUpStepsValue),
        () => value)
    }
  }
}

object WarmUpDecay {
  def apply(warmUpSteps: Int, schedule: WarmUpDecay.Schedule, name: String = "WarmUpDecay"): WarmUpDecay = {
    new WarmUpDecay(warmUpSteps, schedule, name)
  }

  /** Warm-up decay schedule. This is what determines the functional form of the warm-up decay factor. */
  sealed trait Schedule {
    /** Computes the warm-up decay weight for the current step.
      *
      * @param  step        Current step.
      * @param  warmUpSteps Number of warm-up steps.
      * @return Learning rate decay weight for the current step.
      */
    def apply(step: Output, warmUpSteps: Output): Output
  }

  /** Linear warm-up decay schedule. For the first `warmUpSteps` steps the learning rate is multiplied by:
    * `start + ((1.0f - start) / warmUpSteps) * step`.
    *
    * @param  offset Linear decay offset.
    */
  case class Linear(offset: Float = 0.35f) extends Schedule {
    override def apply(step: Output, warmUpSteps: Output): Output = {
      val offsetValue = Basic.constant(offset, FLOAT32)
      offsetValue + ((1.0f - offsetValue) / warmUpSteps) * step
    }
  }

  /** Exponential warm-up decay schedule. For the first `warmUpSteps` steps the learning rate is multiplied by:
    * `exp(log(warmUpFactor) / step) ^ (warmUpSteps - step)`.
    *
    * @param  warmUpFactor Warm-up learning rate scaling factor.
    */
  case class Exponential(warmUpFactor: Float = 0.01f) extends Schedule {
    override def apply(step: Output, warmUpSteps: Output): Output = {
      val warmUpFactorValue = Basic.constant(warmUpFactor, FLOAT32)
      Math.exp(Math.log(warmUpFactorValue) / warmUpSteps) ** (warmUpSteps - step)
    }
  }
}
