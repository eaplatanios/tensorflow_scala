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

/** Exponential decay method.
  *
  * This method applies an exponential decay function to a provided initial learning rate (i.e., `value`). It requires a
  * step value to be provided in it's application function, in order to compute the decayed learning rate. You may
  * simply pass a TensorFlow variable that you increment at each training step.
  *
  * The decayed value is computed as follows:
  * {{{
  *    decayed = value * decayRate ^ (step / decaySteps)
  * }}}
  * where if `staircase = true`, then `(step / decaySteps)` is an integer division and the decayed learning rate follows
  * a staircase function.
  *
  * @param  decayRate  Decay rate.
  * @param  decaySteps Decay steps.
  * @param  staircase  If `true`, the decay will occur at discrete intervals.
  * @param  startStep  Step after which to start decaying the learning rate.
  *
  * @author Emmanouil Antonios Platanios
  */
class ExponentialDecay(
    val decayRate: Float,
    val decaySteps: Int,
    val staircase: Boolean = false,
    val startStep: Long = 0L,
    val name: String = "ExponentialDecay"
) extends Decay {
  @throws[IllegalArgumentException]
  override def apply(value: Output, step: Option[Variable]): Output = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for exponential decay.")
    Op.createWithNameScope(name, Set(value.op, step.get.op)) {
      val rate = Basic.constant(decayRate, value.dataType)
      val steps = Basic.constant(decaySteps, value.dataType)
      val stepValue = Math.cast(step.get.value, value.dataType)
      if (startStep == 0L) {
        decay(value, stepValue, rate, steps, staircase)
      } else {
        val startStepValue = Basic.constant(startStep, value.dataType)
        ControlFlow.cond(
          stepValue < startStepValue,
          () => value,
          () => decay(value, stepValue - startStepValue, rate, steps, staircase))
      }
    }
  }

  private[this] def decay(
      initialValue: Output, step: Output, rate: Output, steps: Output, staircase: Boolean): Output = {
    val power = Math.divide(step, steps)
    val decay = Math.pow(rate, if (staircase) Math.floor(power) else power)
    Math.multiply(initialValue, decay)
  }
}

object ExponentialDecay {
  def apply(
      decayRate: Float, decaySteps: Int, staircase: Boolean = false, startStep: Long = 0L,
      name: String = "ExponentialDecay"
  ): ExponentialDecay = {
    new ExponentialDecay(decayRate, decaySteps, staircase, startStep, name)
  }
}

/** A particular instance of [[ExponentialDecay]] that was used in [Luong (2016)](https://github.com/lmthang/thesis). */
class LuongExponentialDecay(val numTrainSteps: Int, val name: String = "LuongExponentialDecay")
    extends ExponentialDecay(0.5f, numTrainSteps / 10, staircase = true, numTrainSteps / 2)

object LuongExponentialDecay {
  def apply(numTrainSteps: Int, name: String = "LuongExponentialDecay"): LuongExponentialDecay = {
    new LuongExponentialDecay(numTrainSteps, name)
  }
}
