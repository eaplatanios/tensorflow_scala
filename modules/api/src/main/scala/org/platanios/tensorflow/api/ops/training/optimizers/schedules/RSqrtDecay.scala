/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.types.{TF, IsIntOrLong}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.Variable

/** Reciprocal square root decay method.
  *
  * This method applies a square root decay function to a provided initial learning rate (i.e., `value`). It requires a
  * step value to be provided in it's application function, in order to compute the decayed learning rate. You may
  * simply pass a TensorFlow variable that you increment at each training step.
  *
  * The decayed value is computed as follows:
  * {{{
  *    decayed = value * decayFactor / sqrt(max(step, decayThreshold))
  * }}}
  *
  * @param  decayFactor    Decay factor.
  * @param  decayThreshold Decay threshold.
  * @param  startStep      Step after which to start decaying the learning rate.
  *
  * @author Emmanouil Antonios Platanios
  */
class RSqrtDecay protected (
    val decayFactor: Float = 500.0f,
    val decayThreshold: Float = 1.0f,
    val startStep: Long = 0L,
    val name: String = "RSqrtDecay"
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
  override def apply[I: TF : IsIntOrLong](
      value: Output[Float],
      step: Option[Variable[I]]
  ): Output[Float] = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for square-root decay.")
    Op.nameScope(name) {
      val stepValue = step.get.value.castTo[Float]
      val decayFactorValue = Basic.constant(decayFactor).castTo[Float]
      val decayThresholdValue = Basic.constant(decayThreshold).castTo[Float]
      if (startStep == 0L) {
        decay(value, stepValue, decayFactorValue, decayThresholdValue)
      } else {
        val startStepValue = Basic.constant(startStep).castTo[Float]
        ControlFlow.cond(
          stepValue < startStepValue,
          () => value,
          () => decay(value, stepValue - startStepValue, decayFactorValue, decayThresholdValue))
      }
    }
  }

  private def decay(
      initialValue: Output[Float],
      step: Output[Float],
      decayFactor: Output[Float],
      decayThreshold: Output[Float]
  ): Output[Float] = {
    decayFactor * Math.rsqrt(Math.maximum(step, decayThreshold))
  }
}

object RSqrtDecay {
  def apply(
      decayFactor: Float = 500.0f,
      decayThreshold: Float = 1.0f,
      startStep: Long = 0L,
      name: String = "RSqrtDecay"
  ): RSqrtDecay = {
    new RSqrtDecay(decayFactor, decayThreshold, startStep, name)
  }
}
