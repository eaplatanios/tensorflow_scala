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

import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.ops.training.optimizers
import org.platanios.tensorflow.api.ops.variables.Variable

/** Trait for implementing optimization learning rate decay methods.
  *
  * When training a model, it is often recommended to lower the learning rate as the training progresses. Decay methods
  * can be used for that purpose. They define ways in which to decay the learning rate.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Decay {
  /** Applies the decay method to `value`, the current iteration in the optimization loop is `step` and returns the
    * result.
    *
    * @param  value Value to decay.
    * @param  step  Option containing current iteration in the optimization loop, if one has been provided.
    * @return Decayed value.
    * @throws IllegalArgumentException If the decay method requires a value for `step` but the provided option is empty.
    */
  @throws[IllegalArgumentException]
  def apply(value: Output, step: Option[Variable]): Output
}

object Decay {
  private[optimizers] trait API {
    type Decay = optimizers.Decay
    type ExponentialDecay = optimizers.ExponentialDecay

    val noDecay: NoDecay.type = optimizers.NoDecay

    def exponentialDecay(
        decayRate: Float, decaySteps: Int, staircase: Boolean = false,
        name: String = "ExponentialDecay"): ExponentialDecay = {
      ExponentialDecay(decayRate = decayRate, decaySteps = decaySteps, staircase = staircase, name = name)
    }
  }
}

/** Dummy decay method representing no decay being used. Useful as a default value for `Decay`-valued function
  * arguments. */
case object NoDecay extends Decay {
  def apply(value: Output, step: Option[Variable]): Output = value
}

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
  */
case class ExponentialDecay private[optimizers](
    decayRate: Float, decaySteps: Int, staircase: Boolean = false, name: String = "ExponentialDecay")
    extends Decay {
  @throws[IllegalArgumentException]
  override def apply(value: Output, step: Option[Variable]): Output = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for exponential decay.")
    Op.createWithNameScope(name, Set(value.op, step.get.op)) {
      val stepValue = Math.cast(step.get.value, value.dataType)
      val rate = Basic.constant(decayRate, value.dataType)
      val steps = Basic.constant(decaySteps, value.dataType)
      val power = Math.divide(stepValue, steps)
      val decay = Math.pow(rate, if (staircase) Math.floor(power) else power)
      Math.multiply(value, decay)
    }
  }
}
