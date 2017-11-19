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

import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
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
    type LuongExponentialDecay = optimizers.LuongExponentialDecay
    type WarmUpDecay = optimizers.WarmUpDecay

    val NoDecay              : optimizers.NoDecay.type               = optimizers.NoDecay
    val ExponentialDecay     : optimizers.ExponentialDecay.type      = optimizers.ExponentialDecay
    val LuongExponentialDecay: optimizers.LuongExponentialDecay.type = optimizers.LuongExponentialDecay
    val WarmUpDecay          : optimizers.WarmUpDecay.type           = optimizers.WarmUpDecay
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
  * @param  startStep  Step after which to start decaying the learning rate.
  */
class ExponentialDecay(
    decayRate: Float,
    decaySteps: Int,
    staircase: Boolean = false,
    startStep: Long = 0L,
    name: String = "ExponentialDecay"
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
class LuongExponentialDecay(numTrainSteps: Int, name: String = "LuongExponentialDecay")
    extends ExponentialDecay(0.5f, numTrainSteps / 10, staircase = true, numTrainSteps / 2)

object LuongExponentialDecay {
  def apply(numTrainSteps: Int, name: String = "LuongExponentialDecay"): LuongExponentialDecay = {
    new LuongExponentialDecay(numTrainSteps, name)
  }
}

/** Learning rate decay wrapper that implements a warm-up scheme, similar to the one proposed in
  * [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
  *
  * For the first `warmUpSteps` steps the learning rate is multiplied by:
  * `exp(log(warmUpFactor) / step) ^ (warmUpSteps - step)`.
  *
  * @param  warmUpSteps  Number of warm-up steps.
  * @param  warmUpFactor Warm-up learning rate scaling factor.
  * @param  decay        Learning rate decay method being wrapped.
  */
class WarmUpDecay(warmUpSteps: Int, warmUpFactor: Float = 0.01f, decay: Decay = NoDecay) extends Decay {
  @throws[IllegalArgumentException]
  override def apply(value: Output, step: Option[Variable]): Output = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for exponential decay.")
    val stepValue = Math.cast(step.get.value, value.dataType)
    val warmUpStepsValue = Basic.constant(warmUpSteps, value.dataType)
    val decayLearningRate = decay.apply(value, step)
    ControlFlow.cond(
      stepValue < warmUpStepsValue,
      () => {
        val warmUpFactorValue = Basic.constant(warmUpFactor, value.dataType)
        decayLearningRate * Math.pow(
          Math.exp(Math.log(warmUpFactorValue) / warmUpStepsValue),
          warmUpStepsValue - stepValue)
      },
      () => decayLearningRate)
  }
}

object WarmUpDecay {
  def apply(warmUpSteps: Int, warmUpFactor: Float = 0.01f, decay: Decay = NoDecay): WarmUpDecay = {
    new WarmUpDecay(warmUpSteps, warmUpFactor, decay)
  }
}
