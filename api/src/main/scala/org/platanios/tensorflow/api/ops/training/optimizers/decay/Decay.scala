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

import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.ops.Output

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

  /** Composes the provided `other` decay method with this decay method and returns the resulting decay method. */
  def >>(other: Decay): ComposedDecay = ComposedDecay(other, this)

  /** Composes this decay method with the provided, `other` decay method and returns the resulting decay method. */
  def compose(other: Decay): ComposedDecay = ComposedDecay(this, other)
}

/** Dummy decay method representing no decay being used. Useful as a default value for `Decay`-valued function
  * arguments. */
case object NoDecay extends Decay {
  def apply(value: Output, step: Option[Variable]): Output = value
}
