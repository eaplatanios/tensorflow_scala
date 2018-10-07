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

import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.{IsInt32OrInt64, TF}

/** Dummy scheduling method representing no schedule being used. Useful as a default value for `Schedule`-valued
  * function arguments.
  *
  * @author Emmanouil Antonios Platanios
  */
case class FixedSchedule[-T: TF]() extends Schedule[T] {
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
  override def apply[V <: T : TF, I: TF : IsInt32OrInt64](
      value: Output[V],
      step: Option[Variable[I]]
  ): Output[V] = {
    value
  }
}
