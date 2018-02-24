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

package org.platanios.tensorflow.api.ops.training.optimizers.schedules

import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.Variable

/** Scheduling method helper for composing two existing learning rate scheduling methods.
  *
  * The resulting learning rate is the initial learning rate after having applied `schedule2` on it,
  * and then `schedule1`.
  *
  * @param  schedule1 First scheduling method.
  * @param  schedule2 Second scheduling method.
  *
  * @author Emmanouil Antonios Platanios
  */
class ComposedSchedule protected (
    val schedule1: Schedule,
    val schedule2: Schedule
) extends Schedule {
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
  override def apply(value: Output, step: Option[Variable]): Output = {
    schedule1(schedule2(value, step), step)
  }
}

object ComposedSchedule {
  def apply(schedule1: Schedule, schedule2: Schedule): ComposedSchedule = {
    new ComposedSchedule(schedule1, schedule2)
  }
}
