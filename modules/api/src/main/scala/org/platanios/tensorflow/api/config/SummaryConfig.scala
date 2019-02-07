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

package org.platanios.tensorflow.api.config

/** Summary configuration used while training models.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait SummaryConfig

/** Summary configuration for not saving any summaries. */
case object NoSummaries extends SummaryConfig

/** Summary configuration for step-based summaries (i.e., summaries every `n` steps).
  *
  * @param  steps Save summaries every this many steps.
  *
  */
case class StepBasedSummaries(steps: Int = 1000) extends SummaryConfig {
  require(steps >= 0, s"'steps' (set to $steps) needs to be a non-negative integer.")
}

/** Summary configuration for time-based summaries (i.e., summaries every `n` seconds).
  *
  * @param  seconds Save summaries every this many seconds.
  *
  */
case class TimeBasedSummaries(seconds: Int = 600) extends SummaryConfig {
  require(seconds >= 0, s"'seconds' (set to $seconds) needs to be a non-negative integer.")
}
