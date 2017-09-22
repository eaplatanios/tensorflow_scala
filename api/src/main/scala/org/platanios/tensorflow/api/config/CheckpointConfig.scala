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

package org.platanios.tensorflow.api.config

/** Checkpoint configuration used while training models.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait CheckpointConfig {
  val maxCheckpointsToKeep     : Int
  val keepCheckpointEveryNHours: Int
}

/** Checkpoint configuration for not saving any checkpoints. */
case object NoCheckpoints extends CheckpointConfig {
  override val maxCheckpointsToKeep     : Int = 0
  override val keepCheckpointEveryNHours: Int = 10000
}

/** Checkpoint configuration base class.
  *
  * @param  maxCheckpointsToKeep      Maximum number of recent checkpoint files to keep. As new files are created, older
  *                                   files are deleted. If 0, then all checkpoint files are kept. Defaults to 5 (that
  *                                   is, the 5 most recent checkpoint files are kept).
  * @param  keepCheckpointEveryNHours Number of hours between each checkpoint to be saved. The default value of 10,000
  *                                   hours effectively disables the feature.
  */
private[config] abstract class CheckpointConfigBase private[config](
    override val maxCheckpointsToKeep: Int = 5,
    override val keepCheckpointEveryNHours: Int = 10000
) extends CheckpointConfig {
  require(
    maxCheckpointsToKeep >= 0,
    s"'maxCheckpointsToKeep' (set to $maxCheckpointsToKeep) needs to be a non-negative integer.")
  require(
    keepCheckpointEveryNHours > 0,
    s"'checkpointEveryNHours' (set to $keepCheckpointEveryNHours) needs to be a positive integer.")
}

/** Checkpoint configuration for step-based checkpoints (i.e., checkpoints every `n` steps).
  *
  * @param  steps                     Save checkpoints every this many steps.
  * @param  maxCheckpointsToKeep      Maximum number of recent checkpoint files to keep. As new files are created, older
  *                                   files are deleted. If 0, then all checkpoint files are kept. Defaults to 5 (that
  *                                   is, the 5 most recent checkpoint files are kept).
  * @param  keepCheckpointEveryNHours Save checkpoints every this many hours. The default value of 10,000 hours
  *                                   effectively disables the feature.
  */
case class StepBasedCheckpoints(
    steps: Int = 1000,
    override val maxCheckpointsToKeep: Int = 5,
    override val keepCheckpointEveryNHours: Int = 10000
) extends CheckpointConfigBase(maxCheckpointsToKeep, keepCheckpointEveryNHours) {
  require(steps >= 0, s"'steps' (set to $steps) needs to be a non-negative integer.")
}

/** Checkpoint configuration for time-based checkpoints (i.e., checkpoints every `n` seconds).
  *
  * @param  seconds                   Save checkpoints every this many seconds.
  * @param  maxCheckpointsToKeep      Maximum number of recent checkpoint files to keep. As new files are created, older
  *                                   files are deleted. If 0, then all checkpoint files are kept. Defaults to 5 (that
  *                                   is, the 5 most recent checkpoint files are kept).
  * @param  keepCheckpointEveryNHours Save checkpoints every this many hours. The default value of 10,000 hours
  *                                   effectively disables the feature.
  */
case class TimeBasedCheckpoints(
    seconds: Int = 600,
    override val maxCheckpointsToKeep: Int = 5,
    override val keepCheckpointEveryNHours: Int = 10000
) extends CheckpointConfigBase(maxCheckpointsToKeep, keepCheckpointEveryNHours) {
  require(seconds >= 0, s"'seconds' (set to $seconds) needs to be a non-negative integer.")
}
