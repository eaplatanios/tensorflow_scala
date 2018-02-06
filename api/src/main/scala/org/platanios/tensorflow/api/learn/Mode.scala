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

package org.platanios.tensorflow.api.learn

/** Represents the mode that a model is on, while being used by a learner (e.g., training mode, evaluation mode, or
  * prediction mode).
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Mode {
  val isTraining: Boolean
}

case object TRAINING extends Mode {
  override val isTraining: Boolean = true
}

case object EVALUATION extends Mode {
  override val isTraining: Boolean = false
}

case object INFERENCE extends Mode {
  override val isTraining: Boolean = false
}
