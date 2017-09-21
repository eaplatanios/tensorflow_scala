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

/**
  *
  * @param  maxEpochs        Number of epochs (i.e., full passes over the data) after which to stop iterating.
  * @param  maxIterations    Number of iterations after which to stop iterating.
  * @param  restartCounting  If `true`, the number of epochs/iterations is counted starting at the current value when
  *                          initializing the iteration. Otherwise, the iteration stops when the epoch/iteration exceeds
  *                          `maxEpochs`/`maxIterations` in value. For example, in that case, if the current epoch is 10
  *                          when the hook is initialized and `maxEpochs` is `100`, the iteration will continue for `90`
  *                          epochs. If `restartCounting` was set to `true`, in that case, it would continue for `100`
  *                          epochs.
  * @param  absLossChangeTol
  * @param  relLossChangeTol
  * @param  maxIterBelowTol
  *
  * @author Emmanouil Antonios Platanios
  */
class TerminationCriteria(
    val maxEpochs: Option[Long] = Some(100L),
    val maxIterations: Option[Long] = Some(10000L),
    val restartCounting: Boolean = true,
    val absLossChangeTol: Option[Double] = Some(1e-3),
    val relLossChangeTol: Option[Double] = Some(1e-3),
    val maxIterBelowTol: Option[Long] = Some(10L))

object TerminationCriteria {
  def apply(
      maxEpochs: Option[Long] = Some(100L),
      maxIterations: Option[Long] = Some(10000L),
      restartCounting: Boolean = true,
      absLossChangeTol: Option[Double] = Some(1e-3),
      relLossChangeTol: Option[Double] = Some(1e-3),
      maxIterBelowTol: Option[Long] = Some(10L)): TerminationCriteria = {
    new TerminationCriteria(
      maxEpochs, maxIterations, restartCounting, absLossChangeTol, relLossChangeTol, maxIterBelowTol)
  }
}
