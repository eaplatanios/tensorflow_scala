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

/** Criteria used to some the training process iteration.
  *
  * @param  maxEpochs        Number of epochs (i.e., full passes over the data) after which to stop iterating.
  * @param  maxSteps         Number of steps after which to stop iterating.
  * @param  restartCounting  If `true`, the number of epochs/steps is counted starting at the current value when
  *                          initializing the iteration. Otherwise, the iteration stops when the epoch/step exceeds
  *                          `maxEpochs`/`maxSteps` in value. For example, in that case, if the current epoch is 10
  *                          when the hook is initialized and `maxEpochs` is `100`, the iteration will continue for `90`
  *                          epochs. If `restartCounting` was set to `true`, in that case, it would continue for `100`
  *                          epochs.
  * @param  absLossChangeTol Loss value absolute change tolerance. If the absolute value of the loss difference between
  *                          the current iteration and the previous one drops below this tolerance value, then the
  *                          number of steps below loss-value tolerance is increased by one. Once that number exceeds
  *                          `maxStepsBelowTol`, the training process stops iterating.
  * @param  relLossChangeTol Loss value relative change tolerance. If the absolute value of the loss difference between
  *                          the current iteration and the previous one divided by the last step loss value, drops below
  *                          this tolerance value, then the number of steps below loss-value tolerance is increased by
  *                          one. Once that number exceeds `maxStepsBelowTol`, the training process stops iterating.
  * @param  maxStepBelowTol  Maximum number of steps to allow before stopping, with the loss value being below absolute
  *                          or relative tolerance. In cases when using some form of stochastic optimization (e.g.,
  *                          stochastic gradient descent), it is useful to set this value to some number larger than 1,
  *                          to avoid stopping due to some noisy gradient that didn't result in the loss value changing
  *                          by much.
  *
  * @author Emmanouil Antonios Platanios
  */
class TerminationCriteria(
    val maxEpochs: Option[Long] = Some(100L),
    val maxSteps: Option[Long] = Some(10000L),
    val restartCounting: Boolean = true,
    val absLossChangeTol: Option[Double] = Some(1e-3),
    val relLossChangeTol: Option[Double] = Some(1e-3),
    val maxStepBelowTol: Option[Long] = Some(10L)) {
  require(maxEpochs.getOrElse(0L) >= 0, "'maxEpochs' needs to be a non-negative number.")
  require(maxSteps.getOrElse(0L) >= 0, "'maxSteps' needs to be a non-negative number.")
  require(absLossChangeTol.getOrElse(0.0) >= 0, "'absLossChangeTol' needs to be a non-negative number.")
  require(absLossChangeTol.getOrElse(0.0) >= 0, "'absLossChangeTol' needs to be a non-negative number.")
  require(maxStepBelowTol.getOrElse(0L) >= 0, "'maxStepBelowTol' needs to be a non-negative number.")

  private[learn] val needEpoch: Boolean = maxEpochs.isDefined
  private[learn] val needStep : Boolean = maxSteps.isDefined || maxStepBelowTol.isDefined
  private[learn] val needLoss : Boolean = absLossChangeTol.isDefined || relLossChangeTol.isDefined
}

object TerminationCriteria {
  def apply(
      maxEpochs: Option[Long] = Some(100L),
      maxSteps: Option[Long] = Some(10000L),
      restartCounting: Boolean = true,
      absLossChangeTol: Option[Double] = Some(1e-3),
      relLossChangeTol: Option[Double] = Some(1e-3),
      maxStepBelowTol: Option[Long] = Some(10L)): TerminationCriteria = {
    new TerminationCriteria(
      maxEpochs, maxSteps, restartCounting, absLossChangeTol, relLossChangeTol, maxStepBelowTol)
  }
}
