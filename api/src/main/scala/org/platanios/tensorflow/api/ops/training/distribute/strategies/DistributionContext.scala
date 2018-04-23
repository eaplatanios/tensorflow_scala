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

package org.platanios.tensorflow.api.ops.training.distribute.strategies

import org.platanios.tensorflow.api.ops.training.distribute

// TODO: [DISTRIBUTE] !!! I'm not sure if the current way of stacking contexts works correctly. What if an in-tower
// context is implicitly available and we also make a cross-tower context implicitly available?

/** Represents a distribution context (e.g., in-tower or cross-tower).
  *
  * Distribution contexts are used to constrain what actions are allowed at every point in the code, and enforce those
  * constraints at compile time, using implicits. More specifically, the current distribution context is always
  * available implicitly and can be checked (e.g., some method require an implicit cross-tower context and will not be
  * able to compile if there current context is an in-tower context).
  *
  * For example, for the following execution steps:
  *
  *   1. Start in the default (single-tower) tower context (i.e., implicit in-tower context is available).
  *   2. Switch to a cross-tower context, when entering a distribution strategy scope (this will make available a
  *      cross-tower context implicit, for the all code within that scope).
  *   3. Switch to a (non-default) tower context inside `forEachTower(fn, ...)` (i.e., the code in `fn` will have an
  *      in-tower implicit context available).
  *   4. If `fn` calls `currentTowerContext->mergeCall(mergeFn, ...)`, then inside `mergeFn`, a cross-tower context
  *      will again be implicitly available.
  *
  * Note that you can also go directly from step 1 to 4 to switch to a cross-tower context for the default distribution
  * strategy. You may also switch from the cross-tower context of 4 to an in-tower context by calling
  * `forEachTower()`, jumping back to step 3.
  *
  * Most distribution API methods may only be executed in cross-tower contexts.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DistributionContext {
  /** Current distribution strategy. */
  val strategy: DistributionStrategy

  def scope[R](block: => R): R
}

/** Represents a cross-tower context (as opposed to an in-tower context).
  *
  * This context typically becomes available during a `distributionStrategy.scope` call. That call also sets up a new
  * variable scope that changes variable creation calls (e.g., to make mirrored variables). This is intended as an outer
  * scope that users enter once, around their model creation and graph definition.
  *
  * @param  strategy Distribution strategy.
  */
case class CrossTowerContext(
    override val strategy: DistributionStrategy
) extends DistributionContext {
  override def scope[R](block: => R): R = {
    implicit val context: CrossTowerContext = this
    block
  }
}

/** Represents an in-tower context (as opposed to a cross-tower context).
  *
  * This context is only present during a `forEachTower()` call (except during a `mergeRun()` call), and in such a scope
  * it will be implicitly available.
  *
  * @param  strategy Distribution strategy.
  * @param  towerID  ID of the tower that is being defined, which is a number in `[0, numTowers - 1]`.
  */
case class InTowerContext(
    override val strategy: DistributionStrategy,
    towerID: Int
) extends DistributionContext {
  override def scope[R](block: => R): R = {
    implicit val context: InTowerContext = this
    block
  }

  /** Returns the device this tower is to be executed on, as a string. */
  def device: String = {
    distribute.currentDevice
  }
}
