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

package org.platanios.tensorflow.api.ops.training.distribute

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, OutputLike, graphConstructionScope}
import org.platanios.tensorflow.api.ops.training.distribute.strategies.{CrossTowerContext, DistributionContext, DistributionStrategy}
import org.platanios.tensorflow.api.ops.training.distribute.values.{DistributedValue, MirroredValue}

/**
  * @author Emmanouil Antonios Platanios
  */
trait API {
  /** Returns the current device if in a `distributionStrategy.update()` call. */
  def currentUpdateDevice: Option[String] = {
    updateDevice.value
  }

  /** Returns the current device. */
  def currentDevice: String = {
    graphConstructionScope.value.deviceFunction(OpSpecification("", "", graphConstructionScope.value.device))
  }

  /** Returns the current distribution strategy. */
  def currentStrategy(implicit context: DistributionContext): DistributionStrategy = {
    // TODO: [DISTRIBUTE] Is this method really needed?
    context.strategy
  }

  /** Executes `block` within a scope where new variables will not be mirrored.
    *
    * There will still be one component variable per tower, but there is no requirement that they stay in sync. Instead,
    * when saving them or calling `fetch()`, we use the value that results when calling `reduce()` on all the towers'
    * variables. Note that tower-local implies not trainable. Instead, it is expected that each tower will directly
    * update (e.g., using `assignAdd()`) its local variable instance but only the aggregated value (accessible using
    * `fetch()`) will be exported from the model. When it is acceptable to only aggregate on export, we greatly reduce
    * communication overhead by using tower-local variables.
    *
    * Note that all component variables will be initialized to the same value, using the initialization expression from
    * the first tower. The values will match even if the initialization expression uses random numbers.
    *
    * @param  reduction Reduction method used to get the value to save when creating checkpoints.
    * @param  block     Code block to execute in this scope.
    * @return Value returned by `block`.
    */
  def towerLocalVariableScope[R](reduction: Reduction)(block: => R)(implicit context: DistributionContext): R = {
    context.strategy.towerLocalVariableScope(reduction)(block)
  }

  /** Executes `block` within a scope that controls which devices variables will be created on.
    *
    * No operations should be added to the graph inside this scope; it should only be used when creating variables (some
    * implementations work by changing variable creation and others work by using a `colocateWith` scope). This may only
    * be used inside `DistributionStrategy.scope`.
    *
    * For example:
    * {{{
    *   distributionStrategy.scope {
    *     val variable1 = tf.variable(...)
    *     distributionStrategy.colocateVariablesWith(Set(variable1.op)) {
    *       // `variable2` and `variable3` will be created on the same device(s) as `variable1`.
    *       val variable2 = tf.variable(...)
    *       val variable3 = tf.variable(...)
    *     }
    *
    *     def fn(v1: Variable, v2: Variable, v3: Variable): Unit = {
    *       // Operates on `v1` from `variable1`, `v2` from `variable2`, and `v3` from `variable3`.
    *     }
    *
    *     // `fn` runs on every device `v1` is on, and `v2` and `v3` will be there too.
    *     distributionStrategy.update(variable1, fn, variable2, variable3)
    *   }
    * }}}
    *
    * @param  colocationOps Variables created in `block` will be on the same set of devices as these ops.
    * @param  block         Code block to execute in this scope.
    * @return Value returned by `block`.
    */
  def colocateVariablesWith[R](colocationOps: Set[Op])(block: => R)(implicit context: DistributionContext): R = {
    context.strategy.colocateVariablesWith(colocationOps)(block)
  }

  /** Mirrors `value` to all worker devices.
    *
    * @param  value   Value to broadcast.
    * @param  devices Destination devices.
    * @return Mirrored value.
    */
  def broadcast[O <: OutputLike](
      value: O,
      devices: Seq[DeviceSpecification] = Seq.empty
  )(implicit context: CrossTowerContext): MirroredValue[O] = {
    context.strategy.broadcast(value, devices)
  }

  /** Runs `fn` once per tower.
    *
    * `fn` may call `tf.currentTowerContext` to access fields and methods such as `towerID` and `mergeCall()`.
    * `mergeCall()` is used to communicate between the towers and re-enter the cross-tower context. All towers pause
    * their execution having encountered a `mergeCall()` call. After that the `mergeFn`-function is executed. Its
    * results are then unwrapped and given back to each tower call. After that execution resumes until `fn` is complete
    * or another `mergeCall()` is encountered.
    *
    * For example:
    * {{{
    *   // Called once in "cross-tower" context.
    *   def mergeFn(distributionStrategy: DistributionStrategy, threePlusTowerID: Int): tf.Output = {
    *     // Sum the values across towers.
    *     tf.addN(distribution.unwrap(threePlusTowerID))
    *   }
    *
    *   // Called once per tower in `distributionStrategy`, in a "tower" context.
    *   def fn(three: Int): Output = {
    *     val towerContext = tf.currentTowerContext
    *     val v = three + towerContext.towerID
    *     // Computes the sum of the `v` values across all towers.
    *     val s = towerContext.mergeCall(mergeFn(_, v))
    *     s + v
    *   }
    *
    *   distributionStrategy.scope {
    *     // In "cross-tower" context
    *     ...
    *     val mergedResults = distributionStrategy.forEachTower(() => fn(3))
    *     // `mergedResults` has the values from every tower execution of `fn`.
    *     val resultsList = distributionStrategy.unwrap(mergedResults)
    *   }
    * }}}
    *
    * @param  fn     Function that will be run once per tower.
    * @param  values Wrapped values that will be unwrapped when invoking `fn` on each tower.
    * @return Merged return value of `fn` across all towers.
    */
  def forEachTower[T: Distributable, R](
      fn: Seq[T] => R,
      values: Seq[DistributedValue[T]]
  )(implicit context: CrossTowerContext): R = {
    context.strategy.forEachTower(fn, values)
  }
}
