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

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, Op, OutputLike}
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.values.{DistributedValue, MirroredValue, MirroredVariable}
import org.platanios.tensorflow.api.ops.variables.Variable

/**
  * @author Emmanouil Antonios Platanios
  */
class DefaultDistributionStrategy extends DistributionStrategy {
  override protected def createVariable: ColocatedVariableGetter = {
    ???
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
  override def colocateVariablesWith[R](
      colocationOps: Set[Op]
  )(block: => R)(implicit context: DistributionContext): R = {
    Op.colocateWith(colocationOps) {
      block
    }
  }

  /** Mirrors `value` to all worker devices.
    *
    * @param  value   Value to broadcast.
    * @param  devices Destination devices.
    * @return Mirrored value.
    */
  override def broadcast[T: Distributable](
      value: T,
      devices: Seq[DeviceSpecification] = Seq.empty
  )(implicit context: CrossTowerContext): MirroredValue[T] = {
    if (devices.isEmpty) {
      val device = implicitly[Distributable[T]].device(value)
      MirroredValue(Map(device -> value))
    } else {
      throw new UnsupportedOperationException("The default distribution strategy does not yet support broadcasting.")
    }
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
  override def forEachTower[T: Distributable, R](
      fn: Seq[T] => R,
      values: Seq[DistributedValue[T]]
  )(implicit context: CrossTowerContext): R = {
    implicit val context: InTowerContext = InTowerContext(this, towerID = 0)
    fn(values.map(_.get()))
  }

  /** Combines values across towers into one value.
    *
    * @param  reduction Reduction method to use.
    * @param  value     Value to reduce.
    * @param  devices   Optional devices on which to copy the reduced value.
    * @return Reduced value.
    */
  override def reduce[T: Distributable](
      reduction: Reduction,
      value: DistributedValue[T],
      devices: Seq[DeviceSpecification]
  )(implicit context: CrossTowerContext): MirroredValue[T] = {
    // TODO: [DISTRIBUTE] !!! Use `devices`.
    MirroredValue(value.index)
  }

  /** Runs `fn` to update `variable` using inputs mirrored to the same devices.
    *
    * If `variable` is mirrored across multiple devices, then this method implements logic like:
    * {{{
    *   val results = variable.index.map {
    *     case (deviceSpec, variable) => tf.createWith(device = deviceSpec.toString) {
    *       fn(variable)
    *     }
    *   }
    *   merged(results)
    * }}}
    *
    * Otherwise this returns `fn(variable)` colocated with `variable`.
    *
    * @param  variable  Variable to update.
    * @param  fn        Update function to use.
    * @param  arguments Mirrored arguments that should be passed to `fn`.
    * @return Merged return value of `fn` across all towers.
    */
  override def update[T: Distributable, R](
      variable: MirroredVariable,
      fn: (Variable, Seq[T]) => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R] = {
    val resultIndex = variable.index.map {
      case (deviceSpec, localVariable) =>
        val localArguments = arguments.map(_.index(deviceSpec))
        Op.colocateWith(Set(localVariable.op)) {
          withUpdateDevice(localVariable.device) {
            deviceSpec -> fn(localVariable, localArguments)
          }
        }
    }
    MirroredValue(resultIndex)
  }

  /** Runs `fn` on the devices specified by `colocateWith`, with the provided arguments.
    *
    * @param  colocateWith Destination on which to execute `fn`.
    * @param  fn           Function to use for the update.
    * @param  arguments    Mirrored arguments that should be passed to `fn`.
    * @return Merged return value of `fn` across all towers.
    * @throws InvalidArgumentException If the provided `colocateWith` argument is invalid (e.g., too many devices).
    */
  @throws[InvalidArgumentException]
  override def updateNonSlot[D: Destination, T: Distributable, R](
      colocateWith: D,
      fn: Seq[T] => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R] = {
    val colocateWithDevices = Destination.devicesFrom(colocateWith)
    if (colocateWithDevices.size > 1)
      throw InvalidArgumentException("Too many devices specified for the colocation. Only one device is supported.")
    val resultIndex = arguments.head.index.keys.map(deviceSpec => {
      val localArguments = arguments.map(_.index(deviceSpec))
      Op.device(colocateWithDevices.head.toString) {
        withUpdateDevice(colocateWithDevices.head.toString) {
          deviceSpec -> fn(localArguments)
        }
      }
    })
    MirroredValue(resultIndex.toMap)
  }

  /** Returns a copy of `fn(value)` on `device`. This is useful for getting a mirrored value onto a device. The method
    * will attempt to avoid a copy by checking if the value is already on the destination device.
    *
    * @param  value       Value (which may be mirrored) to copy and fetch.
    * @param  destination Device to copy the value to.
    * @param  fn          Optional function to apply to the value on the source device, before copying.
    * @return Fetched value in `device`.
    */
  override def fetch[T: Distributable : OutputLike](
      value: DistributedValue[T],
      destination: String = "/device:CPU:0",
      fn: T => T = (t: T) => t
  )(implicit context: CrossTowerContext): T = {
    val processedValue = DistributedValue(value.index.map(kv => {
      Op.colocateWith(Set(implicitly[Distributable[T]].op(kv._2))) {
        kv._1 -> fn(kv._2)
      }
    }), value.distributionType)
    Op.createWith(device = destination) {
      Basic.identity(processedValue.index.values.head)
    }
  }

  /** Returns the list of all per-device values contained in `value`.
    *
    * @param  value A value returned by `forEachTower()`, or a variable created in `scope`.
    * @return Sequence of values contained in `value`.
    */
  override def unwrap[T: Distributable](
      value: DistributedValue[T]
  )(implicit context: CrossTowerContext): Seq[T] = {
    Seq(value.get())
  }

  /** Returns `true` if there is only a single tower, and `false`, otherwise.
    *
    * If `true`, `forEachTower(fn)` will only call `fn` once.
    * If `false`, `forEachTower(fn)` may call `fn` multiple times.
    */
  override def isSingleTower: Boolean = true

  /** Returns number of towers, for purposes of averaging across towers. */
  override def numTowers: Int = 1

  /** Returns the devices used to run `forEachTower()` calls. */
  override def workerDevices: Seq[String] = {
    throw new UnsupportedOperationException(
      "`workerDevices` is not supported by the default distribution strategy.")
  }

  /** Returns the devices used for variable and updates placement. */
  override def parameterDevices: Seq[String] = {
    throw new UnsupportedOperationException(
      "`parameterDevices` is not supported by the default distribution strategy.")
  }

  /** Returns the devices used for non-slot variables.
    *
    * Create variables on these devices in a `colocateVariablesWith(nonSlotDevices(...)):` block. Then, update them
    * using `updateNonSlot()`.
    *
    * @param  variables Variables being optimized.
    * @return Colocation ops for non-slot variables.
    */
  override def nonSlotColocationOps(variables: Seq[Variable]): Set[Op] = {
    Set(variables.minBy(_.name).op)
  }

  /** Returns a map from worker devices to indices.
    *
    * TODO: [DISTRIBUTE] Settle on the interface of `forEachTower()` first.
    * This map might be passed as an argument to `forEachTower()`, as in:
    * {{{
    *   distributionStrategy.scope {
    *     def fn(deviceIndex: Int): Unit = {
    *       // `fn` is being executed on device `distributionStrategy.workerDevices(deviceIndex)`.
    *     }
    *     distributionStrategy.forEachTower(fn, distributionStrategy.workerDeviceIndex)
    *   }
    * }}}
    */
  override def workerDeviceIndex(implicit context: CrossTowerContext): Map[String, Int] = {
    throw new UnsupportedOperationException(
      "`workerDeviceIndex` is not supported by the default distribution strategy.")
  }
}
