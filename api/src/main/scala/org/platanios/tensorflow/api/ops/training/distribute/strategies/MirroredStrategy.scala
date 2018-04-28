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
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, Op, Output, OutputLike}
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.ops.{CrossTowerOps, SingleDeviceReduceCrossTowerOps}
import org.platanios.tensorflow.api.ops.training.distribute.values._
import org.platanios.tensorflow.api.ops.variables.Variable

/**
  * @author Emmanouil Antonios Platanios
  */
class MirroredStrategy protected(
    val devices: Set[String],
    val prefetchingDevice: Option[String] = None,
    protected var crossTowerOps: Option[CrossTowerOps] = None
) extends DistributionStrategy {
  require(devices.size > 1, "Must specify at least two devices.")

  /** Device specifications. */
  val deviceSpecs: Set[DeviceSpecification] = devices.map(DeviceSpecification.fromString)

  /** Device indices. */
  val deviceIndex: Map[DeviceSpecification, Int] = deviceSpecs.zipWithIndex.toMap

  override def configure(sessionConfig: SessionConfig): Unit = crossTowerOps match {
    case Some(_) => ()
    case None => crossTowerOps = Some(CrossTowerOps.best(deviceSpecs, Some(sessionConfig)))
  }

  protected def getCrossTowerOps: CrossTowerOps = crossTowerOps match {
    case Some(ops) => ops
    case None =>
      val ops = SingleDeviceReduceCrossTowerOps()
      crossTowerOps = Some(ops)
      ops
  }

  override protected def createVariable: ColocatedVariableGetter = {
    ???
  }

  override def forEachTower[T: Distributable, R](
      fn: Seq[T] => R,
      values: Seq[DistributedValue[T]]
  )(implicit context: CrossTowerContext): R = {
    ???
  }

  override def broadcast[O <: OutputLike](
      value: O,
      devices: Seq[DeviceSpecification] = Seq.empty
  )(implicit context: CrossTowerContext): MirroredValue[O] = {
    getCrossTowerOps.broadcast(value, if (devices.isEmpty) deviceSpecs else devices)
  }

  override def reduce[D: Destination](
      reduction: Reduction,
      value: PerDeviceValue[OutputLike],
      destination: Option[D] = None
  )(implicit context: CrossTowerContext): MirroredValue[OutputLike] = {
    getCrossTowerOps.reduce(reduction, value, destination)
  }

  override def batchReduce[D: Destination](
      reduction: Reduction,
      valueDestinationPairs: Seq[(PerDeviceValue[OutputLike], Option[D])]
  )(implicit context: CrossTowerContext): Seq[DistributedValue[OutputLike]] = {
    getCrossTowerOps.batchReduce(reduction, valueDestinationPairs)
  }

  override def update[T: Distributable, R: Distributable](
      variable: MirroredVariable,
      fn: (Variable, Seq[T]) => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R] = {
    val resultIndex = variable.index.map {
      case (deviceSpec, localVariable) =>
        val device = deviceSpec.toString
        Op.createWithNameScope(s"Update${deviceIndex(deviceSpec)}") {
          Op.device(device) {
            withUpdateDevice(device) {
              val localArguments = arguments.map(_.get(device))
              deviceSpec -> fn(localVariable, localArguments)
            }
          }
        }
    }
    // TODO: !!! values.regroup
    // TODO: !!! Maybe this should be aware of the "structure" of `colocateWith`.
    MirroredValue(resultIndex)
  }

  @throws[InvalidArgumentException]
  override def updateNonSlot[D: Destination, T: Distributable, R: Distributable](
      colocateWith: D,
      fn: Seq[T] => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R] = {
    val valuePerDevice = Destination.devicesFrom(colocateWith).map(deviceSpec => {
      val device = deviceSpec.toString
      Op.createWithNameScope(s"Update${deviceIndex(deviceSpec)}") {
        Op.device(device) {
          withUpdateDevice(device) {
            val localArguments = arguments.map(_.get(device))
            deviceSpec -> fn(localArguments)
          }
        }
      }
    }).toMap
    // TODO: !!! values.regroup
    // TODO: !!! Maybe this should be aware of the "structure" of `colocateWith`.
    MirroredValue(valuePerDevice)
  }

  @throws[InvalidArgumentException]
  override def fetch(
      variable: DistributedVariable,
      destination: String = "/device:CPU:0",
      fn: Output => Output = (o: Output) => o
  )(implicit context: CrossTowerContext): Output = {
    variable match {
      case v: PerDeviceVariable =>
        val processedValue = reduce(v.reduction, PerDeviceValue(v.index.mapValues(_.value)), Some(destination))
        Op.device(destination)(fn(unwrap(processedValue).head.toOutput))
      case v: MirroredVariable if v.onDevice(destination) => Op.device(destination)(fn(v.get(destination).value))
      case v: MirroredVariable =>
        devices.find(v.onDevice) match {
          case Some(d) => Op.device(destination)(Basic.identity(Op.device(d)(fn(v.get(d).value))))
          case None => throw InvalidArgumentException(
            s"Could not find destination '$destination' in list of devices '${v.devices}'.")
        }
      case _ => throw InvalidArgumentException(
        "Unsupported distributed variable type. Must be either per-device or mirrored.")
    }
  }

  override def unwrap[T: Distributable](
      value: DistributedValue[T]
  )(implicit context: CrossTowerContext): Seq[T] = {
    // Return in a deterministic order.
    value.devices.map(_.toString).sorted.map(value.get)
  }

  override def isSingleTower: Boolean = false

  override def numTowers: Int = devices.size

  override def workerDevices: Set[String] = devices

  override def parameterDevices: Set[String] = devices

  override def nonSlotDevices(variables: Seq[Variable]): Set[DeviceSpecification] = deviceSpecs

  override def workerDeviceIndex(implicit context: CrossTowerContext): Map[DeviceSpecification, Int] = deviceIndex
}
