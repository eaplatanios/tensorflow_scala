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

package org.platanios.tensorflow.api.ops.training.distribute.ops

import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.core.{DeviceSpecification, Devices}
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.packers.ConcatenateAndSplitPacker
import org.platanios.tensorflow.api.ops.training.distribute.values.{MirroredValue, PerDeviceValue}
import org.platanios.tensorflow.api.ops.{Basic, Op, OutputLike}

import scala.collection.JavaConverters._

/** Base class for cross-tower reduction and broadcasting algorithms.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class CrossTowerOps {
  /** Reduces `value` to `destination`.
    *
    * It runs the reduction operation defined by `reduction` and puts the result on `destination`.
    *
    * @param  reduction   Reduction method.
    * @param  value       Per-device value to be reduced.
    * @param  destination Reduction destination.
    * @return Reduced mirrored value.
    */
  def reduce[D: Destination](
      reduction: Reduction,
      value: PerDeviceValue[OutputLike],
      destination: Option[D]
  ): MirroredValue[OutputLike]

  /** Reduces a batch per-device values to the corresponding provided destinations.
    *
    * For each tuple in `valueDestinationPairs`, this method reduces each first element to each second element which
    * indicates the corresponding destination.
    *
    * @param  reduction             Reduction method.
    * @param  valueDestinationPairs Sequence of per-device values and destinations pairs. If a destination is `None`,
    *                               then the destination is set to match the devices of the corresponding per-device
    *                               value.
    * @return Sequence of reduced mirrored values.
    */
  def batchReduce[D: Destination](
      reduction: Reduction,
      valueDestinationPairs: Seq[(PerDeviceValue[OutputLike], Option[D])]
  ): Seq[MirroredValue[OutputLike]]

  /** Broadcasts `value` to `destination`.
    *
    * @param  value       Value to broadcast.
    * @param  destination Broadcast destination.
    * @return Broadcasted mirrored value.
    */
  def broadcast[O <: OutputLike, D: Destination](
      value: O,
      destination: D
  ): MirroredValue[O] = {
    CrossTowerOps.simpleBroadcast(value, destination)
  }
}

object CrossTowerOps {
  /** Picks the best cross-tower ops implementation, based on the requested devices and provided session configuration.
    *
    * @param  requestedDevices Requested devices passed to the distribution strategy.
    * @param  sessionConfig    Optional TensorFlow session configuration.
    * @return Cross-tower ops implementation instance to use.
    */
  def best(
      requestedDevices: Set[DeviceSpecification],
      sessionConfig: Option[SessionConfig] = None
  ): CrossTowerOps = {
    val machineDevices = Devices.local(sessionConfig)
    val usingDevices = machineDevices.filter(d => {
      val name = d.getName
      val requested = requestedDevices.contains(DeviceSpecification.fromString(name))
      if (!requested)
        logger.info(s"Available device not used by the distribution strategy because it was not requested: $name.")
      requested
    })
    if (usingDevices.size != requestedDevices.size) {
      logger.info("Not all devices requested in the distribute strategy are visible to TensorFlow sessions and thus, " +
          "defaulting to 'SingleDeviceReduceCrossTowerOps'..")
      SingleDeviceReduceCrossTowerOps()
    } else if (usingDevices.exists(_.getDeviceType.toLowerCase != "gpu")) {
      logger.info("Non-GPU devices do not support all-reduce cross-tower ops and thus, " +
          "defaulting to 'SingleDeviceReduceCrossTowerOps'.")
      SingleDeviceReduceCrossTowerOps()
    } else {
      val deviceLinks = usingDevices.map(_.getLocality.getLinks.getLinkList.asScala.map(_.getDeviceId).toSet)
      pickAllReduceAlgorithm(deviceLinks)
    }
  }

  /** Returns an all-reduce cross tower ops instance, configured in a reasonable/efficient way based on the provided
    * device links.
    *
    * @param  deviceLinks Sequence containing the device IDs of the devices, each device is connected to.
    * @return Cross-tower ops instance to use.
    */
  private def pickAllReduceAlgorithm(deviceLinks: Seq[Set[Int]]): CrossTowerOps = {
    val hasDGX1LikeLinks = deviceLinks.zip(DGX1_LINKS)
        .zipWithIndex
        .forall(links => links._1._1 == links._1._2 || links._1._1 == links._1._2 - links._2)
    if (hasDGX1LikeLinks) {
      logger.info(
        "Configured cross-tower ops to use the hierarchical copy all-reduce, " +
            s"using a concatenate-and-split packer with ${deviceLinks.size} packs.")
      AllReduceCrossTowerOps(ConcatenateAndSplitPacker(deviceLinks.size), AllReduceCrossTowerOps.HierarchicalCopy)
    } else {
      logger.info(
        "Configured cross-tower ops to use the NCCL all-reduce, using a concatenate-and-split packer with 1 pack.")
      AllReduceCrossTowerOps(ConcatenateAndSplitPacker(deviceLinks.size), AllReduceCrossTowerOps.NCCL)
    }
  }

  /** Represents the machine topology of a DGX-1 server.
    *
    * The device peer-to-peer matrix looks as follows:
    * DMA: 0 1 2 3 4 5 6 7
    * 0:   Y Y Y Y Y N N N
    * 1:   Y Y Y Y N Y N N
    * 2:   Y Y Y Y N N Y N
    * 3:   Y Y Y Y N N N Y
    * 4:   Y N N N Y Y Y Y
    * 5:   N Y N N Y Y Y Y
    * 6:   N N Y N Y Y Y Y
    * 7:   N N N Y Y Y Y Y
    */
  private val DGX1_LINKS: Seq[Set[Int]] = Seq(
    Set(0, 1, 2, 3, 4), Set(0, 1, 2, 3, 5), Set(0, 1, 2, 3, 6), Set(0, 1, 2, 3, 7),
    Set(0, 4, 5, 6, 7), Set(1, 4, 5, 6, 7), Set(2, 4, 5, 6, 7), Set(3, 4, 5, 6, 7))

  private[distribute] def simpleBroadcast[O <: OutputLike, D: Destination](
      value: O,
      destination: D
  ): MirroredValue[O] = {
    val index = implicitly[Destination[D]].devices(destination).map(d => {
      Op.device(d.toString) {
        d -> Basic.identity(value)
      }
    }).toMap
    MirroredValue(index)
  }

  private[distribute] def simpleReduce(
      value: PerDeviceValue[OutputLike],
      reduceToDevice: DeviceSpecification,
      reduction: Reduction
  ): OutputLike = {
    // TODO: [DISTRIBUTE] What about "MapOutput"?
    val values = value.index.values.toSeq
    if (values.isEmpty)
      throw new IllegalArgumentException("The value being reduced must be non-empty.")
    Op.device(reduceToDevice.toString) {
      reduction.reduce(values)
    }
  }
}
