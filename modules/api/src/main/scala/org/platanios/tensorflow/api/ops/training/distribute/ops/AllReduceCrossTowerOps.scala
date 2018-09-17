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

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.{Output, OutputLike}
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.packers.Packer
import org.platanios.tensorflow.api.ops.training.distribute.values.{MirroredValue, PerDeviceValue}

/** Cross-tower ops that perform an all-reduce operation.
  *
  * @param  packer    Packer that can be used to repack/aggregate values before applying the all-reduce,
  *                   in order to allow for more efficient cross-device transportation.
  * @param  algorithm All-reduce algorithm to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class AllReduceCrossTowerOps[P] protected(
    val packer: Packer[P],
    val algorithm: AllReduceCrossTowerOps.Algorithm
) extends CrossTowerOps {
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
  ): MirroredValue[OutputLike] = destination match {
    case None =>
      batchAllReduce(reduction, Seq(value)).head
    case Some(d) if Destination.devicesMatch(d, value) =>
      batchAllReduce(reduction, Seq(value)).head
    case Some(d) =>
      val devices = Destination.devicesFrom(d)
      val reduceToDevice = devices.head
      val reduced = CrossTowerOps.simpleReduce(value, reduceToDevice, reduction)
      broadcast(reduced, devices)
  }

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
  ): Seq[MirroredValue[OutputLike]] = {
    if (Destination.allDevicesMatch(valueDestinationPairs)) {
      batchAllReduce(reduction, valueDestinationPairs.map(_._1))
    } else {
      logger.warn("Efficient 'batchReduce' is not supported if the destinations are different.")
      valueDestinationPairs.map(p => reduce(reduction, p._1, p._2))
    }
  }

  /** Performs a batched all-reduce operation.
    *
    * @param  reduction Reduction method.
    * @param  values    Per-device values to be reduced.
    * @return Sequence of reduced mirrored values.
    */
  protected def batchAllReduce(
      reduction: Reduction,
      values: Seq[PerDeviceValue[OutputLike]]
  ): Seq[MirroredValue[OutputLike]] = {
    // Pack the gradients before performing the reduction, in order to ensure efficient communication.
    val destinations = values.head.devices
    val grouped = AllReduceCrossTowerOps.groupValueByDevice(values)
    val (packed, packInformation) = packer.pack(grouped)

    // Perform the actual aggregation of the packed values. Note that the packed values sharded among different
    // aggregation trees. Therefore, it is important to strike a balance on the number of splits.
    val reduced = algorithm(packed)

    // Unpack the reduction result.
    val unpacked = packer.unpack(reduced, packInformation)

    // Finally, ungroup into mirrored values.
    AllReduceCrossTowerOps.ungroupToMirrored(unpacked, destinations, reduction)
  }
}

object AllReduceCrossTowerOps {
  def apply[P](
      packer: Packer[P],
      allReduceAlgorithm: AllReduceCrossTowerOps.Algorithm
  ): AllReduceCrossTowerOps[P] = {
    new AllReduceCrossTowerOps[P](packer, allReduceAlgorithm)
  }

  trait Algorithm {
    def apply(groupedValues: Seq[Seq[OutputLike]]): Seq[Seq[OutputLike]]
  }

  object HierarchicalCopy extends Algorithm {
    override def apply(groupedValues: Seq[Seq[OutputLike]]): Seq[Seq[OutputLike]] = ??? // TODO: [DISTRIBUTE] !!!
  }

  object NCCL extends Algorithm {
    override def apply(groupedValues: Seq[Seq[OutputLike]]): Seq[Seq[OutputLike]] = ??? // TODO: [DISTRIBUTE] !!!
  }

  /** Groups values into sub-lists based on their devices.
    *
    * This grouping is needed in order to call the all-reduce library.
    *
    * @param  values Sequence of per-device values.
    * @return Sequence of sequences for each device in the provided per-device values, where each sub-sequence has all
    *         values for the corresponding device.
    * @throws IllegalArgumentException If the provided values are not all distributed on the same devices.
    */
  @throws[IllegalArgumentException]
  private[AllReduceCrossTowerOps] def groupValueByDevice[T](
      values: Seq[PerDeviceValue[T]]
  ): Seq[Seq[T]] = {
    val devices = values.head.devices
    values.map(perDeviceValue => {
      if (perDeviceValue.devices != devices)
        throw new IllegalArgumentException("The values are not all distributed on the same devices.")
      perDeviceValue.index.values
    }).transpose
  }

  /** Ungroups results from all-reduce operations into mirrored values.
    *
    * Based on the provided reduction method, the mirrored values may be preprocessed. For example, for `MeanReduction`,
    * each all-reduce result will be divided by the number of destinations.
    *
    * @param  groupedReduced Sequence of sequences for each device in the provided per-device values, where each
    *                        sub-sequence has all values for the corresponding device.
    * @param  destinations   Destination devices for the returned mirrored values.
    * @param  reduction      Reduction method to use.
    * @return Sequence of ungrouped mirrored values.
    */
  private[AllReduceCrossTowerOps] def ungroupToMirrored(
      groupedReduced: Seq[Seq[OutputLike]],
      destinations: Seq[DeviceSpecification],
      reduction: Reduction
  ): Seq[MirroredValue[OutputLike]] = {
    groupedReduced.transpose.map(perDeviceReduced => {
      val processedPerDeviceReduced = perDeviceReduced.map(reduction.processUngroupedValue(_, destinations))
      MirroredValue(destinations.zip(processedPerDeviceReduced).toMap)
    })
  }
}
