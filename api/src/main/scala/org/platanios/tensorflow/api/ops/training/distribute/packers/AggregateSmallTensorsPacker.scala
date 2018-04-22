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

package org.platanios.tensorflow.api.ops.training.distribute.packers

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}
import org.platanios.tensorflow.api.types.FLOAT32

import scala.collection.mutable

/** Packer that concatenates small tensors together for reduction.
  *
  * @param  maxBytes  Largest tensor eligible for aggregation, in terms of the number of bytes it takes up.
  * @param  maxGroups Largest permitted aggregation of small tensors.
  * @throws InvalidArgumentException If `maxBytes` or `maxGroups` is less than 1.
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[InvalidArgumentException]
class AggregateSmallTensorsPacker protected(
    val maxBytes: Long,
    val maxGroups: Int
) extends Packer[AggregateSmallTensorsPacker.PackInformation] {
  if (maxBytes < 1)
    throw InvalidArgumentException(s"'maxBytes' must be at least 1, but was set to $maxBytes.")
  if (maxGroups < 1)
    throw InvalidArgumentException(s"'maxGroups' must be at least 1, but was set to $maxGroups.")

  /** Packs the provided values.
    *
    * @param  grouped Grouped values (per device).
    * @return Packed values, ready for reduction, along with information that is necessary for unpacking later on.
    * @throws InvalidArgumentException If the provided grouped values are inconsistent in any way.
    */
  @throws[InvalidArgumentException]
  override def pack(
      grouped: Seq[Seq[Output]]
  ): (Seq[Seq[Output]], Option[AggregateSmallTensorsPacker.PackInformation]) = {
    val partitions = grouped.head.zipWithIndex.partition(gi => {
      gi._1.dataType == FLOAT32 && 4 * gi._1.shape.numElements <= maxBytes
    })
    val smallIndices = partitions._1.map(_._2)
    val largeIndices = partitions._2.map(_._2)
    val (smallRanges, smallSingles) = AggregateSmallTensorsPacker.extractRanges(smallIndices, maxGroups)
    if (smallRanges.isEmpty) {
      (grouped, None)
    } else {
      val ungroupedIndices = (largeIndices ++ smallSingles).sorted
      val numValues = grouped.head.size
      val packZipped = grouped.zipWithIndex.map {
        case (values, deviceIndex) =>
          if (values.size != numValues)
            throw InvalidArgumentException("The number of values must match across devices.")
          val (packedSmall, packInformationSeq) = smallRanges.zipWithIndex.map(ri => {
            val range = ri._1
            val (members, shapes) = Op.createWith(nameScope = "Pack") {
              values.slice(range._1, range._2 + 1).map(v => {
                val member = Op.device(v.device)(Basic.reshape(v, Shape(-1)))
                (member, v.shape)
              }).unzip
            }
            val packed = Op.device(members.head.device)(Basic.concatenate(members))
            val information = AggregateSmallTensorsPacker.PackedValuesInformation(range, shapes)
            (packed, (deviceIndex, ri._2) -> information)
          }).unzip
          val packed = packedSmall ++ ungroupedIndices.map(values)
          val packInformation = packInformationSeq.toMap
          (packed, packInformation)
      }
      val (packed, perDevicePackInformation) = packZipped.unzip
      val packInformation = AggregateSmallTensorsPacker.PackInformation(perDevicePackInformation.reduce(_ ++ _))
      (packed, Some(packInformation))
    }
  }

  /** Reverses the packing performed by `pack`, on the provided packed values.
    *
    * @param  packed          Packed values to unpack.
    * @param  packInformation Information from the packing process that is necessary for unpacking.
    * @return Unpacked `packed`.
    * @throws InvalidArgumentException If not pack information is provided, while it is actually necessary.
    */
  @throws[InvalidArgumentException]
  override def unpack(
      packed: Seq[Seq[Output]],
      packInformation: Option[AggregateSmallTensorsPacker.PackInformation]
  ): Seq[Seq[Output]] = packInformation match {
    case None => packed
    case Some(information) =>
      val numDevices = packed.size
      val numPacked = information.map.size / numDevices
      packed.zipWithIndex.map {
        case (deviceValues, deviceIndex) =>
          val newValues = mutable.ListBuffer(deviceValues.slice(numPacked, deviceValues.size): _*)
          (0 until numPacked).foreach(i => {
            val value = deviceValues(i)
            val packedValuesInformation = information.map((deviceIndex, i))
            val unpacked = Op.device(value.device) {
              Op.createWith(nameScope = "Unpack") {
                val splits = Basic.split(value, packedValuesInformation.shapes.map(_.numElements))
                val shapes = packedValuesInformation.shapes
                splits.zip(shapes).map(ss => Basic.reshape(ss._1, ss._2))
              }
            }
            // TODO: [DISTRIBUTE] Can become more elegant with a map and without the mutable list buffer.
            (packedValuesInformation.range._1 until packedValuesInformation.range._2)
                .zipWithIndex
                .foreach(i => newValues.insert(i._1, unpacked(i._2)))
          })
          newValues
      }
  }
}

object AggregateSmallTensorsPacker {
  def apply(
      maxBytes: Long = 1048576,
      maxGroups: Int = 16
  ): AggregateSmallTensorsPacker = {
    new AggregateSmallTensorsPacker(maxBytes, maxGroups)
  }

  case class PackedValuesInformation(range: (Int, Int), shapes: Seq[Shape])

  /** Contains information collected while packing that is necessary for unpacking.
    *
    * @param  map Map from device index and value index pairs, to packed values information.
    */
  case class PackInformation(map: Map[(Int, Int), PackedValuesInformation])

  /** Extracts consecutive ranges and singles from the provided sequence of indices.
    *
    * @param  indices        Sequence of indices.
    * @param  rangeSizeLimit Largest range size to return. If a larger consecutive range exists, it will be returned
    *                        as multiple ranges.
    * @return Tuple with a sequence of ranges and a sequence of singles, in the original order in which they appear in
    *         `indices`. Note that for the ranges, both range boundaries are inclusive.
    */
  private[AggregateSmallTensorsPacker] def extractRanges(
      indices: Seq[Int],
      rangeSizeLimit: Int = 32
  ): (Seq[(Int, Int)], Seq[Int]) = {
    if (indices.isEmpty) {
      (Seq.empty, Seq.empty)
    } else {
      // TODO: [DISTRIBUTE] Can become more elegant with a fold.
      var first = indices.head
      var last = first
      val ranges = mutable.ListBuffer.empty[(Int, Int)]
      val singles = mutable.ListBuffer.empty[Int]
      indices.tail.foreach {
        case i if i == last + 1 && (last - first) <= rangeSizeLimit => last = i
        case i =>
          if (last > first)
            ranges.append((first, last))
          else
            singles.append(first)
          first = i
          last = i
      }
      if (last > first)
        ranges.append((first, last))
      else
        singles.append(first)
      (ranges, singles)
    }
  }
}
