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
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.FLOAT32

import scala.collection.mutable

/** Gradients packer that concatenates small tensors together for reduction.
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

  /** Packs the gradients.
    *
    * This method looks through the first tower for gradients of the same type (`FLOAT32`), and small size, that are all
    * sequential. Replace each such group with a new tensor that is a flattened concatenation of the tensors in the
    * group.
    *
    * @param  groupedGradientsAndVariables Grouped gradients and variables (per device).
    * @return Packed gradients, ready for reduction, along with information that is necessary for unpacking later on.
    * @throws InvalidArgumentException If the provided grouped gradients and variables are inconsistent in some way.
    */
  @throws[InvalidArgumentException]
  override def pack(
      groupedGradientsAndVariables: Seq[Seq[(Output, Variable)]]
  ): (Seq[Seq[Output]], Option[AggregateSmallTensorsPacker.PackInformation]) = {
    val partitions = groupedGradientsAndVariables.head.map(_._1).zipWithIndex.partition(gi => {
      gi._1.dataType == FLOAT32 && 4 * gi._1.shape.numElements <= maxBytes
    })
    val smallIndices = partitions._1.map(_._2)
    val largeIndices = partitions._2.map(_._2)
    val (smallRanges, smallSingles) = AggregateSmallTensorsPacker.extractRanges(smallIndices, maxGroups)
    if (smallRanges.isEmpty) {
      (groupedGradientsAndVariables.map(_.map(_._1)), None)
    } else {
      val ungroupedIndices = (largeIndices ++ smallSingles).sorted
      val numGradients = groupedGradientsAndVariables.head.size
      val packZipped = groupedGradientsAndVariables.zipWithIndex.map {
        case (gradientsAndVariables, deviceIndex) =>
          if (gradientsAndVariables.size != numGradients)
            throw InvalidArgumentException("The number of gradients must match across devices.")
          val (packedSmallGradients, packInformationSeq) = smallRanges.zipWithIndex.map(ri => {
            val range = ri._1
            val (members, variables, shapes) = Op.createWith(nameScope = "Pack") {
              gradientsAndVariables.slice(range._1, range._2 + 1).map(gv => {
                val member = Op.device(gv._1.device)(Basic.reshape(gv._1, Shape(-1)))
                (member, gv._2, gv._1.shape)
              }).unzip3
            }
            val packedGradient = Op.device(members.head.device)(Basic.concatenate(members))
            val information = AggregateSmallTensorsPacker.PackedGradientInformation(range, variables, shapes)
            (packedGradient, (deviceIndex, ri._2) -> information)
          }).unzip
          val packedGradients = packedSmallGradients ++ ungroupedIndices.map(gradientsAndVariables(_)._1)
          val packInformation = packInformationSeq.toMap
          (packedGradients, packInformation)
      }
      val (packedGradients, perDevicePackInformation) = packZipped.unzip
      val packInformation = AggregateSmallTensorsPacker.PackInformation(perDevicePackInformation.reduce(_ ++ _))
      (packedGradients, Some(packInformation))
    }
  }

  /** Reverses the packing performed by `pack`, on the provided packed gradients.
    *
    * @param  packedGradients Packed gradients to unpack.
    * @param  packInformation Information from the packing process that is necessary for unpacking.
    * @return Unpacked `packedGradients`.
    * @throws InvalidArgumentException If not pack information is provided, while it is actually necessary.
    */
  @throws[InvalidArgumentException]
  override def unpack(
      packedGradients: Seq[Seq[(Output, Variable)]],
      packInformation: Option[AggregateSmallTensorsPacker.PackInformation]
  ): Seq[Seq[(Output, Variable)]] = packInformation match {
    case None => packedGradients
    case Some(information) =>
      val numDevices = packedGradients.size
      val numPacked = information.map.size / numDevices
      packedGradients.zipWithIndex.map {
        case (deviceGradients, deviceIndex) =>
          val newGradientsAndVariables = mutable.ListBuffer(deviceGradients.slice(numPacked, deviceGradients.size): _*)
          (0 until numPacked).foreach(i => {
            val gradient = deviceGradients(i)._1
            val packedGradientInformation = information.map((deviceIndex, i))
            val unpackedGradientsAndVariables = Op.device(gradient.device) {
              Op.createWith(nameScope = "Unpack") {
                val splits = Basic.split(gradient, packedGradientInformation.shapes.map(_.numElements))
                val shapes = packedGradientInformation.shapes
                val variables = packedGradientInformation.variables
                (splits, shapes, variables).zipped.map {
                  case (split, shape, variable) => (Basic.reshape(split, shape), variable)
                }
              }
            }
            // TODO: [DISTRIBUTE] Can become more elegant with a map and without the mutable list buffer.
            (packedGradientInformation.range._1 until packedGradientInformation.range._2)
                .zipWithIndex
                .foreach(i => newGradientsAndVariables.insert(i._1, unpackedGradientsAndVariables(i._2)))
          })
          newGradientsAndVariables
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

  case class PackedGradientInformation(range: (Int, Int), variables: Seq[Variable], shapes: Seq[Shape])

  /** Contains information collected while packing that is necessary for unpacking.
    *
    * @param  map Map from device index and variable index pairs, to packed gradient information.
    */
  case class PackInformation(map: Map[(Int, Int), PackedGradientInformation])

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
