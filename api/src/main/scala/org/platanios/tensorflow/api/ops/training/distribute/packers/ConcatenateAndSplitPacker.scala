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

import org.platanios.tensorflow.api.core.{NewAxis, Shape}
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputLike}

/** Packer that concatenates all tensors together and then splits them into packs for reduction.
  *
  * This packer aggregates values into a total of `numPacks` splits.
  *
  * @param  numPacks Number of packs to split the values into.
  * @throws InvalidArgumentException If `numPacks` is less than 1.
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[InvalidArgumentException]
class ConcatenateAndSplitPacker protected(val numPacks: Int)
    extends Packer[ConcatenateAndSplitPacker.PackInformation] {
  if (numPacks < 1)
    throw InvalidArgumentException(s"'numPacks' must be at least 1, but was set to $numPacks.")

  /** Packs the provided values.
    *
    * @param  grouped Grouped values (per device).
    * @return Packed values, ready for reduction, along with information that is necessary for unpacking later on.
    * @throws InvalidArgumentException If the provided grouped values are inconsistent in any way.
    */
  @throws[InvalidArgumentException]
  override def pack(
      grouped: Seq[Seq[OutputLike]]
  ): (Seq[Seq[OutputLike]], Option[ConcatenateAndSplitPacker.PackInformation]) = {
    val packed = grouped.map(values => {
      Op.colocateWith(Set(values.head.op)) {
        // Flatten all the values.
        val flattened = values.map(v => Basic.reshape(v, Shape(-1)))
        // Remember the original shapes and sizes of all the values.
        val towerShapes = values.map(v => Basic.shape(v))
        val towerSizes = values.map(v => Basic.size(v))
        // Concatenate all the flat values into a big flat tensor.
        val concatenated = Basic.concatenate(flattened, axis = 0)

        // Split the concatenated tensor into packs. In cases where the total size is not divisible by `numPacks`, the
        // last pack gets more elements.
        // TODO: [DISTRIBUTE] It is also possible to optimize away the concatenation.
        val numSplits = Basic.constant(numPacks, name = "NumSplits")
        val totalSize = Basic.size(concatenated)
        val splitSize = Math.truncateDivide(totalSize, numSplits)
        val splitSizeLast = totalSize - splitSize * (numSplits - 1)
        val splitSizes = Basic.concatenate(Seq(Basic.fill(shape = numSplits - 1)(splitSize), splitSizeLast(NewAxis)))
        val valuePacks = Basic.split(concatenated, splitSizes)

        // Ready to aggregate the repacked values.
        (valuePacks, towerShapes, towerSizes)
      }
    }).unzip3
    val packInformation = ConcatenateAndSplitPacker.PackInformation(
      allTowerShapes = packed._2,
      allTowerSizes = packed._3)
    (packed._1, Some(packInformation))
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
      packed: Seq[Seq[OutputLike]],
      packInformation: Option[ConcatenateAndSplitPacker.PackInformation]
  ): Seq[Seq[OutputLike]] = packInformation match {
    case None => throw InvalidArgumentException("Cannot unpack values because no pack information is provided.")
    case Some(information) =>
      packed.zip(information.allTowerShapes.zip(information.allTowerSizes))
          .map {
            case (deviceValues, (shapes, sizes)) =>
              // Reverse the previous operations that `pack` applied, in order to convert the packed values back
              // into their original shapes.
              Op.colocateWith(Set(deviceValues.head.op)) {
                // Concatenate the packed values into a big flat tensor.
                val concatenatedDeviceValues = Basic.concatenate(deviceValues.map(_.toOutput))
                // Split the tensors back into their original sizes.
                val splitValues = Basic.split(concatenatedDeviceValues, Basic.stack(sizes))
                // Reshape the tensors back into their original shapes.
                splitValues.zip(shapes).map(vs => Basic.reshape(vs._1, vs._2))
              }
          }
  }
}

object ConcatenateAndSplitPacker {
  def apply(numPacks: Int = 1): ConcatenateAndSplitPacker = {
    new ConcatenateAndSplitPacker(numPacks)
  }

  /** Contains information collected while packing that is necessary for unpacking.
    *
    * @param  allTowerShapes Shapes of the values for all towers.
    * @param  allTowerSizes  Sizes of the values for all towers.
    */
  case class PackInformation(
      allTowerShapes: Seq[Seq[Output]],
      allTowerSizes: Seq[Seq[Output]])
}
