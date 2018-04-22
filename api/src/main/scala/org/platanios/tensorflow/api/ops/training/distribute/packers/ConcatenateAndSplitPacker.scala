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
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable

/** Gradients packer that concatenates all gradients together and then splits them into packs for reduction.
  *
  * @param  numPacks Number of packs to split the gradients into.
  * @throws InvalidArgumentException If `numPacks` is less than 1.
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[InvalidArgumentException]
class ConcatenateAndSplitPacker protected(val numPacks: Int)
    extends Packer[ConcatenateAndSplitPacker.PackInformation] {
  if (numPacks < 1)
    throw InvalidArgumentException(s"'numPacks' must be at least 1, but was set to $numPacks.")

  /** Packs the gradients.
    *
    * @param  groupedGradientsAndVariables Grouped gradients and variables (per device).
    * @return Packed gradients, ready for reduction, along with information that is necessary for unpacking later on.
    * @throws InvalidArgumentException If the provided grouped gradients and variables are inconsistent in some way.
    */
  @throws[InvalidArgumentException]
  override def pack(
      groupedGradientsAndVariables: Seq[Seq[(Output, Variable)]]
  ): (Seq[Seq[Output]], Option[ConcatenateAndSplitPacker.PackInformation]) = {
    val packed = groupedGradientsAndVariables.map(gradientsAndVariables => {
      Op.colocateWith(Set(gradientsAndVariables.head._1.op)) {
        // Flatten all the gradients.
        val flatGradients = gradientsAndVariables.map(gv => Basic.reshape(gv._1, Shape(-1)))
        // Remember the original shapes and sizes of all the gradients.
        val towerShapes = gradientsAndVariables.map(gv => Basic.shape(gv._1))
        val towerSizes = gradientsAndVariables.map(gv => Basic.size(gv._1))
        // Concatenate all the flat gradients into a big flat tensor.
        val concatenatedGradients = Basic.concatenate(flatGradients, axis = 0)

        // Split the concatenated tensor into packs. In cases where the total size is not divisible by `numPacks`, the
        // last pack gets more elements.
        // TODO: [DISTRIBUTE] It is also possible to optimize away the concatenation.
        val numSplits = Basic.constant(numPacks, name = "NumSplits")
        val totalGradientSize = Basic.size(concatenatedGradients)
        val splitSize = Math.truncateDivide(totalGradientSize, numSplits)
        val splitSizeLast = totalGradientSize - splitSize * (numSplits - 1)
        val splitSizes = Basic.concatenate(Seq(Basic.fill(shape = numSplits - 1)(splitSize), splitSizeLast(NewAxis)))
        val gradientPacks = Basic.split(concatenatedGradients, splitSizes)

        // Ready to aggregate the repacked gradients.
        (gradientPacks, towerShapes, towerSizes)
      }
    }).unzip3
    val packInformation = ConcatenateAndSplitPacker.PackInformation(
      groupedGradientsAndVariables = groupedGradientsAndVariables,
      allTowerShapes = packed._2,
      allTowerSizes = packed._3)
    (packed._1, Some(packInformation))
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
      packInformation: Option[ConcatenateAndSplitPacker.PackInformation]
  ): Seq[Seq[(Output, Variable)]] = packInformation match {
    case None => throw InvalidArgumentException("Cannot unpack gradients because no pack information is provided.")
    case Some(information) =>
      packedGradients.zip(information.groupedGradientsAndVariables)
          .zip(information.allTowerShapes.zip(information.allTowerSizes))
          .map {
            case ((deviceGradients, gradientsAndVariables), (shapes, sizes)) =>
              // Reverse the previous operations that `pack` applied, in order to convert the summed gradients back
              // into their original shapes.
              Op.colocateWith(Set(deviceGradients.head._1.op)) {
                // Concatenate the summed gradient packs into a big flat tensor.
                val concatenatedDeviceGradients = Basic.concatenate(deviceGradients.map(_._1))
                // Split the tensors back into their original sizes.
                val splitGradients = Basic.split(concatenatedDeviceGradients, Basic.stack(sizes))
                // Reshape the tensors back into their original shapes.
                val reshapedGradients = splitGradients.zip(shapes).map(gs => Basic.reshape(gs._1, gs._2))
                // Form the original sequence of gradients and variables using the reshaped gradients.
                reshapedGradients.zip(gradientsAndVariables).map(ggv => (ggv._1, ggv._2._2))
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
    * @param  groupedGradientsAndVariables Grouped gradients and variables that were packed.
    * @param  allTowerShapes               Shapes of the gradients for all towers.
    * @param  allTowerSizes                Sizes of the gradients for all towers.
    */
  case class PackInformation(
      groupedGradientsAndVariables: Seq[Seq[(Output, Variable)]],
      allTowerShapes: Seq[Seq[Output]],
      allTowerSizes: Seq[Seq[Output]])
}
