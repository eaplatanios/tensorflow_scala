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

package org.platanios.tensorflow.api.ops.basic

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices}

import scala.language.postfixOps

/** Contains ops related to masking tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Masking {
  /** $OpDocBasicWhere
    *
    * @group BasicOps
    * @param  input Input boolean tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def where[T: TF : IsBooleanOrNumeric](
      input: Output[T],
      name: String = "Where"
  ): Output[Long] = {
    Op.Builder[Output[T], Output[Long]](
      opType = "Where",
      name = name,
      input = input
    ).build().output
  }

  /** $OpDocBasicBooleanMask
    *
    * @group BasicOps
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional tensor, where `K <= N` and `K` must be known statically.
    * @param  name  Name for the created op output.
    * @return Created op output.
    */
  def booleanMask[T: TF](
      input: Output[T],
      mask: Output[Boolean],
      name: String = "BooleanMask"
  ): Output[T] = {
    Op.nameScope(name) {
      val inputShape = input.shape
      val maskShape = mask.shape
      val maskRank = maskShape.rank
      if (maskRank < 0)
        throw InvalidShapeException(
          "The rank of the boolean mask must be known, even if some dimension sizes are unknown. For example, " +
              "'Shape(-1)' is fine, but 'Shape.unknown()' is not.")
      inputShape(0 :: maskRank).assertIsCompatibleWith(maskShape)
      val dynamicInputShape = Manipulation.shape(input)
      val leadingSize = Math.prod(dynamicInputShape(0 :: maskRank), Seq(0)).reshape(Shape(1))
      val reshapedInput = Manipulation.reshape(
        input,
        Manipulation.concatenate(Seq(leadingSize, dynamicInputShape(maskRank ::)), 0))
      val firstDimension = inputShape(0 :: maskRank).numElements.toInt
      if (maskRank >= inputShape.rank)
        reshapedInput.setShape(Shape(firstDimension))
      else
        reshapedInput.setShape(Shape(firstDimension).concatenateWith(inputShape(maskRank ::)))
      Manipulation.gather(
        reshapedInput,
        Manipulation.squeeze(
          where(Manipulation.reshape(mask, Seq(-1))),
          axes = Seq(1)
        ), axis = 0)
    }
  }

  /** $OpDocBasicSequenceMask
    *
    * @group BasicOps
    * @param  lengths   One-dimensional tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @param  name      Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `maxLength` is not a scalar.
    */
  @throws[IllegalArgumentException]
  def sequenceMask[T: TF : IsIntOrUInt](
      lengths: Output[T],
      maxLength: Output[T] = null,
      name: String = "SequenceMask"
  ): Output[Boolean] = {
    require(maxLength == null || maxLength.rank == -1 || maxLength.rank == 0, "'maxLength' must be a scalar.")
    val ops = if (maxLength == null) Set(lengths.op) else Set(lengths.op, maxLength.op)
    Op.nameScope(name) {
      val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
      // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
      // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
      // matrix of size [lengths.shape(0), maxLen].
      val rowVector = Math.range(Basic.zerosLike(maxLen), maxLen, Basic.onesLike(maxLen))
      // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
      // into Int, then so do the elements of 'lengths'.
      val matrix = Manipulation.expandDims(lengths, -1).castTo[T]
      Math.less(rowVector, matrix)
    }
  }

  /** $OpDocBasicIndexedSlicesMask
    *
    * @group BasicOps
    * @param  input       Input indexed slices.
    * @param  maskIndices One-dimensional tensor containing the indices of the elements to mask.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def indexedSlicesMask[T: TF](
      input: OutputIndexedSlices[T],
      maskIndices: Output[Long],
      name: String = "IndexedSlicesMask"
  ): OutputIndexedSlices[T] = {
    Op.nameScope(name) {
      val (outputIndices, toGather) = listDiff(input.indices, maskIndices, indicesDataType = Long)
      val outputValues = Manipulation.gather(input.values, toGather, axis = 0)
      OutputIndexedSlices(indices = outputIndices, values = outputValues, denseShape = input.denseShape)
    }
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    * @param  x               One-dimensional tensor containing the values to keep.
    * @param  y               One-dimensional tensor containing the values to remove.
    * @param  indicesDataType Data type to use for the output indices of this op.
    * @param  name            Name for the created op.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff[T: TF, I: TF : IsInt32OrInt64](
      x: Output[T],
      y: Output[T],
      indicesDataType: DataType[I],
      name: String = "ListDiff"
  ): (Output[T], Output[I]) = {
    Op.Builder[(Output[T], Output[T]), (Output[T], Output[I])](
      opType = "ListDiff",
      name = name,
      input = (x, y)
    ).setAttribute("out_idx", indicesDataType)
        .build().output
  }
}

object Masking extends Masking
