/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.ops.Basic.{ConstantPadding, PaddingMode}
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import scala.language.postfixOps
import scala.util.DynamicVariable

/** Contains functions for executing ops related to basic tensor manipulation.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Basic {
  //region Tensor Shape Ops

  /** $OpDocBasicRank
    *
    * @group BasicOps
    * @param  input Tensor whose rank to return.
    * @return Result as a new tensor.
    */
  def rank[T <: TensorLike](input: T): Tensor = {
    input match {
      case t: Tensor => Tensor.fill(INT32, Shape())(t.rank)
      case t: TensorIndexedSlices => size(t.denseShape)
      case t: SparseTensor => size(t.denseShape)
    }
  }

  /** $OpDocBasicSize
    *
    * @group BasicOps
    * @param  input    Tensor whose size to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @return Result as a new tensor.
    */
  def size[T <: TensorLike](input: T, dataType: DataType = INT32): Tensor = {
    input match {
      case t: Tensor => Tensor.fill(dataType, Shape())(t.size)
      case t: TensorIndexedSlices => Math.prod(Math.cast(t.denseShape, dataType), Array(0))
      case t: SparseTensor => Math.prod(Math.cast(t.denseShape, dataType), Array(0))
    }
  }

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input    Tensor whose shape to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @return Result as a new tensor.
    */
  def shape[T <: TensorLike](input: T, dataType: DataType = INT32): Tensor = {
    input match {
      case t: Tensor => t.shape.toTensor(dataType)
      case t: TensorIndexedSlices => Math.cast(t.denseShape, dataType)
      case t: SparseTensor => Math.cast(t.denseShape, dataType)
    }
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs   Tensors whose shapes to return.
    * @param  dataType Optional data type to use for the outputs of this op.
    * @return Result as a sequence of new tensors.
    */
  def shapeN(inputs: Seq[Tensor], dataType: DataType = INT32): Seq[Tensor] = {
    inputs.map(_.shape.toTensor(dataType))
  }

  //endregion Tensor Shape Ops

  //region Tensor Manipulation Ops

  /** $OpDocBasicExpandDims
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  axis  Dimension index at which to expand the shape of `input`.
    * @return Result as a new tensor.
    */
  def expandDims(input: Tensor, axis: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.expandDims(context.value.nativeHandle, input.nativeHandle, axis.nativeHandle))
  }

  /** $OpDocBasicSqueeze
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  axes  Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
    *               will be squeezed.
    * @return Result as a new tensor.
    */
  def squeeze(input: Tensor, axes: Seq[Int] = null)(implicit context: DynamicVariable[Context]): Tensor = {
    val longAxes: Array[Long] = if (axes == null) null else axes.map(_.toLong).toArray
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.squeeze(context.value.nativeHandle, input.nativeHandle, longAxes))
  }

  /** $OpDocBasicStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @param  axis   Dimension along which to stack the input tensors.
    * @return Result as a new tensor.
    */
  def stack(inputs: Seq[Tensor], axis: Int = 0)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.pack(context.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis))
  }

  /** $OpDocBasicParallelStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @return Result as a new tensor.
    */
  def parallelStack(inputs: Array[Tensor])(implicit context: DynamicVariable[Context]): Tensor = {
    val outputShape = Shape(inputs.length).concatenateWith(inputs.head.shape)
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.parallelConcat(
        context.value.nativeHandle, inputs.map(_.nativeHandle), outputShape.asArray.map(_.toLong)))
  }

  /** $OpDocBasicUnstack
    *
    * @group BasicOps
    * @param  input  Rank `R > 0` `Tensor` to be unstacked.
    * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
    * @param  axis   Dimension along which to unstack the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def unstack(
      input: Tensor, number: Int = -1, axis: Int = 0)(implicit context: DynamicVariable[Context]): Seq[Tensor] = {
    val num: Int = if (number >= 0) number else input.shape(axis)
    NativeTensorOpsBasic.unpack(context.value.nativeHandle, input.nativeHandle, num, axis).map(Tensor.fromNativeHandle)
  }

  /** $OpDocBasicConcatenate
    *
    * @group BasicOps
    * @param  inputs Input tensors to be concatenated.
    * @param  axis   Dimension along which to concatenate the input tensors.
    * @return Result as a new tensor.
    */
  def concatenate(inputs: Seq[Tensor], axis: Tensor = Tensor(0))(implicit context: DynamicVariable[Context]): Tensor = {
    if (inputs.length == 1)
      inputs.head
    else
      Tensor.fromNativeHandle(
        NativeTensorOpsBasic.concatV2(
          context.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis.nativeHandle))
  }

  /** $OpDocBasicConcatenateOffset
    *
    * @group BasicOps
    * @param  shapes Sequence of `N` `INT32` vectors representing the shapes of the tensors being concatenated.
    * @param  axis   `INT32` scalar representing the dimension along which to concatenate.
    * @return Sequence of `N` `INT32` vectors representing the starting offset of the input tensors within the
    *         concatenated tensor.
    */
  private[ops] def concatenateOffset(
      shapes: Seq[Tensor], axis: Tensor)(implicit context: DynamicVariable[Context]): Seq[Tensor] = {
    NativeTensorOpsBasic.concatOffset(context.value.nativeHandle, axis.nativeHandle, shapes.map(_.nativeHandle).toArray)
        .map(Tensor.fromNativeHandle)
  }

  /** $OpDocBasicSplitEvenly
    *
    * @group BasicOps
    * @param  input     Input tensor to split.
    * @param  numSplits Number of splits to obtain along the `axis` dimension.
    * @param  axis      Dimension along which to split the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def splitEvenly(
      input: Tensor, numSplits: Int, axis: Tensor = 0)(implicit context: DynamicVariable[Context]): Seq[Tensor] = {
    NativeTensorOpsBasic.split(context.value.nativeHandle, axis.nativeHandle, input.nativeHandle, numSplits.toLong)
        .map(Tensor.fromNativeHandle)
  }

  /** $OpDocBasicSplit
    *
    * @group BasicOps
    * @param  input      Input tensor to split.
    * @param  splitSizes Sizes for the splits to obtain.
    * @param  axis       Dimension along which to split the input tensor.
    * @return Result as a new tensor.
    */
  def split(
      input: Tensor, splitSizes: Tensor, axis: Tensor = 0)(implicit context: DynamicVariable[Context]): Seq[Tensor] = {
    NativeTensorOpsBasic.splitV(
      context.value.nativeHandle, input.nativeHandle, splitSizes.nativeHandle, axis.nativeHandle, splitSizes.shape(0))
        .map(Tensor.fromNativeHandle)
  }

  /** $OpDocBasicTile
    *
    * @group BasicOps
    * @param  input     Tensor to tile.
    * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
    *                   of `input`.
    * @return Result as a new tensor.
    */
  def tile(input: Tensor, multiples: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.tile(context.value.nativeHandle, input.nativeHandle, multiples.nativeHandle))
  }

  /** $OpDocBasicPad
    *
    * @group BasicOps
    * @param  input    Input tensor to be padded.
    * @param  paddings `INT32` or `INT64` tensor containing the paddings.
    * @param  mode     Padding mode to use.
    * @return Result as a new tensor.
    */
  def pad(
      input: Tensor, paddings: Tensor, mode: PaddingMode = ConstantPadding)(
      implicit context: DynamicVariable[Context]): Tensor = {
    mode.pad(input, paddings)
  }

  /** $OpDocBasicReshape
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  shape Shape of the output tensor.
    * @return Result as a new tensor.
    */
  def reshape(input: Tensor, shape: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.reshape(context.value.nativeHandle, input.nativeHandle, shape.nativeHandle))
  }

  /** $OpDocBasicTranspose
    *
    * @group BasicOps
    * @param  input       Input tensor to transpose.
    * @param  permutation Permutation of the input tensor dimensions.
    * @return Result as a new tensor.
    */
  def transpose(input: Tensor, permutation: Tensor = null)(implicit context: DynamicVariable[Context]): Tensor = {
    if (permutation == null) {
      val inputRank = rank(input)
      val reversePermutation = inputRank - 1 - Math.range(0, inputRank, 1)
      Tensor.fromNativeHandle(
        NativeTensorOpsBasic.transpose(
          context.value.nativeHandle, input.nativeHandle, reversePermutation.nativeHandle))
    } else {
      Tensor.fromNativeHandle(
        NativeTensorOpsBasic.transpose(
          context.value.nativeHandle, input.nativeHandle, permutation.nativeHandle))
    }
  }

  /** $OpDocBasicMatrixTranspose
    *
    * @group BasicOps
    * @param  input Input tensor to transpose.
    * @return Result as a new tensor.
    */
  def matrixTranspose(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    val inputRank = input.rank
    if (inputRank < 2)
      throw InvalidShapeException(s"'input' should be a (batch) matrix, with rank > 2. Found shape '${input.shape}'.")
    val permutation = Range(0, inputRank - 2).toArray ++ Array(inputRank - 1, inputRank - 2)
    transpose(input, permutation)
  }

  /** $OpDocBasicInvertPermutation
    *
    * @group BasicOps
    * @param  input One-dimensional [[INT32]] or [[INT64]] input tensor.
    * @return Result as a new tensor.
    */
  def invertPermutation(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.invertPermutation(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicReverse
    *
    * @group BasicOps
    * @param  input Input tensor to reverse. It must have rank at most 8.
    * @param  axes  Dimensions of the input tensor to reverse. Has to be [[INT32]] or [[INT64]].
    * @return Result as a new tensor which has the same shape as `input`.
    */
  def reverse(input: Tensor, axes: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.reverseV2(context.value.nativeHandle, input.nativeHandle, axes.nativeHandle))
  }

  /** $OpDocBasicReverseSequence
    *
    * @group BasicOps
    * @param  input           Input tensor to reverse.
    * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
    *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
    * @param  sequenceAxis    Tensor dimension which is partially reversed.
    * @param  batchAxis       Tensor dimension along which the reversal is performed.
    * @return Result as a new tensor which has the same shape as `input`.
    */
  def reverseSequence(
      input: Tensor, sequenceLengths: Tensor, sequenceAxis: Int, batchAxis: Int = 0)(
      implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.reverseSequence(
        context.value.nativeHandle, input.nativeHandle, sequenceLengths.nativeHandle, sequenceAxis, batchAxis))
  }

  /** $OpDocBasicSpaceToBatch
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  paddings  `2`-dimensional [[INT32]] or [[INT64]] tensor containing non-negative integers with shape
    *                   `[2, 2]`.
    * @return Result as a new tensor.
    */
  def spaceToBatch(
      input: Tensor, blockSize: Int, paddings: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    spaceToBatchND(input, Tensor(blockSize, blockSize), paddings)
  }

  /** $OpDocBasicSpaceToBatchND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape One-dimensional [[INT32]] or [[INT64]] tensor with shape `[M]` whose elements must all be
    *                    `>= 1`.
    * @param  paddings   Two-dimensional [[INT32]] or [[INT64]] tensor with shape `[M, 2]` whose elements must all be
    *                    non-negative. `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension
    *                    `i + 1`, which corresponds to spatial dimension `i`. It is required that `blockShape(i)`
    *                    divides `inputShape(i + 1) + padStart + padEnd`.
    * @return Result as a new tensor.
    */
  def spaceToBatchND(
      input: Tensor, blockShape: Tensor, paddings: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.spaceToBatchND(
        context.value.nativeHandle, input.nativeHandle, blockShape.nativeHandle, paddings.nativeHandle))
  }

  /** $OpDocBasicBatchToSpace
    *
    * @group BasicOps
    *
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  crops     `2`-dimensional [[INT32]] or [[INT64]] tensor containing non-negative integers with shape
    *                   `[2, 2]`.
    * @return Result as a new tensor.
    */
  def batchToSpace(input: Tensor, blockSize: Int, crops: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    batchToSpaceND(input, Tensor(blockSize, blockSize), crops)
  }

  /** $OpDocBasicBatchToSpaceND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape One-dimensional [[INT32]] or [[INT64]] tensor with shape `[M]` whose elements must all be
    *                    `>= 1`.
    * @param  crops      Two-dimensional [[INT32]] or [[INT64]] tensor with shape `[M, 2]` whose elements must all be
    *                    non-negative. `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input
    *                    dimension `i + 1`, which corresponds to spatial dimension `i`. It is required that
    *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
    * @return Result as a new tensor.
    */
  def batchToSpaceND(
      input: Tensor, blockShape: Tensor, crops: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.batchToSpaceND(
        context.value.nativeHandle, input.nativeHandle, blockShape.nativeHandle, crops.nativeHandle))
  }

  /** $OpDocBasicRequiredSpaceToBatchPaddingsAndCrops
    *
    * @group BasicOps
    * @param  inputShape   `INT32` tensor with shape `[N]`.
    * @param  blockShape   `INT32` tensor with shape `[N]`.
    * @param  basePaddings Optional `INT32` tensor with shape `[N, 2]` that specifies the minimum amount of padding to
    *                      use. All elements must be non-negative. Defaults to a tensor containing all zeros.
    * @return Tuple containing the paddings and crops required.
    * @throws IllegalArgumentException If `inputShape`, `blockShape`, or `basePaddings`, has invalid data type or shape.
    */
  @throws[IllegalArgumentException]
  def requiredSpaceToBatchPaddingsAndCrops(
      inputShape: Tensor, blockShape: Tensor, basePaddings: Tensor = null)(
      implicit context: DynamicVariable[Context]): (Tensor, Tensor) = {
    if (inputShape.dataType != INT32 || (inputShape.rank != -1 && inputShape.rank != 1))
      throw new IllegalArgumentException(
        s"'inputShape' (dataType = ${inputShape.dataType}, shape = ${inputShape.shape}) " +
            s"must be an INT32 one-dimensional tensor.")
    if (blockShape.dataType != INT32 || (blockShape.rank != -1 && blockShape.rank != 1))
      throw new IllegalArgumentException(
        s"'blockShape' (dataType = ${blockShape.dataType}, shape = ${blockShape.shape}) " +
            s"must be an INT32 one-dimensional tensor.")
    if (basePaddings != null && (basePaddings.dataType != INT32 || (basePaddings.rank != -1 && basePaddings.rank != 2)))
      throw new IllegalArgumentException(
        s"'basePaddings' (dataType = ${basePaddings.dataType}, shape = ${basePaddings.shape}) " +
            s"must be an INT32 two-dimensional tensor, or 'null'.")
    blockShape.shape.assertFullyDefined()
    blockShape.shape.assertHasRank(1)
    val numBlockDims = blockShape.shape(0)
    if (numBlockDims == 0) {
      (Tensor.zeros(INT32, Shape(0, 2)), Tensor.zeros(INT32, Shape(0, 2)))
    } else {
      inputShape.shape.assertIsCompatibleWith(Shape(numBlockDims))
      val actualBasePaddings = {
        if (basePaddings != null) {
          basePaddings.shape.assertIsCompatibleWith(Shape(numBlockDims, 2))
          basePaddings
        } else {
          Tensor.zeros(INT32, Shape(numBlockDims, 2))
        }
      }
      val padStart = actualBasePaddings(::, 0)
      val originalPadEnd = actualBasePaddings(::, 1)
      val fullInputShape = inputShape + padStart + originalPadEnd
      val extraPadEnd = (blockShape - (fullInputShape % blockShape)) % blockShape
      val padEnd = originalPadEnd + extraPadEnd
      val resultPaddings = stack((0 until numBlockDims).map(i => concatenate(Seq(padStart(i), padEnd(i)))))
      val zero = Tensor(padStart.dataType, 0)
      val resultCrops = stack((0 until numBlockDims).map(i => concatenate(Seq(zero, extraPadEnd(i)))))
      (resultPaddings, resultCrops)
    }
  }

  /** $OpDocBasicSpaceToDepth
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @return Result as a new tensor.
    */
  def spaceToDepth(input: Tensor, blockSize: Int)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.spaceToDepth(context.value.nativeHandle, input.nativeHandle, blockSize))
  }

  /** $OpDocBasicDepthToSpace
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @return Result as a new tensor.
    */
  def depthToSpace(input: Tensor, blockSize: Int)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.depthToSpace(context.value.nativeHandle, input.nativeHandle, blockSize))
  }

  //endregion Tensor Manipulation Ops

  //region Tensor Masking Ops

  /** $OpDocBasicWhere
    *
    * @group BasicOps
    * @param  input Input boolean tensor.
    * @return Result as a new tensor.
    */
  def where(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.where(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicBooleanMask
    *
    * @group BasicOps
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
    * @return Result as a new tensor.
    */
  def booleanMask(input: Tensor, mask: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    val inputShape: Shape = input.shape
    val maskShape: Shape = mask.shape
    val maskRank: Int = maskShape.rank
    val leadingSize = Math.prod(input.shape(0 :: maskRank), Array(0))
    val reshapedInput = reshape(input, concatenate(Array[Tensor](leadingSize, input.shape(maskRank ::)), 0))
    val firstDimension = inputShape(0 :: maskRank).rank
    reshapedInput.setShape(Shape(firstDimension).concatenateWith(inputShape(maskRank ::)))
    gather(reshapedInput, squeeze(where(reshape(mask, Array(-1))), axes = Array(1)))
  }

  /** $OpDocBasicSequenceMask
    *
    * @group BasicOps
    *
    * @param  lengths   One-dimensional integer tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @param  dataType  Data type for the output tensor.
    * @return Result as a new tensor.
    */
  def sequenceMask(
      lengths: Tensor, maxLength: Tensor = null, dataType: DataType = BOOLEAN)(
      implicit context: DynamicVariable[Context]): Tensor = {
    val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
    // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
    // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
    // matrix of size [lengths.shape(0), maxLen].
    val rowVector = Math.range(Tensor(maxLen.dataType, 0), maxLen, Tensor(maxLen.dataType, 1))
    // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
    // into INT32, then so do the elements of 'lengths'.
    val matrix = Math.cast(expandDims(lengths, 1), maxLen.dataType)
    val result = Math.less(rowVector, matrix)
    if (result.dataType == dataType)
      result
    else
      Math.cast(result, dataType)
  }

  /** $OpDocBasicIndexedSlicesMask
    *
    * @group BasicOps
    *
    * @param  input       Input indexed slices.
    * @param  maskIndices One-dimensional tensor containing the indices of the elements to mask.
    * @return Result as a new tensor indexed slices object.
    */
  @throws[IllegalArgumentException]
  def indexedSlicesMask(
      input: TensorIndexedSlices, maskIndices: Tensor)(
      implicit context: DynamicVariable[Context]): TensorIndexedSlices = {
    val (outputIndices, toGather) = listDiff(input.indices, maskIndices)
    val outputValues = gather(input.values, toGather)
    TensorIndexedSlices(indices = outputIndices, values = outputValues, denseShape = input.denseShape)
  }

  //endregion Tensor Masking Ops

  //region Tensor Counting and Set Ops

  /** $OpDocBasicUnique
    *
    * @group BasicOps
    *
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices. Must be [[INT32]] or [[INT64]].
    * @return Tuple containing `output` and `indices`.
    */
  def unique(input: Tensor, indicesDataType: DataType = INT32)(
      implicit context: DynamicVariable[Context]): (Tensor, Tensor) = {
    val tensors = NativeTensorOpsBasic.unique(
      context.value.nativeHandle, input.nativeHandle, indicesDataType.cValue).map(Tensor.fromNativeHandle)
    (tensors(0), tensors(1))
  }

  /** $OpDocBasicUniqueWithCounts
    *
    * @group BasicOps
    *
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices. Must be [[INT32]] or [[INT64]].
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts(input: Tensor, indicesDataType: DataType = INT32)(
      implicit context: DynamicVariable[Context]): (Tensor, Tensor, Tensor) = {
    val tensors = NativeTensorOpsBasic.uniqueWithCounts(
      context.value.nativeHandle, input.nativeHandle, indicesDataType.cValue).map(Tensor.fromNativeHandle)
    (tensors(0), tensors(1), tensors(2))
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    *
    * @param  x               One-dimensional tensor containing the values to keep.
    * @param  y               One-dimensional tensor containing the values to remove.
    * @param  indicesDataType Data type to use for the output indices of this op. Must be [[INT32]] or [[INT64]].
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff(
      x: Tensor, y: Tensor, indicesDataType: DataType = INT32)(
      implicit context: DynamicVariable[Context]): (Tensor, Tensor) = {
    val tensors = NativeTensorOpsBasic.listDiff(
      context.value.nativeHandle, x.nativeHandle, y.nativeHandle, indicesDataType.cValue).map(Tensor.fromNativeHandle)
    (tensors(0), tensors(1))
  }

  //endregion Tensor Counting and Set Ops

  //region Tensor Slicing Ops

  /** $OpDocBasicGather
    *
    * @group BasicOps
    *
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  axis    Tensor containing the axis along which to gather.
    * @return Result as a new tensor.
    */
  def gather(input: Tensor, indices: Tensor, axis: Tensor = 0)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.gatherV2(
        context.value.nativeHandle, input.nativeHandle, indices.nativeHandle, axis.nativeHandle))
  }

  /** $OpDocBasicGatherND
    *
    * @group BasicOps
    *
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
    *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    */
  def gatherND(input: Tensor, indices: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.gatherNd(context.value.nativeHandle, input.nativeHandle, indices.nativeHandle))
  }

  /** $OpDocBasicScatterND
    *
    * @group BasicOps
    *
    * @param  indices Indices tensor (must have `INT32` or `INT64` data type).
    * @param  updates Updates to scatter into the output tensor.
    * @param  shape   One-dimensional `INT32` or `INT64` tensor specifying the shape of the output tensor.
    * @return Result as a new tensor.
    */
  def scatterND(indices: Tensor, updates: Tensor, shape: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.scatterNd(
        context.value.nativeHandle, indices.nativeHandle, updates.nativeHandle, shape.nativeHandle))
  }

  /** $OpDocBasicSlice
    *
    * @group BasicOps
    *
    * @param  input Tensor to slice.
    * @param  begin Begin index tensor (must have data type of `INT32` or `INT64`). `begin(i)` specifies the offset into
    *               the `i`th dimension of `input` to slice from.
    * @param  size  Slice size tensor (must have data type of `INT32` or `INT64`). `size(i)` specifies the number of
    *               elements of the `i`th dimension of `input` to slice. If `size(i) == -1`, then all the remaining
    *               elements in dimension `i` are included in the slice (i.e., this is equivalent to setting
    *               `size(i) = input.shape(i) - begin(i)`).
    * @return Result as a new tensor.
    */
  private[ops] def slice(
      input: Tensor, begin: Tensor, size: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.slice(context.value.nativeHandle, input.nativeHandle, begin.nativeHandle, size.nativeHandle))
  }

  /** $OpDocBasicStridedSlice
    *
    * @group BasicOps
    *
    * @param  input          Tensor to slice.
    * @param  begin          One-dimensional integer tensor. `begin(i)` specifies the begin offset into the `i`th range
    *                        specification. The exact dimension this corresponds to will be determined by context.
    *                        Out-of-bounds values will be silently clamped. If the `i`th bit of `beginMask` is `1`, then
    *                        `begin(i)` is ignored and the full range of the appropriate dimension is used instead.
    *                        Negative values causes indexing to start from the highest element.
    * @param  end            One-dimensional integer tensor. `end(i)` is like `begin(i)` with the exception that it
    *                        determines the end offset into the `i`th range specification, and that `endMask` is used to
    *                        determine full ranges.
    * @param  strides        One-dimensional integer tensor. `strides(i)` specifies the increment in the `i`th range
    *                        specification after extracting a given element. Negative indices will reverse the original
    *                        order. Out-of-bounds values are clamped to `[0, shape(i)) if slice(i) > 0` or
    *                        `[-1, shape(i) - 1] if slice(i) < 0`.
    * @param  beginMask      Integer value representing a bitmask where bit `i` being `1` means to ignore the begin
    *                        value and instead use the largest interval possible. At runtime `begin(i)` will be replaced
    *                        with `[0, shape(i) - 1) if stride(i) > 0` or `[-1, shape(i) - 1]` if `stride(i) < 0`.
    * @param  endMask        Integer value analogous to `beginMask`, but for specifying the end offset of the slice.
    * @param  ellipsisMask   Integer value representing a bitmask where bit `i` being `1` means that the `i`th position
    *                        is actually an ellipsis. At most one bit can be `1`. If `ellipsisMask == 0`, then an
    *                        implicit ellipsis mask with value `1 << (m + 1)` is provided. This means that
    *                        `foo(3 :: 5) == foo(3 :: 5, ---)`. An ellipsis implicitly creates as many range
    *                        specifications as necessary to fully specify the sliced range for every dimension. For
    *                        example, for a 4-dimensional tensor `foo` the slice `foo(2, ---, 5 :: 8)` implies
    *                        `foo(2, ::, ::, 5 :: 8)`.
    * @param  newAxisMask    Integer value representing a bitmask where bit `i` being `1` means that the `i`th range
    *                        specification creates a new dimension with size `1`. For example,
    *                        `foo(0 :: 4, NewAxis, 0 :: 2)` will produce a tensor with shape `[4, 1, 2]`.
    * @param  shrinkAxisMask Integer value representing a bitmask where bit `i` being `1` means that the `i`th range
    *                        specification should shrink the dimensionality. `begin` and `end` must imply a slice of
    *                        size `1` in the dimension. For example, in `foo(0 :: 4, 3, 0 :: 2)` would result in a
    *                        tensor with shape `[4, 2]`.
    * @return Result as a new tensor.
    */
  private[ops] def stridedSlice(
      input: Tensor, begin: Tensor, end: Tensor, strides: Tensor = null, beginMask: Long = 0, endMask: Long = 0,
      ellipsisMask: Long = 0, newAxisMask: Long = 0, shrinkAxisMask: Long = 0)(
      implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.stridedSlice(
        context.value.nativeHandle, input.nativeHandle, begin.nativeHandle, end.nativeHandle, strides.nativeHandle,
        beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask))
  }

  //endregion Tensor Slicing Ops

  //region Tensor Ungrouped Ops

  /** $OpDocBasicCheckNumerics
    *
    * @group BasicOps
    *
    * @param  input   Input tensor.
    * @param  message Prefix to print for the error message.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def checkNumerics(input: Tensor, message: String = "")(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.checkNumerics(context.value.nativeHandle, input.nativeHandle, message.getBytes()))
  }

  /** $OpDocBasicEditDistance
    *
    * @group BasicOps
    *
    * @param  hypothesis Sparse tensor that contains the hypothesis sequences.
    * @param  truth      Sparse tensor that contains the truth sequences.
    * @param  normalize  Optional boolean value indicating whether to normalize the Levenshtein distance by the
    *                    length of `truth`.
    * @return Result as a new tensor.
    */
  def editDistance(
      hypothesis: SparseTensor, truth: SparseTensor, normalize: Boolean = true)(
      implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.editDistance(
        context.value.nativeHandle,
        hypothesis.indices.nativeHandle,
        hypothesis.values.nativeHandle,
        hypothesis.denseShape.nativeHandle,
        truth.indices.nativeHandle,
        truth.values.nativeHandle,
        truth.denseShape.nativeHandle,
        normalize))
  }

  /** $OpDocBasicOneHot
    *
    * @group BasicOps
    *
    * @param  indices  Tensor containing the indices for the "on" values.
    * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
    * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
    *                  Defaults to the value `1` with type `dataType`.
    * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
    *                  Defaults to the value `0` with type `dataType`.
    * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
    * @param  dataType Data type of the output tensor. If not provided, the function will attempt to assume the data
    *                  type of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`,
    *                  or `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
    * @return Result as a new tensor.
    */
  def oneHot(
      indices: Tensor, depth: Tensor, onValue: Tensor = null, offValue: Tensor = null, axis: Int = -1,
      dataType: DataType = null)(implicit context: DynamicVariable[Context]): Tensor = {
    val inferredDataType = {
      if (dataType != null) {
        dataType
      } else {
        if (onValue != null)
          onValue.dataType
        else if (offValue != null)
          offValue.dataType
        else
          FLOAT32
      }
    }
    val actualOnValue = if (onValue != null) onValue else Tensor(inferredDataType, 1)
    val actualOffValue = if (offValue != null) offValue else Tensor(inferredDataType, 1)
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.oneHot(
        context.value.nativeHandle, indices.nativeHandle, depth.nativeHandle, actualOnValue.nativeHandle,
        actualOffValue.nativeHandle, axis))
  }

  //endregion Tensor Ungrouped Ops

  // TODO: Add support for all the quantization ops.
  // TODO: Add support for all the broadcasting ops.

  //region Tensor Gradient Ops

  /** $OpDocBasicStopGradient
    *
    * @group BasicOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def stopGradient(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.stopGradient(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicPreventGradient
    *
    * @group BasicOps
    *
    * @param  input   Input tensor.
    * @param  message Message to print along with the error.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def preventGradient(input: Tensor, message: String = "")(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsBasic.preventGradient(context.value.nativeHandle, input.nativeHandle, message.getBytes()))
  }

  //endregion Tensor Gradient Ops
}

private[api] object Basic extends Basic {
  private[ops] trait Implicits {
    implicit def tensorToBasicOps(value: Tensor): BasicOps = BasicOps(value)
    implicit def tensorConvertibleToBasicOps[T](value: T)(implicit f: (T) => Tensor): BasicOps = BasicOps(f(value))
  }

  case class BasicOps private[ops](tensor: Tensor) {
    //region Tensor Manipulation Ops

    /** $OpDocBasicExpandDims
      *
      * @group BasicOps
      *
      * @param  axis  Dimension index at which to expand the shape of this tensor.
      * @return Result as a new tensor.
      */
    def expandDims(axis: Tensor): Tensor = Basic.expandDims(tensor, axis)

    /** $OpDocBasicSqueeze
      *
      * @group BasicOps
      *
      * @param  axes  Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
      *               will be squeezed.
      * @return Result as a new tensor.
      */
    def squeeze(axes: Seq[Int] = null): Tensor = Basic.squeeze(tensor, axes)

    /** $OpDocBasicUnstack
      *
      * @group BasicOps
      *
      * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
      * @param  axis   Dimension along which to unstack the input tensor.
      * @return Result as a new tensor.
      */
    def unstack(number: Int, axis: Int = 0): Seq[Tensor] = Basic.unstack(tensor, number, axis)

    /** $OpDocBasicSplitEvenly
      *
      * @group BasicOps
      *
      * @param  numSplits Number of splits to obtain along the `axis` dimension.
      * @param  axis      Dimension along which to split the input tensor.
      * @return Result as a sequence of new tensors.
      */
    def splitEvenly(numSplits: Int, axis: Tensor = 0): Seq[Tensor] = Basic.splitEvenly(tensor, numSplits, axis)

    /** $OpDocBasicSplit
      *
      * @group BasicOps
      *
      * @param  splitSizes Sizes for the splits to obtain.
      * @param  axis       Dimension along which to split the input tensor.
      * @return Result as a new tensor.
      */
    def split(splitSizes: Tensor, axis: Tensor = 0): Seq[Tensor] = Basic.split(tensor, splitSizes, axis)

    /** $OpDocBasicTile
      *
      * @group BasicOps
      *
      * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
      *                   of `input`.
      * @return Result as a new tensor.
      */
    def tile(multiples: Tensor): Tensor = Basic.tile(tensor, multiples)

    /** $OpDocBasicPad
      *
      * @group BasicOps
      *
      * @param  paddings `INT32` or `INT64` tensor containing the paddings.
      * @param  mode     Padding mode to use.
      * @return Result as a new tensor.
      */
    def pad(paddings: Tensor, mode: PaddingMode = ConstantPadding): Tensor = Basic.pad(tensor, paddings, mode)

    /** $OpDocBasicReshape
      *
      * @group BasicOps
      *
      * @param  shape Shape of the output tensor.
      * @return Result as a new tensor.
      */
    def reshape[T: TensorConvertible](shape: T): Tensor = Basic.reshape(tensor, shape)

    /** $OpDocBasicTranspose
      *
      * @group BasicOps
      *
      * @param  permutation Permutation of the input tensor dimensions.
      * @return Result as a new tensor.
      */
    def transpose(permutation: Tensor = null): Tensor = Basic.transpose(tensor, permutation)

    /** $OpDocBasicMatrixTranspose
      *
      * @group BasicOps
      *
      * @return Result as a new tensor.
      */
    def matrixTranspose: Tensor = Basic.matrixTranspose(tensor)

    /** $OpDocBasicInvertPermutation
      *
      * @group BasicOps
      *
      * @return Result as a new tensor.
      */
    def invertPermutation(): Tensor = Basic.invertPermutation(tensor)

    /** $OpDocBasicReverse
      *
      * @group BasicOps
      *
      * @param  axes  Dimensions of the input tensor to reverse. Has to be [[INT32]] or [[INT64]].
      * @return Result as a new tensor which has the same shape as `input`.
      */
    def reverse(axes: Tensor): Tensor = Basic.reverse(tensor, axes)

    /** $OpDocBasicReverseSequence
      *
      * @group BasicOps
      *
      * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
      *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
      * @param  sequenceAxis    Tensor dimension which is partially reversed.
      * @param  batchAxis       Tensor dimension along which the reversal is performed.
      * @return Result as a new tensor which has the same shape as `input`.
      */
    def reverseSequence(sequenceLengths: Tensor, sequenceAxis: Int, batchAxis: Int = 0): Tensor = {
      Basic.reverseSequence(tensor, sequenceLengths, sequenceAxis, batchAxis)
    }

    /** $OpDocBasicSpaceToBatch
      *
      * @group BasicOps
      *
      * @param  blockSize Block size which must be greater than `1`.
      * @param  paddings  `2`-dimensional [[INT32]] or [[INT64]] tensor containing non-negative integers with shape
      *                   `[2, 2]`.
      * @return Result as a new tensor.
      */
    def spaceToBatch(blockSize: Int, paddings: Tensor): Tensor = Basic.spaceToBatch(tensor, blockSize, paddings)

    /** $OpDocBasicSpaceToBatchND
      *
      * @group BasicOps
      *
      * @param  blockShape One-dimensional [[INT32]] or [[INT64]] tensor with shape `[M]` whose elements must all be
      *                    `>= 1`.
      * @param  paddings   Two-dimensional [[INT32]] or [[INT64]] tensor with shape `[M, 2]` whose elements must all be
      *                    non-negative. `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension
      *                    `i + 1`, which corresponds to spatial dimension `i`. It is required that `blockShape(i)`
      *                    divides `inputShape(i + 1) + padStart + padEnd`.
      * @return Result as a new tensor.
      */
    def spaceToBatchND(blockShape: Tensor, paddings: Tensor): Tensor = {
      Basic.spaceToBatchND(tensor, blockShape, paddings)
    }

    /** $OpDocBasicBatchToSpace
      *
      * @group BasicOps
      *
      * @param  blockSize Block size which must be greater than `1`.
      * @param  crops     `2`-dimensional [[INT32]] or [[INT64]] tensor containing non-negative integers with shape
      *                   `[2, 2]`.
      * @return Result as a new tensor.
      */
    def batchToSpace(blockSize: Int, crops: Tensor): Tensor = Basic.batchToSpace(tensor, blockSize, crops)

    /** $OpDocBasicBatchToSpaceND
      *
      * @group BasicOps
      *
      * @param  blockShape One-dimensional [[INT32]] or [[INT64]] tensor with shape `[M]` whose elements must all be
      *                    `>= 1`.
      * @param  crops      Two-dimensional [[INT32]] or [[INT64]] tensor with shape `[M, 2]` whose elements must all be
      *                    non-negative. `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input
      *                    dimension `i + 1`, which corresponds to spatial dimension `i`. It is required that
      *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
      * @return Result as a new tensor.
      */
    def batchToSpaceND(blockShape: Tensor, crops: Tensor): Tensor = Basic.batchToSpaceND(tensor, blockShape, crops)

    /** $OpDocBasicSpaceToDepth
      *
      * @group BasicOps
      *
      * @param  blockSize Block size which must be greater than `1`.
      * @return Result as a new tensor.
      */
    def spaceToDepth(blockSize: Int): Tensor = Basic.spaceToDepth(tensor, blockSize)

    /** $OpDocBasicDepthToSpace
      *
      * @group BasicOps
      *
      * @param  blockSize Block size which must be greater than `1`.
      * @return Result as a new tensor.
      */
    def depthToSpace(blockSize: Int): Tensor = Basic.depthToSpace(tensor, blockSize)

    //endregion Tensor Manipulation Ops

    //region Tensor Masking Ops

    /** $OpDocBasicWhere
      *
      * @group BasicOps
      *
      * @return Result as a new tensor.
      */
    def where(): Tensor = Basic.where(tensor)

    /** $OpDocBasicBooleanMask
      *
      * @group BasicOps
      *
      * @param  mask  `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
      * @return Result as a new tensor.
      */
    def booleanMask(mask: Tensor): Tensor = Basic.booleanMask(tensor, mask)

    /** $OpDocBasicSequenceMask
      *
      * @group BasicOps
      *
      * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum
      *                   value in this tensor.
      * @param  dataType  Data type for the output tensor.
      * @return Result as a new tensor.
      */
    def sequenceMask(maxLength: Tensor = null, dataType: DataType = BOOLEAN): Tensor = {
      Basic.sequenceMask(tensor, maxLength, dataType)
    }

    //endregion Tensor Masking Ops

    //region Tensor Counting and Set Ops

    /** $OpDocBasicUnique
      *
      * @group BasicOps
      *
      * @param  indicesDataType Data type of the returned indices. Must be [[INT32]] or [[INT64]].
      * @return Tuple containing `output` and `indices`.
      */
    def unique(indicesDataType: DataType = INT32): (Tensor, Tensor) = Basic.unique(tensor, indicesDataType)

    /** $OpDocBasicUniqueWithCounts
      *
      * @group BasicOps
      *
      * @param  indicesDataType Data type of the returned indices. Must be [[INT32]] or [[INT64]].
      * @return Tuple containing `output`, `indices`, and `counts`.
      */
    def uniqueWithCounts(indicesDataType: DataType = INT32): (Tensor, Tensor, Tensor) = {
      Basic.uniqueWithCounts(tensor, indicesDataType)
    }

    /** $OpDocBasicListDiff
      *
      * @group BasicOps
      *
      * @param  other           One-dimensional tensor containing the values to remove.
      * @param  indicesDataType Data type to use for the output indices of this op. Must be [[INT32]] or [[INT64]].
      * @return Tuple containing `output` and `indices`, from the method description.
      */
    def listDiff(other: Tensor, indicesDataType: DataType = INT32): (Tensor, Tensor) = {
      Basic.listDiff(tensor, other, indicesDataType)
    }

    //endregion Tensor Counting and Set Ops

    //region Tensor Slicing Ops

    /** $OpDocBasicGather
      *
      * @group BasicOps
      *
      * @param  indices Tensor containing indices to gather.
      * @param  axis    Tensor containing the axis along which to gather.
      * @return Result as a new tensor.
      */
    def gather(indices: Tensor, axis: Tensor = 0): Tensor = Basic.gather(tensor, indices, axis)

    /** $OpDocBasicGatherND
      *
      * @group BasicOps
      *
      * @param  indices Tensor containing indices to gather.
      * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
      *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
      */
    def gatherND(indices: Tensor): Tensor = Basic.gatherND(tensor, indices)

    /** $OpDocBasicScatterND
      *
      * @group BasicOps
      *
      * @param  updates Updates to scatter into the output tensor.
      * @param  shape   One-dimensional `INT32` or `INT64` tensor specifying the shape of the output tensor.
      * @return Result as a new tensor.
      */
    def scatterND(updates: Tensor, shape: Tensor): Tensor = Basic.scatterND(tensor, updates, shape)

    /** Creates an op that slices this op according to the provided indexers.
      *
      * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
      *
      * @param  indexers Sequence of indexers to use.
      * @return Created op.
      */
    def slice(indexers: Indexer*): Tensor = {
      val stridedSlice = Indexer.toStridedSlice(indexers: _*)
      val beginTensor: Tensor = stridedSlice._1
      val endTensor: Tensor = stridedSlice._2
      val stridesTensor: Tensor = stridedSlice._3
      val result = Basic.stridedSlice(
        tensor, beginTensor, endTensor, stridesTensor, stridedSlice._4, stridedSlice._5, stridedSlice._6,
        stridedSlice._7, stridedSlice._8)
      beginTensor.close()
      endTensor.close()
      stridesTensor.close()
      result
    }

    //endregion Tensor Slicing Ops

    //region Tensor Ungrouped Ops

    /** $OpDocBasicCheckNumerics
      *
      * @group BasicOps
      *
      * @param  message Prefix to print for the error message.
      * @return Result as a new tensor which has the same value as the input tensor.
      */
    def checkNumerics(message: String = ""): Tensor = Basic.checkNumerics(tensor)

    /** $OpDocBasicOneHot
      *
      * @group BasicOps
      *
      * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
      * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
      *                  Defaults to the value `1` with type `dataType`.
      * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
      *                  Defaults to the value `0` with type `dataType`.
      * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
      * @param  dataType Data type of the output tensor. If not provided, the function will attempt to assume the data
      *                  type of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`,
      *                  or `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
      * @return Result as a new tensor.
      */
    def oneHot(
        depth: Tensor, onValue: Tensor = null, offValue: Tensor = null, axis: Int = -1,
        dataType: DataType = null): Tensor = Basic.oneHot(tensor, depth, onValue, offValue, axis, dataType)

    //endregion Tensor Ungrouped Ops

    //region Tensor Gradient Ops

    /** $OpDocBasicStopGradient
      *
      * @group BasicOps
      *
      * @return Result as a new tensor which has the same value as this tensor.
      */
    def stopGradient(): Tensor = Basic.stopGradient(tensor)

    /** $OpDocBasicPreventGradient
      *
      * @group BasicOps
      *
      * @param  message Message to print along with the error.
      * @return Result as a new tensor which has the same value as this tensor.
      */
    def preventGradient(message: String = ""): Tensor = Basic.preventGradient(tensor, message)

    //endregion Tensor Gradient Ops
  }
}
