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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.Basic.{ConstantPadding, PaddingMode}
import org.platanios.tensorflow.api.ops.NN.CNNDataFormat
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import java.nio.charset.StandardCharsets

import scala.language.postfixOps

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
  def rank[T <: TensorLike[_]](input: T): Tensor[INT32] = {
    input match {
      case t: Tensor[_] => Tensor.fill(INT32, Shape())(t.rank)
      case t: TensorIndexedSlices[_] => size(t.denseShape, INT32)
      case t: SparseTensor[_] => size(t.denseShape, INT32)
    }
  }

  /** $OpDocBasicSize
    *
    * @group BasicOps
    * @param  input Tensor whose size to return.
    * @return Result as a new tensor.
    */
  def size[T <: TensorLike[_]](input: T): Tensor[INT64] = size(input, INT64)

  /** $OpDocBasicSize
    *
    * @group BasicOps
    * @param  input    Tensor whose size to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @return Result as a new tensor.
    */
  def size[T <: TensorLike[_], DR <: ReducibleDataType](input: T, dataType: DR): Tensor[DR] = {
    input match {
      case t: Tensor[_] => Tensor.fill(dataType, Shape())(t.size)
      case t: TensorIndexedSlices[_] => Math.prod(Cast.cast(t.denseShape, dataType), Array(0))
      case t: SparseTensor[_] => Math.prod(Cast.cast(t.denseShape, dataType), Array(0))
    }
  }

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input Tensor whose shape to return.
    * @return Result as a new tensor.
    */
  def shape[T <: TensorLike[_]](input: T): Tensor[INT64] = shape(input, INT64)

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input    Tensor whose shape to return.
    * @param  dataType Optional data type to use for the output of this op.
    * @return Result as a new tensor.
    */
  def shape[T <: TensorLike[_], DR <: DataType](input: T, dataType: DR): Tensor[DR] = {
    input match {
      case t: Tensor[_] => t.shape.toTensor(dataType)
      case t: TensorIndexedSlices[_] => Cast.cast(t.denseShape, dataType)
      case t: SparseTensor[_] => Cast.cast(t.denseShape, dataType)
    }
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs Tensors whose shapes to return.
    * @return Result as a sequence of new tensors.
    */
  def shapeN(inputs: Seq[Tensor[_]]): Seq[Tensor[INT64]] = shapeN(inputs, INT64)

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs   Tensors whose shapes to return.
    * @param  dataType Optional data type to use for the outputs of this op.
    * @return Result as a sequence of new tensors.
    */
  def shapeN[DR <: DataType](inputs: Seq[Tensor[_]], dataType: DR): Seq[Tensor[DR]] = {
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
  def expandDims[D <: DataType](input: Tensor[D], axis: Tensor[INT32]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.expandDims(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle))
  }

  /** $OpDocBasicSqueeze
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  axes  Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
    *               will be squeezed.
    * @return Result as a new tensor.
    */
  def squeeze[D <: DataType](input: Tensor[D], axes: Seq[Int] = null): Tensor[D] = {
    val longAxes: Array[Long] = if (axes == null) null else axes.map(_.toLong).toArray
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.squeeze(
      executionContext.value.nativeHandle, input.nativeHandle, longAxes))
  }

  /** $OpDocBasicStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @param  axis   Dimension along which to stack the input tensors.
    * @return Result as a new tensor.
    */
  def stack[D <: DataType](inputs: Seq[Tensor[D]], axis: Int = 0): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.pack(
      executionContext.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis))
  }

  /** $OpDocBasicParallelStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @return Result as a new tensor.
    */
  def parallelStack[D <: DataType](inputs: Array[Tensor[D]]): Tensor[D] = {
    val outputShape = Shape(inputs.length).concatenateWith(inputs.head.shape)
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.parallelConcat(
      executionContext.value.nativeHandle, inputs.map(_.nativeHandle), outputShape.asArray.map(_.toLong)))
  }

  /** $OpDocBasicUnstack
    *
    * @group BasicOps
    * @param  input  Rank `R > 0` `Tensor` to be unstacked.
    * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
    * @param  axis   Dimension along which to unstack the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def unstack[D <: DataType](
      input: Tensor[D],
      number: Int = -1,
      axis: Int = 0
  ): Seq[Tensor[D]] = {
    val num: Int = if (number >= 0) number else input.shape(axis)
    NativeTensorOpsBasic.unpack(
      executionContext.value.nativeHandle, input.nativeHandle, num, axis).map(Tensor.fromNativeHandle[D])
  }

  /** $OpDocBasicConcatenate
    *
    * @group BasicOps
    * @param  inputs Input tensors to be concatenated.
    * @param  axis   Dimension along which to concatenate the input tensors.
    * @return Result as a new tensor.
    */
  def concatenate[D <: DataType](inputs: Seq[Tensor[D]], axis: Tensor[INT32] = 0): Tensor[D] = {
    if (inputs.lengthCompare(1) == 0)
      inputs.head
    else
      Tensor.fromNativeHandle[D](
        NativeTensorOpsBasic.concatV2(
          executionContext.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis.nativeHandle))
  }

  /** $OpDocBasicConcatenateOffset
    *
    * @group BasicOps
    * @param  shapes Sequence of `N` vectors representing the shapes of the tensors being concatenated.
    * @param  axis   Scalar representing the dimension along which to concatenate.
    * @return Sequence of `N` vectors representing the starting offset of the input tensors within the
    *         concatenated tensor.
    */
  private[ops] def concatenateOffset(shapes: Seq[Tensor[INT32]], axis: Tensor[INT32]): Seq[Tensor[INT32]] = {
    NativeTensorOpsBasic.concatOffset(
      executionContext.value.nativeHandle, axis.nativeHandle, shapes.map(_.nativeHandle).toArray)
        .map(Tensor.fromNativeHandle[INT32])
  }

  /** $OpDocBasicSplitEvenly
    *
    * @group BasicOps
    * @param  input     Input tensor to split.
    * @param  numSplits Number of splits to obtain along the `axis` dimension.
    * @param  axis      Dimension along which to split the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def splitEvenly[D <: DataType](input: Tensor[D], numSplits: Int, axis: Tensor[INT32] = 0): Seq[Tensor[D]] = {
    NativeTensorOpsBasic.split(
      executionContext.value.nativeHandle, axis.nativeHandle, input.nativeHandle, numSplits.toLong)
        .map(Tensor.fromNativeHandle[D])
  }

  /** $OpDocBasicSplit
    *
    * @group BasicOps
    * @param  input      Input tensor to split.
    * @param  splitSizes Sizes for the splits to obtain.
    * @param  axis       Dimension along which to split the input tensor.
    * @return Result as a new tensor.
    */
  def split[D <: DataType, I <: IntOrUInt](
      input: Tensor[D],
      splitSizes: Tensor[I],
      axis: Tensor[INT32] = 0
  ): Seq[Tensor[D]] = {
    NativeTensorOpsBasic.splitV(
      executionContext.value.nativeHandle, input.nativeHandle, splitSizes.nativeHandle,
      axis.nativeHandle, splitSizes.shape(0))
        .map(Tensor.fromNativeHandle[D])
  }

  /** $OpDocBasicTile
    *
    * @group BasicOps
    * @param  input     Tensor to tile.
    * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
    *                   of `input`.
    * @return Result as a new tensor.
    */
  def tile[D <: DataType, I <: Int32OrInt64](input: Tensor[D], multiples: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.tile(
      executionContext.value.nativeHandle, input.nativeHandle, multiples.nativeHandle))
  }

  /** $OpDocBasicPad
    *
    * @group BasicOps
    * @param  input    Input tensor to be padded.
    * @param  paddings Tensor containing the paddings.
    * @param  mode     Padding mode to use.
    * @return Result as a new tensor.
    */
  def pad[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      paddings: Tensor[I],
      mode: PaddingMode = ConstantPadding(Some(Tensor(0)))
  ): Tensor[D] = {
    mode.pad(input, paddings)
  }

  /** $OpDocBasicReshape
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  shape Shape of the output tensor.
    * @return Result as a new tensor.
    */
  def reshape[D <: DataType, I <: Int32OrInt64](input: Tensor[D], shape: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.reshape(
      executionContext.value.nativeHandle, input.nativeHandle, shape.nativeHandle))
  }

  /** $OpDocBasicTranspose
    *
    * @group BasicOps
    * @param  input       Input tensor to transpose.
    * @param  permutation Permutation of the input tensor dimensions.
    * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
    * @return Result as a new tensor.
    */
  def transpose[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      permutation: Tensor[I] = null,
      conjugate: Boolean = false
  ): Tensor[D] = {
    if (permutation == null) {
      val inputRank = rank(input)
      val reversePermutation = inputRank - 1 - Math.range(0, inputRank, 1)
      if (conjugate && input.dataType.isComplex)
        Tensor.fromNativeHandle[D](NativeTensorOpsBasic.conjugateTranspose(
          executionContext.value.nativeHandle, input.nativeHandle, reversePermutation.nativeHandle))
      else
        Tensor.fromNativeHandle[D](NativeTensorOpsBasic.transpose(
          executionContext.value.nativeHandle, input.nativeHandle, reversePermutation.nativeHandle))
    } else {
      if (conjugate && input.dataType.isComplex)
        Tensor.fromNativeHandle[D](NativeTensorOpsBasic.conjugateTranspose(
          executionContext.value.nativeHandle, input.nativeHandle, permutation.nativeHandle))
      else
        Tensor.fromNativeHandle[D](NativeTensorOpsBasic.transpose(
          executionContext.value.nativeHandle, input.nativeHandle, permutation.nativeHandle))
    }
  }

  /** $OpDocBasicMatrixTranspose
    *
    * @group BasicOps
    * @param  input     Input tensor to transpose.
    * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
    * @return Result as a new tensor.
    * @throws InvalidShapeException If the input tensor has rank <= 2.
    */
  @throws[InvalidShapeException]
  def matrixTranspose[D <: DataType](input: Tensor[D], conjugate: Boolean = false): Tensor[D] = {
    val inputRank = input.rank
    if (inputRank < 2)
      throw InvalidShapeException(s"'input' should be a (batch) matrix, with rank > 2. Found shape '${input.shape}'.")
    val permutation = Range(0, inputRank - 2).toArray ++ Array(inputRank - 1, inputRank - 2)
    transpose(input, permutation, conjugate)
  }

  /** $OpDocBasicInvertPermutation
    *
    * @group BasicOps
    * @param  input One-dimensional input tensor.
    * @return Result as a new tensor.
    */
  def invertPermutation[I <: Int32OrInt64](input: Tensor[I]): Tensor[I] = {
    Tensor.fromNativeHandle[I](NativeTensorOpsBasic.invertPermutation(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicReverse
    *
    * @group BasicOps
    * @param  input Input tensor to reverse. It must have rank at most 8.
    * @param  axes  Dimensions of the input tensor to reverse.
    * @return Result as a new tensor which has the same shape as `input`.
    */
  def reverse[D <: DataType, I <: Int32OrInt64](input: Tensor[D], axes: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.reverseV2(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle))
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
  def reverseSequence[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      sequenceLengths: Tensor[I],
      sequenceAxis: Int,
      batchAxis: Int = 0
  ): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.reverseSequence(
      executionContext.value.nativeHandle, input.nativeHandle, sequenceLengths.nativeHandle, sequenceAxis, batchAxis))
  }

  /** $OpDocBasicSpaceToBatch
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
    * @return Result as a new tensor.
    */
  def spaceToBatch[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      blockSize: Int,
      paddings: Tensor[I]
  ): Tensor[D] = {
    spaceToBatchND(input, Tensor(blockSize, blockSize), paddings)
  }

  /** $OpDocBasicSpaceToBatchND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
    * @param  paddings   Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
    *                    `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension `i + 1`, which
    *                    corresponds to spatial dimension `i`. It is required that `blockShape(i)` divides
    *                    `inputShape(i + 1) + padStart + padEnd`.
    * @return Result as a new tensor.
    */
  def spaceToBatchND[D <: DataType, I1 <: Int32OrInt64, I2 <: Int32OrInt64](
      input: Tensor[D],
      blockShape: Tensor[I1],
      paddings: Tensor[I2]
  ): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.spaceToBatchND(
      executionContext.value.nativeHandle, input.nativeHandle, blockShape.nativeHandle, paddings.nativeHandle))
  }

  /** $OpDocBasicBatchToSpace
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  crops     `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
    * @return Result as a new tensor.
    */
  def batchToSpace[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      blockSize: Int,
      crops: Tensor[I]
  ): Tensor[D] = {
    batchToSpaceND(input, Tensor(blockSize, blockSize), crops)
  }

  /** $OpDocBasicBatchToSpaceND
    *
    * @group BasicOps
    * @param  input      `N`-dimensional tensor with shape `inputShape = [batch] + spatialShape + remainingShape`, where
    *                    spatialShape has `M` dimensions.
    * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
    * @param  crops      Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
    *                    `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input dimension `i + 1`,
    *                    which corresponds to spatial dimension `i`. It is required that
    *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
    * @return Result as a new tensor.
    */
  def batchToSpaceND[D <: DataType, I1 <: Int32OrInt64, I2 <: Int32OrInt64](
      input: Tensor[D],
      blockShape: Tensor[I1],
      crops: Tensor[I2]
  ): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsBasic.batchToSpaceND(
      executionContext.value.nativeHandle, input.nativeHandle, blockShape.nativeHandle, crops.nativeHandle))
  }

  /** $OpDocBasicRequiredSpaceToBatchPaddingsAndCrops
    *
    * @group BasicOps
    * @param  inputShape   Tensor with shape `[N]`.
    * @param  blockShape   Tensor with shape `[N]`.
    * @param  basePaddings Optional tensor with shape `[N, 2]` that specifies the minimum amount of padding to use. All
    *                      elements must be non-negative. Defaults to a tensor containing all zeros.
    * @return Tuple containing the paddings and crops required.
    * @throws InvalidShapeException If `inputShape`, `blockShape`, or `basePaddings`, has invalid shape.
    */
  @throws[InvalidShapeException]
  def requiredSpaceToBatchPaddingsAndCrops(
      inputShape: Tensor[INT32],
      blockShape: Tensor[INT32],
      basePaddings: Tensor[INT32] = null
  ): (Tensor[INT32], Tensor[INT32]) = {
    if (inputShape.rank != -1 && inputShape.rank != 1)
      throw InvalidShapeException(s"'inputShape' (shape = ${inputShape.shape}) must be a one-dimensional tensor.")
    if (blockShape.rank != -1 && blockShape.rank != 1)
      throw InvalidShapeException(s"'blockShape' (shape = ${blockShape.shape}) must be a one-dimensional tensor.")
    if (basePaddings != null && (basePaddings.rank != -1 && basePaddings.rank != 2))
      throw InvalidShapeException(
        s"'basePaddings' (shape = ${basePaddings.shape}) must be a two-dimensional tensor, or 'null'.")
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
    * @param  input      `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize  Block size which must be greater than `1`.
    * @param  dataFormat Format of the input and output data.
    * @return Result as a new tensor.
    */
  def spaceToDepth[D <: DataType](
      input: Tensor[D],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.spaceToDepth(
      executionContext.value.nativeHandle, input.nativeHandle, blockSize,
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1)))
  }

  /** $OpDocBasicDepthToSpace
    *
    * @group BasicOps
    * @param  input      `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize  Block size which must be greater than `1`.
    * @param  dataFormat Format of the input and output data.
    * @return Result as a new tensor.
    */
  def depthToSpace[D <: DataType](
      input: Tensor[D],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.depthToSpace(
      executionContext.value.nativeHandle, input.nativeHandle, blockSize,
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1)))
  }

  //endregion Tensor Manipulation Ops

  //region Tensor Masking Ops

  /** $OpDocBasicWhere
    *
    * @group BasicOps
    * @param  input Input boolean tensor.
    * @return Result as a new tensor.
    */
  def where(input: Tensor[BOOLEAN]): Tensor[INT64] = {
    Tensor.fromNativeHandle[INT64](NativeTensorOpsBasic.where(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicBooleanMask
    *
    * @group BasicOps
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
    * @return Result as a new tensor.
    */
  def booleanMask[D <: DataType](input: Tensor[D], mask: Tensor[BOOLEAN]): Tensor[D] = {
    val maskShape: Shape = mask.shape
    val maskRank: Int = maskShape.rank
    val leadingSize = reshape(Math.prod(input.shape(0 :: maskRank), Seq(0)), Shape(1))
    val reshapedInput = reshape(input, concatenate(Seq(leadingSize, input.shape(maskRank ::).toTensor(INT64)), 0))
    gather(reshapedInput, squeeze(where(reshape(mask, Seq(-1))), axes = Seq(1)))
  }

  /** $OpDocBasicSequenceMask
    *
    * @group BasicOps
    * @param  lengths   One-dimensional integer tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @return Result as a new tensor.
    * @throws IllegalArgumentException If `maxLength` is not a scalar.
    */
  @throws[IllegalArgumentException]
  def sequenceMask[D <: NumericDataType](
      lengths: Tensor[D],
      maxLength: Tensor[D] = null
  ): Tensor[BOOLEAN] = {
    require(maxLength == null || maxLength.rank == -1 || maxLength.rank == 0, "'maxLength' must be a scalar.")
    val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
    // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
    // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
    // matrix of size [lengths.shape(0), maxLen].
    val rowVector = Math.range(Tensor.zeros(maxLen.dataType, Shape()), maxLen, Tensor.ones(maxLen.dataType, Shape()))
    // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
    // into INT32, then so do the elements of 'lengths'.
    val matrix = Cast.cast(expandDims(lengths, 1), maxLen.dataType)
    Math.less(rowVector, matrix)
  }

  /** $OpDocBasicIndexedSlicesMask
    *
    * @group BasicOps
    * @param  input       Input indexed slices.
    * @param  maskIndices One-dimensional tensor containing the indices of the elements to mask.
    * @return Result as a new tensor indexed slices object.
    */
  @throws[IllegalArgumentException]
  def indexedSlicesMask[D <: DataType](
      input: TensorIndexedSlices[D],
      maskIndices: Tensor[INT32]
  ): TensorIndexedSlices[D] = {
    val (outputIndices, toGather) = listDiff(input.indices, maskIndices)
    val outputValues = gather(input.values, toGather)
    TensorIndexedSlices(indices = outputIndices.toInt64, values = outputValues, denseShape = input.denseShape)
  }

  //endregion Tensor Masking Ops

  //region Tensor Counting and Set Ops

  /** $OpDocBasicUnique
    *
    * @group BasicOps
    * @param  input One-dimensional input tensor.
    * @return Tuple containing `output` and `indices`.
    */
  def unique[D <: DataType](input: Tensor[D]): (Tensor[D], Tensor[INT32]) = unique(input, INT32)

  /** $OpDocBasicUnique
    *
    * @group BasicOps
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices.
    * @return Tuple containing `output` and `indices`.
    */
  def unique[D <: DataType, I <: Int32OrInt64](input: Tensor[D], indicesDataType: I): (Tensor[D], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.unique(
      executionContext.value.nativeHandle, input.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[D](tensors(0)), Tensor.fromNativeHandle[I](tensors(1)))
  }

  /** $OpDocBasicUniqueWithCounts
    *
    * @group BasicOps
    * @param  input One-dimensional input tensor.
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts[D <: DataType](input: Tensor[D]): (Tensor[D], Tensor[INT32], Tensor[INT32]) = {
    uniqueWithCounts(input, INT32)
  }

  /** $OpDocBasicUniqueWithCounts
    *
    * @group BasicOps
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices.
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      indicesDataType: I
  ): (Tensor[D], Tensor[I], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.uniqueWithCounts(
      executionContext.value.nativeHandle, input.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[D](tensors(0)),
        Tensor.fromNativeHandle[I](tensors(1)),
        Tensor.fromNativeHandle[I](tensors(2)))
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    * @param  x One-dimensional tensor containing the values to keep.
    * @param  y One-dimensional tensor containing the values to remove.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff[D <: DataType](x: Tensor[D], y: Tensor[D]): (Tensor[D], Tensor[INT32]) = {
    listDiff(x, y, INT32)
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    * @param  x               One-dimensional tensor containing the values to keep.
    * @param  y               One-dimensional tensor containing the values to remove.
    * @param  indicesDataType Data type to use for the output indices of this op.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff[D <: DataType, I <: Int32OrInt64](
      x: Tensor[D],
      y: Tensor[D],
      indicesDataType: I
  ): (Tensor[D], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.listDiff(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[D](tensors(0)), Tensor.fromNativeHandle[I](tensors(1)))
  }

  //endregion Tensor Counting and Set Ops

  //region Tensor Slicing Ops

  /** $OpDocBasicGather
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  axis    Tensor containing the axis along which to gather.
    * @return Result as a new tensor.
    */
  def gather[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      indices: Tensor[I],
      axis: Tensor[I] = null
  ): Tensor[D] = {
    val axisWithDefault = if (axis == null) Tensor.zeros(INT32, Shape()) else axis
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.gatherV2(
      executionContext.value.nativeHandle, input.nativeHandle, indices.nativeHandle, axisWithDefault.nativeHandle))
  }

  /** $OpDocBasicGatherND
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
    *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    */
  def gatherND[D <: DataType, I <: Int32OrInt64](input: Tensor[D], indices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.gatherNd(
      executionContext.value.nativeHandle, input.nativeHandle, indices.nativeHandle))
  }

  /** $OpDocBasicScatterND
    *
    * @group BasicOps
    * @param  indices Indices tensor.
    * @param  updates Updates to scatter into the output tensor.
    * @param  shape   One-dimensional tensor specifying the shape of the output tensor.
    * @return Result as a new tensor.
    */
  def scatterND[D <: DataType, I <: Int32OrInt64](
      indices: Tensor[I],
      updates: Tensor[D],
      shape: Tensor[I]
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.scatterNd(
      executionContext.value.nativeHandle, indices.nativeHandle, updates.nativeHandle, shape.nativeHandle))
  }

  /** $OpDocBasicSlice
    *
    * @group BasicOps
    * @param  input Tensor to slice.
    * @param  begin Begin index tensor. `begin(i)` specifies the offset into the `i`th dimension of `input` to slice
    *               from.
    * @param  size  Slice size tensor. `size(i)` specifies the number of elements of the `i`th dimension of `input` to
    *               slice. If `size(i) == -1`, then all the remaining elements in dimension `i` are included in the
    *               slice (i.e., this is equivalent to setting `size(i) = input.shape(i) - begin(i)`).
    * @return Result as a new tensor.
    */
  private[ops] def slice[D <: DataType, I <: Int32OrInt64](
      input: Tensor[D],
      begin: Tensor[I],
      size: Tensor[I]
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.slice(
      executionContext.value.nativeHandle, input.nativeHandle, begin.nativeHandle, size.nativeHandle))
  }

  /** $OpDocBasicStridedSlice
    *
    * @group BasicOps
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
  private[ops] def stridedSlice[D <: DataType, I <: IntOrUInt](
      input: Tensor[D],
      begin: Tensor[I],
      end: Tensor[I],
      strides: Tensor[I] = null,
      beginMask: Long = 0,
      endMask: Long = 0,
      ellipsisMask: Long = 0,
      newAxisMask: Long = 0,
      shrinkAxisMask: Long = 0
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.stridedSlice(
      executionContext.value.nativeHandle, input.nativeHandle, begin.nativeHandle, end.nativeHandle,
      strides.nativeHandle, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask))
  }

  //endregion Tensor Slicing Ops

  //region Tensor Ungrouped Ops

  /** $OpDocBasicCheckNumerics
    *
    * @group BasicOps
    * @param  input   Input tensor.
    * @param  message Prefix to print for the error message.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def checkNumerics[D <: DecimalDataType](input: Tensor[D], message: String = ""): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.checkNumerics(
      executionContext.value.nativeHandle, input.nativeHandle, message.getBytes()))
  }

  /** $OpDocBasicEditDistance
    *
    * @group BasicOps
    * @param  hypothesis Sparse tensor that contains the hypothesis sequences.
    * @param  truth      Sparse tensor that contains the truth sequences.
    * @param  normalize  Optional boolean value indicating whether to normalize the Levenshtein distance by the
    *                    length of `truth`.
    * @return Result as a new tensor.
    */
  def editDistance[D <: DataType](
      hypothesis: SparseTensor[D],
      truth: SparseTensor[D],
      normalize: Boolean = true
  ): Tensor[FLOAT32] = {
    Tensor.fromNativeHandle[FLOAT32](NativeTensorOpsBasic.editDistance(
      executionContext.value.nativeHandle,
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
  def oneHot[D <: DataType, I <: UInt8OrInt32OrInt64](
      indices: Tensor[I],
      depth: Tensor[INT32],
      onValue: Tensor[D] = null,
      offValue: Tensor[D] = null,
      axis: Int = -1,
      dataType: DataType = null
  ): Tensor[D] = {
    val inferredDataType = {
      if (dataType != null) {
        dataType
      } else {
        if (onValue != null && offValue != null)
          DataType.mostPrecise(onValue.dataType, offValue.dataType)
        else if (onValue != null)
          onValue.dataType
        else if (offValue != null)
          offValue.dataType
        else
          FLOAT32
      }
    }
    val actualOnValue = if (onValue != null) onValue else Cast.cast(1: Tensor[INT32], inferredDataType)
    val actualOffValue = if (offValue != null) offValue else Cast.cast(0: Tensor[INT32], inferredDataType)
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.oneHot(
      executionContext.value.nativeHandle, indices.nativeHandle, depth.nativeHandle, actualOnValue.nativeHandle,
      actualOffValue.nativeHandle, axis))
  }

  //endregion Tensor Ungrouped Ops

  // TODO: Add support for all the quantization ops.
  // TODO: Add support for all the broadcasting ops.

  //region Tensor Gradient Ops

  /** $OpDocBasicStopGradient
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def stopGradient[D <: DataType](input: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.stopGradient(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicPreventGradient
    *
    * @group BasicOps
    * @param  input   Input tensor.
    * @param  message Message to print along with the error.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def preventGradient[D <: DataType](input: Tensor[D], message: String = ""): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsBasic.preventGradient(
      executionContext.value.nativeHandle, input.nativeHandle, message.getBytes()))
  }

  //endregion Tensor Gradient Ops
}

object Basic extends Basic {
  private[tensors] trait Implicits {
    implicit class BasicOps[D <: DataType](tensor: Tensor[D]) {
      //region Tensor Manipulation Ops

      /** $OpDocBasicExpandDims
        *
        * @group BasicOps
        * @param  axis Dimension index at which to expand the shape of this tensor.
        * @return Result as a new tensor.
        */
      def expandDims(axis: Tensor[INT32]): Tensor[D] = Basic.expandDims(tensor, axis)

      /** $OpDocBasicSqueeze
        *
        * @group BasicOps
        * @param  axes Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
        *              will be squeezed.
        * @return Result as a new tensor.
        */
      def squeeze(axes: Seq[Int] = null): Tensor[D] = Basic.squeeze(tensor, axes)

      /** $OpDocBasicUnstack
        *
        * @group BasicOps
        * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
        * @param  axis   Dimension along which to unstack the input tensor.
        * @return Result as a new tensor.
        */
      def unstack(number: Int = -1, axis: Int = 0): Seq[Tensor[D]] = Basic.unstack(tensor, number, axis)

      /** $OpDocBasicSplitEvenly
        *
        * @group BasicOps
        * @param  numSplits Number of splits to obtain along the `axis` dimension.
        * @param  axis      Dimension along which to split the input tensor.
        * @return Result as a sequence of new tensors.
        */
      def splitEvenly(numSplits: Int, axis: Tensor[INT32] = 0): Seq[Tensor[D]] = {
        Basic.splitEvenly(tensor, numSplits, axis)
      }

      /** $OpDocBasicSplit
        *
        * @group BasicOps
        * @param  splitSizes Sizes for the splits to obtain.
        * @param  axis       Dimension along which to split the input tensor.
        * @return Result as a new tensor.
        */
      def split[I <: IntOrUInt](splitSizes: Tensor[I], axis: Tensor[INT32] = 0): Seq[Tensor[D]] = {
        Basic.split(tensor, splitSizes, axis)
      }

      /** $OpDocBasicTile
        *
        * @group BasicOps
        * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the
        *                   rank of `input`.
        * @return Result as a new tensor.
        */
      def tile[I <: Int32OrInt64](multiples: Tensor[I]): Tensor[D] = Basic.tile(tensor, multiples)

      /** $OpDocBasicPad
        *
        * @group BasicOps
        * @param  paddings Tensor containing the paddings.
        * @param  mode     Padding mode to use.
        * @return Result as a new tensor.
        */
      def pad[I <: Int32OrInt64](paddings: Tensor[I], mode: PaddingMode = ConstantPadding(Some(Tensor(0)))): Tensor[D] = {
        Basic.pad(tensor, paddings, mode)
      }

      /** $OpDocBasicReshape
        *
        * @group BasicOps
        * @param  shape Shape of the output tensor.
        * @return Result as a new tensor.
        */
      def reshape[I <: Int32OrInt64](shape: Tensor[I]): Tensor[D] = Basic.reshape(tensor, shape)

      /** $OpDocBasicTranspose
        *
        * @group BasicOps
        * @param  permutation Permutation of the input tensor dimensions.
        * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def transpose[I <: Int32OrInt64](permutation: Tensor[I] = null, conjugate: Boolean = false): Tensor[D] = {
        Basic.transpose(tensor, permutation, conjugate)
      }

      /** $OpDocBasicMatrixTranspose
        *
        * @group BasicOps
        * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def matrixTranspose(conjugate: Boolean = false): Tensor[D] = Basic.matrixTranspose(tensor, conjugate)

      /** $OpDocBasicReverse
        *
        * @group BasicOps
        * @param  axes Dimensions of the input tensor to reverse.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverse[I <: Int32OrInt64](axes: Tensor[I]): Tensor[D] = Basic.reverse(tensor, axes)

      /** $OpDocBasicReverseSequence
        *
        * @group BasicOps
        * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
        *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
        * @param  sequenceAxis    Tensor dimension which is partially reversed.
        * @param  batchAxis       Tensor dimension along which the reversal is performed.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverseSequence[I <: Int32OrInt64](
          sequenceLengths: Tensor[I],
          sequenceAxis: Int,
          batchAxis: Int = 0
      ): Tensor[D] = {
        Basic.reverseSequence(tensor, sequenceLengths, sequenceAxis, batchAxis)
      }

      /** $OpDocBasicSpaceToBatch
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def spaceToBatch[I <: Int32OrInt64](blockSize: Int, paddings: Tensor[I]): Tensor[D] = {
        Basic.spaceToBatch(tensor, blockSize, paddings)
      }

      /** $OpDocBasicSpaceToBatchND
        *
        * @group BasicOps
        * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
        * @param  paddings   Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
        *                    `paddings(i) = [padStart, padEnd]` specifies the padding for input dimension `i + 1`, which
        *                    corresponds to spatial dimension `i`. It is required that `blockShape(i)` divides
        *                    `inputShape(i + 1) + padStart + padEnd`.
        * @return Result as a new tensor.
        */
      def spaceToBatchND[I1 <: Int32OrInt64, I2 <: Int32OrInt64](
          blockShape: Tensor[I1],
          paddings: Tensor[I2]
      ): Tensor[D] = {
        Basic.spaceToBatchND(tensor, blockShape, paddings)
      }

      /** $OpDocBasicBatchToSpace
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  crops     `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def batchToSpace[I <: Int32OrInt64](blockSize: Int, crops: Tensor[I]): Tensor[D] = {
        Basic.batchToSpace(tensor, blockSize, crops)
      }

      /** $OpDocBasicBatchToSpaceND
        *
        * @group BasicOps
        * @param  blockShape One-dimensional tensor with shape `[M]` whose elements must all be `>= 1`.
        * @param  crops      Two-dimensional tensor with shape `[M, 2]` whose elements must all be non-negative.
        *                    `crops(i) = [cropStart, cropEnd]` specifies the amount to crop from input dimension `i + 1`,
        *                    which corresponds to spatial dimension `i`. It is required that
        *                    `cropStart(i) + cropEnd(i) <= blockShape(i) * inputShape(i + 1)`.
        * @return Result as a new tensor.
        */
      def batchToSpaceND[I1 <: Int32OrInt64, I2 <: Int32OrInt64](
          blockShape: Tensor[I1],
          crops: Tensor[I2]
      ): Tensor[D] = {
        Basic.batchToSpaceND(tensor, blockShape, crops)
      }

      /** $OpDocBasicSpaceToDepth
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def spaceToDepth(blockSize: Int, dataFormat: CNNDataFormat = CNNDataFormat.default): Tensor[D] = {
        Basic.spaceToDepth(tensor, blockSize, dataFormat)
      }

      /** $OpDocBasicDepthToSpace
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def depthToSpace(blockSize: Int, dataFormat: CNNDataFormat = CNNDataFormat.default): Tensor[D] = {
        Basic.depthToSpace(tensor, blockSize, dataFormat)
      }

      //endregion Tensor Manipulation Ops

      //region Tensor Masking Ops

      /** $OpDocBasicBooleanMask
        *
        * @group BasicOps
        * @param  mask `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
        * @return Result as a new tensor.
        */
      def booleanMask(mask: Tensor[BOOLEAN]): Tensor[D] = Basic.booleanMask(tensor, mask)

      //endregion Tensor Masking Ops

      //region Tensor Counting and Set Ops

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @return Tuple containing `output` and `indices`.
        */
      def unique: (Tensor[D], Tensor[INT32]) = Basic.unique(tensor, INT32)

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output` and `indices`.
        */
      def unique[I <: Int32OrInt64](indicesDataType: I): (Tensor[D], Tensor[I]) = Basic.unique(tensor, indicesDataType)

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts: (Tensor[D], Tensor[INT32], Tensor[INT32]) = Basic.uniqueWithCounts(tensor, INT32)

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts[I <: Int32OrInt64](indicesDataType: I): (Tensor[D], Tensor[I], Tensor[I]) = {
        Basic.uniqueWithCounts(tensor, indicesDataType)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other One-dimensional tensor containing the values to remove.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff(other: Tensor[D]): (Tensor[D], Tensor[INT32]) = {
        Basic.listDiff(tensor, other, INT32)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other           One-dimensional tensor containing the values to remove.
        * @param  indicesDataType Data type to use for the output indices of this op.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff[I <: Int32OrInt64](other: Tensor[D], indicesDataType: I): (Tensor[D], Tensor[I]) = {
        Basic.listDiff(tensor, other, indicesDataType)
      }

      //endregion Tensor Counting and Set Ops

      //region Tensor Slicing Ops

      /** $OpDocBasicGather
        *
        * @group BasicOps
        * @param  indices Tensor containing indices to gather.
        * @param  axis    Tensor containing the axis along which to gather.
        * @return Result as a new tensor.
        */
      def gather[I <: Int32OrInt64](indices: Tensor[I], axis: Tensor[I] = null): Tensor[D] = {
        Basic.gather(tensor, indices, axis)
      }

      /** $OpDocBasicGatherND
        *
        * @group BasicOps
        * @param  indices Tensor containing indices to gather.
        * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
        *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
        */
      def gatherND[I <: Int32OrInt64](indices: Tensor[I]): Tensor[D] = Basic.gatherND(tensor, indices)

      /** Slices this tensor according to the provided indexers.
        *
        * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
        *
        * @group BasicOps
        * @param  firstIndexer  First indexer to use.
        * @param  otherIndexers Rest of the indexers to use.
        * @return Resulting tensor.
        */
      def slice(firstIndexer: Indexer, otherIndexers: Indexer*): Tensor[D] = {
        val stridedSlice = Indexer.toStridedSlice(firstIndexer, otherIndexers: _*)
        val beginTensor: Tensor[INT32] = stridedSlice._1
        val endTensor: Tensor[INT32] = stridedSlice._2
        val stridesTensor: Tensor[INT32] = stridedSlice._3
        val result = Basic.stridedSlice(
          tensor, beginTensor, endTensor, stridesTensor, stridedSlice._4, stridedSlice._5, stridedSlice._6,
          stridedSlice._7, stridedSlice._8)
        result
      }

      //endregion Tensor Slicing Ops

      //region Tensor Gradient Ops

      /** $OpDocBasicStopGradient
        *
        * @group BasicOps
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def stopGradient(): Tensor[D] = Basic.stopGradient(tensor)

      /** $OpDocBasicPreventGradient
        *
        * @group BasicOps
        * @param  message Message to print along with the error.
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def preventGradient(message: String = ""): Tensor[D] = Basic.preventGradient(tensor, message)

      //endregion Tensor Gradient Ops
    }

    implicit class NumericBasicOps[D <: NumericDataType](val tensor: Tensor[D]) {
      //region Tensor Masking Ops

      /** $OpDocBasicSequenceMask
        *
        * @group BasicOps
        * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum
        *                   value in this tensor.
        * @return Result as a new tensor.
        */
      def sequenceMask(maxLength: Tensor[D] = null): Tensor[BOOLEAN] = {
        Basic.sequenceMask(tensor, maxLength)
      }

      //endregion Tensor Masking Ops
    }

    implicit class DecimalBasicOps[D <: DecimalDataType](val tensor: Tensor[D]) {
      //region Tensor Ungrouped Ops

      /** $OpDocBasicCheckNumerics
        *
        * @group BasicOps
        * @param  message Prefix to print for the error message.
        * @return Result as a new tensor which has the same value as the input tensor.
        */
      def checkNumerics(message: String = ""): Tensor[D] = Basic.checkNumerics(tensor)

      //endregion Tensor Ungrouped Ops
    }

    implicit class Int32OrInt64BasicOps[D <: Int32OrInt64](val tensor: Tensor[D]) {
      //region Tensor Manipulation Ops

      /** $OpDocBasicInvertPermutation
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def invertPermutation(): Tensor[D] = Basic.invertPermutation(tensor)

      //endregion Tensor Manipulation Ops
    }

    implicit class BooleanBasicOps(val tensor: Tensor[BOOLEAN]) {
      //region Tensor Masking Ops

      /** $OpDocBasicWhere
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def where(): Tensor[INT64] = Basic.where(tensor)

      //endregion Tensor Masking Ops
    }

    implicit def tensorConvertibleToBasicOps[D <: DataType, T](value: T)(implicit f: T => Tensor[D]): BasicOps[D] = new BasicOps(f(value))
    implicit def tensorConvertibleToNumericBasicOps[D <: NumericDataType, T](value: T)(implicit f: T => Tensor[D]): NumericBasicOps[D] = new NumericBasicOps(f(value))
    implicit def tensorConvertibleToDecimalBasicOps[D <: DecimalDataType, T](value: T)(implicit f: T => Tensor[D]): DecimalBasicOps[D] = new DecimalBasicOps(f(value))
    implicit def tensorConvertibleToInt32OrInt64BasicOps[D <: Int32OrInt64, T](value: T)(implicit f: T => Tensor[D]): Int32OrInt64BasicOps[D] = new Int32OrInt64BasicOps(f(value))
    implicit def tensorConvertibleToBooleanBasicOps[T](value: T)(implicit f: T => Tensor[BOOLEAN]): BooleanBasicOps = new BooleanBasicOps(f(value))
  }
}
