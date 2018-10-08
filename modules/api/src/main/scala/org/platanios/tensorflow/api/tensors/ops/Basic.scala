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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.Basic.{ConstantPadding, PaddingMode}
import org.platanios.tensorflow.api.ops.NN.CNNDataFormat
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import java.nio.charset.StandardCharsets

import scala.language.postfixOps

/** Contains functions for executing ops related to basic tensor manipulation.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Basic {
  //region Tensor Shape Ops

  /** $OpDocBasicRank
    *
    * @group BasicOps
    * @param  input Tensor whose rank to return.
    * @return Result as a new tensor.
    */
  def rank[T <: TensorLike[_]](input: T): Tensor[Int] = {
    input match {
      case t: Tensor[_] => Tensor.fill[Int](Shape())(t.rank)
      case t: TensorIndexedSlices[_] => size(t.denseShape).toInt
      case t: SparseTensor[_] => size(t.denseShape).toInt
    }
  }

  /** $OpDocBasicSize
    *
    * @group BasicOps
    * @param  input Tensor whose size to return.
    * @return Result as a new tensor.
    */
  def size[T <: TensorLike[_]](input: T): Tensor[Long] = {
    input match {
      case t: Tensor[_] => Tensor.fill[Long](Shape())(t.size)
      case t: TensorIndexedSlices[_] => Math.prod(t.denseShape, Array(0))
      case t: SparseTensor[_] => Math.prod(t.denseShape, Array(0))
    }
  }

  /** $OpDocBasicShape
    *
    * @group BasicOps
    * @param  input    Tensor whose shape to return.
    * @return Result as a new tensor.
    */
  def shape[T <: TensorLike[_]](input: T): Tensor[Long] = {
    input match {
      case t: Tensor[_] => t.shape.toTensor
      case t: TensorIndexedSlices[_] => t.denseShape
      case t: SparseTensor[_] => t.denseShape
    }
  }

  /** $OpDocBasicShapeN
    *
    * @group BasicOps
    * @param  inputs Tensors whose shapes to return.
    * @return Result as a sequence of new tensors.
    */
  def shapeN(inputs: Seq[Tensor[_]]): Seq[Tensor[Long]] = {
    inputs.map(_.shape.toTensor)
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
  def expandDims[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axis: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.expandDims(
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
  def squeeze[T: TF](
      input: Tensor[T],
      axes: Seq[Int] = null
  ): Tensor[T] = {
    val longAxes: Array[Long] = if (axes == null) null else axes.map(_.toLong).toArray
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.squeeze(
      executionContext.value.nativeHandle, input.nativeHandle, longAxes))
  }

  /** $OpDocBasicStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @param  axis   Dimension along which to stack the input tensors.
    * @return Result as a new tensor.
    */
  def stack[T: TF](
      inputs: Seq[Tensor[T]],
      axis: Int = 0
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.pack(
      executionContext.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis))
  }

  /** $OpDocBasicParallelStack
    *
    * @group BasicOps
    * @param  inputs Input tensors to be stacked.
    * @return Result as a new tensor.
    */
  def parallelStack[T: TF](
      inputs: Seq[Tensor[T]]
  ): Tensor[T] = {
    val outputShape = Shape(inputs.length).concatenateWith(inputs.head.shape)
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.parallelConcat(
      executionContext.value.nativeHandle, inputs.map(_.nativeHandle).toArray, outputShape.asArray.map(_.toLong)))
  }

  /** $OpDocBasicUnstack
    *
    * @group BasicOps
    * @param  input  Rank `R > 0` `Tensor` to be unstacked.
    * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
    * @param  axis   Dimension along which to unstack the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def unstack[T: TF](
      input: Tensor[T],
      number: Int = -1,
      axis: Int = 0
  ): Seq[Tensor[T]] = {
    val num: Int = if (number >= 0) number else input.shape(axis)
    NativeTensorOpsBasic.unpack(executionContext.value.nativeHandle, input.nativeHandle, num, axis)
        .map(Tensor.fromNativeHandle[T])
  }

  /** $OpDocBasicConcatenate
    *
    * @group BasicOps
    * @param  inputs Input tensors to be concatenated.
    * @param  axis   Dimension along which to concatenate the input tensors.
    * @return Result as a new tensor.
    */
  def concatenate[T: TF](
      inputs: Seq[Tensor[T]],
      axis: Tensor[Int] = 0
  ): Tensor[T] = {
    if (inputs.lengthCompare(1) == 0)
      inputs.head
    else
      Tensor.fromNativeHandle[T](
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
  private[ops] def concatenateOffset(
      shapes: Seq[Tensor[Int]],
      axis: Tensor[Int]
  ): Seq[Tensor[Int]] = {
    NativeTensorOpsBasic.concatOffset(
      executionContext.value.nativeHandle, axis.nativeHandle, shapes.map(_.nativeHandle).toArray)
        .map(Tensor.fromNativeHandle[Int])
  }

  /** $OpDocBasicSplitEvenly
    *
    * @group BasicOps
    * @param  input     Input tensor to split.
    * @param  numSplits Number of splits to obtain along the `axis` dimension.
    * @param  axis      Dimension along which to split the input tensor.
    * @return Result as a sequence of new tensors.
    */
  def splitEvenly[T: TF](
      input: Tensor[T],
      numSplits: Int,
      axis: Tensor[Int] = 0
  ): Seq[Tensor[T]] = {
    NativeTensorOpsBasic.split(
      executionContext.value.nativeHandle, axis.nativeHandle, input.nativeHandle, numSplits.toLong)
        .map(Tensor.fromNativeHandle[T])
  }

  /** $OpDocBasicSplit
    *
    * @group BasicOps
    * @param  input      Input tensor to split.
    * @param  splitSizes Sizes for the splits to obtain.
    * @param  axis       Dimension along which to split the input tensor.
    * @return Result as a new tensor.
    */
  def split[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      splitSizes: Tensor[I],
      axis: Tensor[Int] = 0
  ): Seq[Tensor[T]] = {
    NativeTensorOpsBasic.splitV(
      executionContext.value.nativeHandle, input.nativeHandle, splitSizes.nativeHandle,
      axis.nativeHandle, splitSizes.shape(0))
        .map(Tensor.fromNativeHandle[T])
  }

  /** $OpDocBasicTile
    *
    * @group BasicOps
    * @param  input     Tensor to tile.
    * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the rank
    *                   of `input`.
    * @return Result as a new tensor.
    */
  def tile[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      multiples: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.tile(
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
  def pad[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      paddings: Tensor[I],
      mode: PaddingMode = ConstantPadding(Some(Tensor(0)))
  ): Tensor[T] = {
    mode.pad(input, paddings)
  }

  /** $OpDocBasicReshape
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @param  shape Shape of the output tensor.
    * @return Result as a new tensor.
    */
  def reshape[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      shape: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.reshape(
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
  def transpose[T: TF, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      permutation: Tensor[I] = null,
      conjugate: Boolean = false
  ): Tensor[T] = {
    if (permutation == null) {
      val inputRank = rank(input)
      val reversePermutation = inputRank - 1 - Math.range(0, inputRank, 1)
      if (conjugate && input.dataType.isComplex)
        Tensor.fromNativeHandle[T](NativeTensorOpsBasic.conjugateTranspose(
          executionContext.value.nativeHandle, input.nativeHandle, reversePermutation.nativeHandle))
      else
        Tensor.fromNativeHandle[T](NativeTensorOpsBasic.transpose(
          executionContext.value.nativeHandle, input.nativeHandle, reversePermutation.nativeHandle))
    } else {
      if (conjugate && input.dataType.isComplex)
        Tensor.fromNativeHandle[T](NativeTensorOpsBasic.conjugateTranspose(
          executionContext.value.nativeHandle, input.nativeHandle, permutation.nativeHandle))
      else
        Tensor.fromNativeHandle[T](NativeTensorOpsBasic.transpose(
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
  def matrixTranspose[T: TF](
      input: Tensor[T],
      conjugate: Boolean = false
  ): Tensor[T] = {
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
  def invertPermutation[I: TF : IsInt32OrInt64](
      input: Tensor[I]
  ): Tensor[I] = {
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
  def reverse[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.reverseV2(
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
  def reverseSequence[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      sequenceLengths: Tensor[I],
      sequenceAxis: Int,
      batchAxis: Int = 0
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.reverseSequence(
      executionContext.value.nativeHandle, input.nativeHandle,
      sequenceLengths.nativeHandle, sequenceAxis, batchAxis))
  }

  /** $OpDocBasicSpaceToBatch
    *
    * @group BasicOps
    * @param  input     `4`-dimensional input tensor with shape `[batch, height, width, depth]`.
    * @param  blockSize Block size which must be greater than `1`.
    * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
    * @return Result as a new tensor.
    */
  def spaceToBatch[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      blockSize: Int,
      paddings: Tensor[I]
  ): Tensor[T] = {
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
  def spaceToBatchND[T: TF, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      input: Tensor[T],
      blockShape: Tensor[I1],
      paddings: Tensor[I2]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.spaceToBatchND(
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
  def batchToSpace[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      blockSize: Int,
      crops: Tensor[I]
  ): Tensor[T] = {
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
  def batchToSpaceND[T: TF, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      input: Tensor[T],
      blockShape: Tensor[I1],
      crops: Tensor[I2]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.batchToSpaceND(
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
      inputShape: Tensor[Int],
      blockShape: Tensor[Int],
      basePaddings: Tensor[Int] = null
  ): (Tensor[Int], Tensor[Int]) = {
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
      (Tensor.zeros[Int](Shape(0, 2)), Tensor.zeros[Int](Shape(0, 2)))
    } else {
      inputShape.shape.assertIsCompatibleWith(Shape(numBlockDims))
      val actualBasePaddings = {
        if (basePaddings != null) {
          basePaddings.shape.assertIsCompatibleWith(Shape(numBlockDims, 2))
          basePaddings
        } else {
          Tensor.zeros[Int](Shape(numBlockDims, 2))
        }
      }
      val padStart = actualBasePaddings(::, 0)
      val originalPadEnd = actualBasePaddings(::, 1)
      val fullInputShape = inputShape + padStart + originalPadEnd
      val extraPadEnd = (blockShape - (fullInputShape % blockShape)) % blockShape
      val padEnd = originalPadEnd + extraPadEnd
      val resultPaddings = stack((0 until numBlockDims).map(i => concatenate(Seq(padStart(i), padEnd(i)))))
      val zero = Tensor[Int](0)
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
  def spaceToDepth[T: TF](
      input: Tensor[T],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.spaceToDepth(
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
  def depthToSpace[T: TF](
      input: Tensor[T],
      blockSize: Int,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.depthToSpace(
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
  def where[T: TF : IsBooleanOrNumeric](
      input: Tensor[T]
  ): Tensor[Long] = {
    Tensor.fromNativeHandle[Long](NativeTensorOpsBasic.where(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicBooleanMask
    *
    * @group BasicOps
    * @param  input `N`-dimensional tensor.
    * @param  mask  `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
    * @return Result as a new tensor.
    */
  def booleanMask[T: TF](
      input: Tensor[T],
      mask: Tensor[Boolean]
  ): Tensor[T] = {
    val maskShape = mask.shape
    val maskRank = maskShape.rank
    val leadingSize = reshape(Math.prod(input.shape(0 :: maskRank), Seq(0)), Shape(1))
    val reshapedInput = reshape(
      input,
      concatenate(
        Seq(leadingSize, input.shape(maskRank ::).toTensor),
        axis = 0))
    gather(reshapedInput, squeeze(where(reshape(mask, Seq(-1))), axes = Seq(1)), axis = 0)
  }

  /** $OpDocBasicSequenceMask
    *
    * @group BasicOps
    * @param  lengths   One-dimensional tensor containing the lengths to keep for each row. If `maxLength` is
    *                   provided, then all values in `lengths` must be smaller than `maxLength`.
    * @param  maxLength Scalar tensor representing the maximum length of each row. Defaults to the maximum value
    *                   in `lengths`.
    * @return Result as a new tensor.
    * @throws IllegalArgumentException If `maxLength` is not a scalar.
    */
  @throws[IllegalArgumentException]
  def sequenceMask[T: TF : IsIntOrUInt](
      lengths: Tensor[T],
      maxLength: Tensor[T] = null
  ): Tensor[Boolean] = {
    require(maxLength == null || maxLength.rank == -1 || maxLength.rank == 0, "'maxLength' must be a scalar.")
    val maxLen = if (maxLength != null) maxLength else Math.max(lengths)
    // The basic idea is to compare a range row vector of size 'maxLen', [0, 1, 2, 3, 4], to 'lengths' as a matrix
    // with one column, [[1], [3], [2]]. Because of broadcasting on both arguments, this comparison results in a
    // matrix of size [lengths.shape(0), maxLen].
    val rowVector = Math.range(
      start = Tensor.zeros(lengths.dataType, Shape()),
      limit = maxLen,
      delta = Tensor.ones(lengths.dataType, Shape()))
    // Since 'maxLen' >= max(lengths), it is safe to use 'maxLen' as a cast authoritative type. Whenever 'maxLen' fits
    // into Int, then so do the elements of 'lengths'.
    val matrix = expandDims(lengths, axis = 1).castTo(lengths.dataType)
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
  def indexedSlicesMask[T: TF](
      input: TensorIndexedSlices[T],
      maskIndices: Tensor[Int]
  ): TensorIndexedSlices[T] = {
    val (outputIndices, toGather) = listDiff(input.indices, maskIndices.toLong, Int)
    val outputValues = gather(input.values, toGather)
    TensorIndexedSlices(
      indices = outputIndices.toLong,
      values = outputValues,
      denseShape = input.denseShape)
  }

  //endregion Tensor Masking Ops

  //region Tensor Counting and Set Ops

  // TODO: [TENSORS] Update to 'uniqueV2' and 'uniqueWithCountsV2'.

  /** $OpDocBasicUnique
    *
    * @group BasicOps
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices.
    * @return Tuple containing `output` and `indices`.
    */
  def unique[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      indicesDataType: DataType[I]
  ): (Tensor[T], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.unique(
      executionContext.value.nativeHandle, input.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[T](tensors(0)), Tensor.fromNativeHandle[I](tensors(1)))
  }

  /** $OpDocBasicUniqueWithCounts
    *
    * @group BasicOps
    * @param  input           One-dimensional input tensor.
    * @param  indicesDataType Data type of the returned indices.
    * @return Tuple containing `output`, `indices`, and `counts`.
    */
  def uniqueWithCounts[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      indicesDataType: DataType[I]
  ): (Tensor[T], Tensor[I], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.uniqueWithCounts(
      executionContext.value.nativeHandle, input.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[T](tensors(0)),
        Tensor.fromNativeHandle[I](tensors(1)),
        Tensor.fromNativeHandle[I](tensors(2)))
  }

  /** $OpDocBasicListDiff
    *
    * @group BasicOps
    * @param  x               One-dimensional tensor containing the values to keep.
    * @param  y               One-dimensional tensor containing the values to remove.
    * @param  indicesDataType Data type to use for the output indices of this op.
    * @return Tuple containing `output` and `indices`, from the method description.
    */
  def listDiff[T: TF, I: TF : IsInt32OrInt64](
      x: Tensor[T],
      y: Tensor[T],
      indicesDataType: DataType[I]
  ): (Tensor[T], Tensor[I]) = {
    val tensors = NativeTensorOpsBasic.listDiff(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, indicesDataType.cValue)
    (Tensor.fromNativeHandle[T](tensors(0)), Tensor.fromNativeHandle[I](tensors(1)))
  }

  //endregion Tensor Counting and Set Ops

  //region Tensor Slicing Ops

  /** $OpDocBasicGather
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @return Result as a new tensor.
    */
  def gather[T: TF, I1: TF : IsInt32OrInt64](
      input: Tensor[T],
      indices: Tensor[I1]
  ): Tensor[T] = {
    gather(input, indices, axis = 0)
  }

  /** $OpDocBasicGather
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @param  axis    Tensor containing the axis along which to gather.
    * @return Result as a new tensor.
    */
  def gather[T: TF, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      input: Tensor[T],
      indices: Tensor[I1],
      axis: Tensor[I2]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.gatherV2(
      executionContext.value.nativeHandle, input.nativeHandle, indices.nativeHandle, axis.nativeHandle))
  }

  /** $OpDocBasicGatherND
    *
    * @group BasicOps
    * @param  input   Tensor from which to gather values.
    * @param  indices Tensor containing indices to gather.
    * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
    *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
    */
  def gatherND[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      indices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.gatherNd(
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
  def scatterND[T: TF, I: TF : IsInt32OrInt64](
      indices: Tensor[I],
      updates: Tensor[T],
      shape: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.scatterNd(
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
  private[ops] def slice[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      begin: Tensor[I],
      size: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.slice(
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
  private[tensors] def stridedSlice[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      begin: Tensor[I],
      end: Tensor[I],
      strides: Tensor[I] = null,
      beginMask: Long = 0,
      endMask: Long = 0,
      ellipsisMask: Long = 0,
      newAxisMask: Long = 0,
      shrinkAxisMask: Long = 0
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.stridedSlice(
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
  def checkNumerics[T: TF : IsDecimal](
      input: Tensor[T],
      message: String = ""
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.checkNumerics(
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
  def editDistance[T: TF](
      hypothesis: SparseTensor[T],
      truth: SparseTensor[T],
      normalize: Boolean = true
  ): Tensor[Float] = {
    Tensor.fromNativeHandle[Float](NativeTensorOpsBasic.editDistance(
      executionContext.value.nativeHandle,
      hypothesis.indices.nativeHandle,
      hypothesis.values.nativeHandle,
      hypothesis.denseShape.nativeHandle,
      truth.indices.nativeHandle,
      truth.values.nativeHandle,
      truth.denseShape.nativeHandle,
      normalize))
  }

  //endregion Tensor Ungrouped Ops

  // TODO: [TENSORS] Add support for all the quantization ops.
  // TODO: [TENSORS] Add support for all the broadcasting ops.

  //region Tensor Gradient Ops

  /** $OpDocBasicStopGradient
    *
    * @group BasicOps
    * @param  input Input tensor.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def stopGradient[T: TF](input: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.stopGradient(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocBasicPreventGradient
    *
    * @group BasicOps
    * @param  input   Input tensor.
    * @param  message Message to print along with the error.
    * @return Result as a new tensor which has the same value as the input tensor.
    */
  def preventGradient[T: TF](input: Tensor[T], message: String = ""): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsBasic.preventGradient(
      executionContext.value.nativeHandle, input.nativeHandle, message.getBytes()))
  }

  //endregion Tensor Gradient Ops
}

object Basic extends Basic {
  private[tensors] trait Implicits {
    implicit def tensorConvertibleToBasicOps[TC, T: TF](
        value: TC
    )(implicit f: TC => Tensor[T]): BasicOps[T] = {
      new BasicOps(f(value))
    }

    implicit class BasicOps[T: TF](tensor: Tensor[T]) {
      //region Tensor Manipulation Ops

      /** $OpDocBasicExpandDims
        *
        * @group BasicOps
        * @param  axis Dimension index at which to expand the shape of this tensor.
        * @return Result as a new tensor.
        */
      def expandDims(axis: Tensor[Int]): Tensor[T] = {
        Basic.expandDims(tensor, axis)
      }

      /** $OpDocBasicSqueeze
        *
        * @group BasicOps
        * @param  axes Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
        *              will be squeezed.
        * @return Result as a new tensor.
        */
      def squeeze(axes: Seq[Int] = null): Tensor[T] = {
        Basic.squeeze(tensor, axes)
      }

      /** $OpDocBasicUnstack
        *
        * @group BasicOps
        * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
        * @param  axis   Dimension along which to unstack the input tensor.
        * @return Result as a new tensor.
        */
      def unstack(number: Int = -1, axis: Int = 0): Seq[Tensor[T]] = {
        Basic.unstack(tensor, number, axis)
      }

      /** $OpDocBasicSplitEvenly
        *
        * @group BasicOps
        * @param  numSplits Number of splits to obtain along the `axis` dimension.
        * @param  axis      Dimension along which to split the input tensor.
        * @return Result as a sequence of new tensors.
        */
      def splitEvenly(
          numSplits: Int,
          axis: Tensor[Int] = 0
      ): Seq[Tensor[T]] = {
        Basic.splitEvenly(tensor, numSplits, axis)
      }

      /** $OpDocBasicSplit
        *
        * @group BasicOps
        * @param  splitSizes Sizes for the splits to obtain.
        * @param  axis       Dimension along which to split the input tensor.
        * @return Result as a new tensor.
        */
      def split[I: TF : IsInt32OrInt64](
          splitSizes: Tensor[I],
          axis: Tensor[Int] = 0
      ): Seq[Tensor[T]] = {
        Basic.split(tensor, splitSizes, axis)
      }

      /** $OpDocBasicTile
        *
        * @group BasicOps
        * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the
        *                   rank of `input`.
        * @return Result as a new tensor.
        */
      def tile[I: TF : IsInt32OrInt64](multiples: Tensor[I]): Tensor[T] = {
        Basic.tile(tensor, multiples)
      }

      /** $OpDocBasicPad
        *
        * @group BasicOps
        * @param  paddings Tensor containing the paddings.
        * @param  mode     Padding mode to use.
        * @return Result as a new tensor.
        */
      def pad[I: TF : IsInt32OrInt64](
          paddings: Tensor[I],
          mode: PaddingMode = ConstantPadding(Some(Tensor(0)))
      ): Tensor[T] = {
        Basic.pad(tensor, paddings, mode)
      }

      /** $OpDocBasicReshape
        *
        * @group BasicOps
        * @param  shape Shape of the output tensor.
        * @return Result as a new tensor.
        */
      def reshape[I: TF : IsInt32OrInt64](shape: Tensor[I]): Tensor[T] = {
        Basic.reshape(tensor, shape)
      }

      /** $OpDocBasicTranspose
        *
        * @group BasicOps
        * @param  permutation Permutation of the input tensor dimensions.
        * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def transpose[I: IntDefault : TF : IsInt32OrInt64](
          permutation: Tensor[I] = null,
          conjugate: Boolean = false
      ): Tensor[T] = {
        Basic.transpose(tensor, permutation, conjugate)
      }

      /** $OpDocBasicMatrixTranspose
        *
        * @group BasicOps
        * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
        * @return Result as a new tensor.
        */
      def matrixTranspose(conjugate: Boolean = false): Tensor[T] = {
        Basic.matrixTranspose(tensor, conjugate)
      }

      /** $OpDocBasicInvertPermutation
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def invertPermutation(implicit ev: IsInt32OrInt64[T]): Tensor[T] = {
        Basic.invertPermutation(tensor)
      }

      /** $OpDocBasicReverse
        *
        * @group BasicOps
        * @param  axes Dimensions of the input tensor to reverse.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverse[I: TF : IsInt32OrInt64](axes: Tensor[I]): Tensor[T] = {
        Basic.reverse(tensor, axes)
      }

      /** $OpDocBasicReverseSequence
        *
        * @group BasicOps
        * @param  sequenceLengths One-dimensional tensor with length `input.shape(batchAxis)` and
        *                         `max(sequenceLengths) <= input.shape(sequenceAxis)`.
        * @param  sequenceAxis    Tensor dimension which is partially reversed.
        * @param  batchAxis       Tensor dimension along which the reversal is performed.
        * @return Result as a new tensor which has the same shape as `input`.
        */
      def reverseSequence[I: TF : IsInt32OrInt64](
          sequenceLengths: Tensor[I],
          sequenceAxis: Int,
          batchAxis: Int = 0
      ): Tensor[T] = {
        Basic.reverseSequence(tensor, sequenceLengths, sequenceAxis, batchAxis)
      }

      /** $OpDocBasicSpaceToBatch
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def spaceToBatch[I: TF : IsInt32OrInt64](
          blockSize: Int,
          paddings: Tensor[I]
      ): Tensor[T] = {
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
      def spaceToBatchND[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          blockShape: Tensor[I1],
          paddings: Tensor[I2]
      ): Tensor[T] = {
        Basic.spaceToBatchND(tensor, blockShape, paddings)
      }

      /** $OpDocBasicBatchToSpace
        *
        * @group BasicOps
        * @param  blockSize Block size which must be greater than `1`.
        * @param  crops     `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
        * @return Result as a new tensor.
        */
      def batchToSpace[I: TF : IsInt32OrInt64](
          blockSize: Int,
          crops: Tensor[I]
      ): Tensor[T] = {
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
      def batchToSpaceND[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          blockShape: Tensor[I1],
          crops: Tensor[I2]
      ): Tensor[T] = {
        Basic.batchToSpaceND(tensor, blockShape, crops)
      }

      /** $OpDocBasicSpaceToDepth
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def spaceToDepth(
          blockSize: Int,
          dataFormat: CNNDataFormat = CNNDataFormat.default
      ): Tensor[T] = {
        Basic.spaceToDepth(tensor, blockSize, dataFormat)
      }

      /** $OpDocBasicDepthToSpace
        *
        * @group BasicOps
        * @param  blockSize  Block size which must be greater than `1`.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new tensor.
        */
      def depthToSpace(
          blockSize: Int,
          dataFormat: CNNDataFormat = CNNDataFormat.default
      ): Tensor[T] = {
        Basic.depthToSpace(tensor, blockSize, dataFormat)
      }

      //endregion Tensor Manipulation Ops

      //region Tensor Masking Ops

      /** $OpDocBasicWhere
        *
        * @group BasicOps
        * @return Result as a new tensor.
        */
      def where(implicit ev: IsBooleanOrNumeric[T]): Tensor[Long] = {
        Basic.where(tensor)
      }

      /** $OpDocBasicBooleanMask
        *
        * @group BasicOps
        * @param  mask `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
        * @return Result as a new tensor.
        */
      def booleanMask(mask: Tensor[Boolean]): Tensor[T] = {
        Basic.booleanMask(tensor, mask)
      }

      /** $OpDocBasicSequenceMask
        *
        * @group BasicOps
        * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum
        *                   value in this tensor.
        * @return Result as a new tensor.
        */
      def sequenceMask(
          maxLength: Tensor[T] = null
      )(implicit ev: IsIntOrUInt[T]): Tensor[Boolean] = {
        Basic.sequenceMask(tensor, maxLength)
      }

      //endregion Tensor Masking Ops

      //region Tensor Counting and Set Ops

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @return Tuple containing `output` and `indices`.
        */
      def unique: (Tensor[T], Tensor[Int]) = {
        Basic.unique(tensor, indicesDataType = Int)
      }

      /** $OpDocBasicUnique
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output` and `indices`.
        */
      def unique[I: TF : IsInt32OrInt64](
          indicesDataType: DataType[I]
      ): (Tensor[T], Tensor[I]) = {
        Basic.unique(tensor, indicesDataType)
      }

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts: (Tensor[T], Tensor[Int], Tensor[Int]) = {
        Basic.uniqueWithCounts(tensor, indicesDataType = Int)
      }

      /** $OpDocBasicUniqueWithCounts
        *
        * @group BasicOps
        * @param  indicesDataType Data type of the returned indices.
        * @return Tuple containing `output`, `indices`, and `counts`.
        */
      def uniqueWithCounts[I: TF : IsInt32OrInt64](
          indicesDataType: DataType[I]
      ): (Tensor[T], Tensor[I], Tensor[I]) = {
        Basic.uniqueWithCounts(tensor, indicesDataType)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other One-dimensional tensor containing the values to remove.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff(other: Tensor[T]): (Tensor[T], Tensor[Int]) = {
        Basic.listDiff(tensor, other, indicesDataType = Int)
      }

      /** $OpDocBasicListDiff
        *
        * @group BasicOps
        * @param  other           One-dimensional tensor containing the values to remove.
        * @param  indicesDataType Data type to use for the output indices of this op.
        * @return Tuple containing `output` and `indices`, from the method description.
        */
      def listDiff[I: TF : IsInt32OrInt64](
          other: Tensor[T],
          indicesDataType: DataType[I]
      ): (Tensor[T], Tensor[I]) = {
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
      def gather[I: TF : IsInt32OrInt64](
          indices: Tensor[I],
          axis: Tensor[I] = null
      ): Tensor[T] = {
        Basic.gather(tensor, indices, axis)
      }

      /** $OpDocBasicGatherND
        *
        * @group BasicOps
        * @param  indices Tensor containing indices to gather.
        * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
        *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
        */
      def gatherND[I: TF : IsInt32OrInt64](indices: Tensor[I]): Tensor[T] = {
        Basic.gatherND(tensor, indices)
      }

      //endregion Tensor Slicing Ops

      //region Tensor Ungrouped Ops

      /** $OpDocBasicCheckNumerics
        *
        * @group BasicOps
        * @param  message Prefix to print for the error message.
        * @return Result as a new tensor which has the same value as the input tensor.
        */
      def checkNumerics(
          message: String = ""
      )(implicit ev: IsDecimal[T]): Tensor[T] = {
        Basic.checkNumerics(tensor)
      }

      // TODO: [TENSORS] !!! oneHot

      //endregion Tensor Ungrouped Ops

      //region Tensor Gradient Ops

      /** $OpDocBasicStopGradient
        *
        * @group BasicOps
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def stopGradient(): Tensor[T] = {
        Basic.stopGradient(tensor)
      }

      /** $OpDocBasicPreventGradient
        *
        * @group BasicOps
        * @param  message Message to print along with the error.
        * @return Result as a new tensor which has the same value as this tensor.
        */
      def preventGradient(message: String = ""): Tensor[T] = {
        Basic.preventGradient(tensor, message)
      }

      //endregion Tensor Gradient Ops
    }
  }
}
