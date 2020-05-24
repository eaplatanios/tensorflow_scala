/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.ops.NN.CNNDataFormat
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.basic.Manipulation.{ConstantPadding, PaddingMode}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

trait BasicImplicits {
  implicit def outputConvertibleToBasicOps[T, OC](
      value: OC
  )(implicit f: OC => Output[T]): BasicOps[T] = {
    new BasicOps(f(value))
  }

  implicit class BasicOps[T](val output: Output[T]) {
    protected implicit val evTTF: TF[T] = {
      TF.fromDataType(output.dataType)
    }

    //region Tensor Manipulation Ops

    /** $OpDocBasicIdentity
      *
      * @group BasicOps
      * @return Created op output.
      */
    def identity: Output[T] = {
      Basic.identity(output)
    }

    /** $OpDocBasicExpandDims
      *
      * @group BasicOps
      * @param  axis Dimension index at which to expand the shape of this tensor.
      * @return Result as a new tensor.
      */
    def expandDims(axis: Output[Int]): Output[T] = {
      Basic.expandDims(output, axis)
    }

    /** $OpDocBasicSqueeze
      *
      * @group BasicOps
      * @param  axes Dimensions of size 1 to squeeze. If this argument is not provided, then all dimensions of size 1
      *              will be squeezed.
      * @return Result as a new tensor.
      */
    def squeeze(axes: Seq[Int] = null): Output[T] = {
      Basic.squeeze(output, axes)
    }

    /** $OpDocBasicUnstack
      *
      * @group BasicOps
      * @param  number Number of tensors to unstack. If set to `-1` (the default value), its value will be inferred.
      * @param  axis   Dimension along which to unstack the input tensor.
      * @return Result as a new tensor.
      */
    def unstack(number: Int = -1, axis: Int = 0): Seq[Output[T]] = {
      Basic.unstack(output, number, axis)
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
        axis: Output[Int] = Output.constant[Int](Tensor.zeros[Int](Shape()))
    ): Seq[Output[T]] = {
      Basic.splitEvenly(output, numSplits, axis)
    }

    /** $OpDocBasicSplit
      *
      * @group BasicOps
      * @param  splitSizes Sizes for the splits to obtain.
      * @param  axis       Dimension along which to split the input tensor.
      * @return Result as a new tensor.
      */
    def split[I: TF : IsIntOrLong](
        splitSizes: Output[I],
        axis: Output[Int] = Output.constant[Int](Tensor.zeros[Int](Shape()))
    ): Seq[Output[T]] = {
      Basic.split(output, splitSizes, axis)
    }

    /** $OpDocBasicTile
      *
      * @group BasicOps
      * @param  multiples One-dimensional tensor containing the tiling multiples. Its length must be the same as the
      *                   rank of `input`.
      * @return Result as a new tensor.
      */
    def tile[I: TF : IsIntOrLong](multiples: Output[I]): Output[T] = {
      Basic.tile(output, multiples)
    }

    /** $OpDocBasicPad
      *
      * @group BasicOps
      * @param  paddings Tensor containing the paddings.
      * @param  mode     Padding mode to use.
      * @return Result as a new tensor.
      */
    def pad[I: TF : IsIntOrLong](
        paddings: Output[I],
        mode: PaddingMode = ConstantPadding(Some(Tensor.zeros[Int](Shape())))
    ): Output[T] = {
      Basic.pad(output, paddings, mode)
    }

    /** $OpDocBasicReshape
      *
      * @group BasicOps
      * @param  shape Shape of the output tensor.
      * @return Result as a new tensor.
      */
    def reshape[I: TF : IsIntOrLong](shape: Output[I]): Output[T] = {
      Basic.reshape(output, shape)
    }

    /** $OpDocBasicTranspose
      *
      * @group BasicOps
      * @param  permutation Permutation of the input tensor dimensions.
      * @param  conjugate   If `true`, then the complex conjugate of the transpose result is returned.
      * @return Result as a new tensor.
      */
    def transpose[I: IntDefault : TF : IsIntOrLong](
        permutation: Output[I] = null,
        conjugate: Boolean = false
    ): Output[T] = {
      Basic.transpose(output, permutation, conjugate)
    }

    /** $OpDocBasicMatrixTranspose
      *
      * @group BasicOps
      * @param  conjugate If `true`, then the complex conjugate of the transpose result is returned.
      * @return Result as a new tensor.
      */
    def matrixTranspose(conjugate: Boolean = false): Output[T] = {
      Basic.matrixTranspose(output, conjugate)
    }

    /** $OpDocBasicInvertPermutation
      *
      * @group BasicOps
      * @return Result as a new tensor.
      */
    def invertPermutation(implicit ev: IsIntOrLong[T]): Output[T] = {
      Basic.invertPermutation(output)
    }

    /** $OpDocBasicReverse
      *
      * @group BasicOps
      * @param  axes Dimensions of the input tensor to reverse.
      * @return Result as a new tensor which has the same shape as `input`.
      */
    def reverse[I: TF : IsIntOrLong](axes: Output[I]): Output[T] = {
      Basic.reverse(output, axes)
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
    def reverseSequence[I: TF : IsIntOrLong](
        sequenceLengths: Output[I],
        sequenceAxis: Int,
        batchAxis: Int = 0
    ): Output[T] = {
      Basic.reverseSequence(output, sequenceLengths, sequenceAxis, batchAxis)
    }

    /** $OpDocBasicSpaceToBatch
      *
      * @group BasicOps
      * @param  blockSize Block size which must be greater than `1`.
      * @param  paddings  `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
      * @return Result as a new tensor.
      */
    def spaceToBatch[I: TF : IsIntOrLong](
        blockSize: Int,
        paddings: Output[I]
    ): Output[T] = {
      Basic.spaceToBatch(output, blockSize, paddings)
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
    def spaceToBatchND[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
        blockShape: Output[I1],
        paddings: Output[I2]
    ): Output[T] = {
      Basic.spaceToBatchND(output, blockShape, paddings)
    }

    /** $OpDocBasicBatchToSpace
      *
      * @group BasicOps
      * @param  blockSize Block size which must be greater than `1`.
      * @param  crops     `2`-dimensional tensor containing non-negative integers with shape `[2, 2]`.
      * @return Result as a new tensor.
      */
    def batchToSpace[I: TF : IsIntOrLong](
        blockSize: Int,
        crops: Output[I]
    ): Output[T] = {
      Basic.batchToSpace(output, blockSize, crops)
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
    def batchToSpaceND[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
        blockShape: Output[I1],
        crops: Output[I2]
    ): Output[T] = {
      Basic.batchToSpaceND(output, blockShape, crops)
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
    ): Output[T] = {
      Basic.spaceToDepth(output, blockSize, dataFormat)
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
    ): Output[T] = {
      Basic.depthToSpace(output, blockSize, dataFormat)
    }

    //endregion Tensor Manipulation Ops

    //region Tensor Masking Ops

    /** $OpDocBasicWhere
      *
      * @group BasicOps
      * @return Result as a new tensor.
      */
    def where(implicit ev: IsBooleanOrNumeric[T]): Output[Long] = {
      Basic.where(output)
    }

    /** $OpDocBasicBooleanMask
      *
      * @group BasicOps
      * @param  mask `K`-dimensional boolean tensor, where `K <= N` and `K` must be known statically.
      * @return Result as a new tensor.
      */
    def booleanMask(mask: Output[Boolean]): Output[T] = {
      Basic.booleanMask(output, mask)
    }

    /** $OpDocBasicSequenceMask
      *
      * @group BasicOps
      * @param  maxLength Scalar integer tensor representing the maximum length of each row. Defaults to the maximum
      *                   value in this tensor.
      * @return Result as a new tensor.
      */
    def sequenceMask(
        maxLength: Output[T] = null
    )(implicit ev: IsIntOrUInt[T]): Output[Boolean] = {
      Basic.sequenceMask(output, maxLength)
    }

    //endregion Tensor Masking Ops

    //region Tensor Counting and Set Ops

    /** $OpDocBasicUnique
      *
      * @group BasicOps
      * @return Tuple containing `output` and `indices`.
      */
    def unique[I1: TF : IsIntOrLong](
        axis: Output[I1]
    ): (Output[T], Output[Int]) = {
      Basic.unique(output, axis, indicesDataType = INT32)
    }

    /** $OpDocBasicUnique
      *
      * @group BasicOps
      * @param  indicesDataType Data type of the returned indices.
      * @return Tuple containing `output` and `indices`.
      */
    def unique[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
        axis: Output[I1],
        indicesDataType: DataType[I2]
    ): (Output[T], Output[I2]) = {
      Basic.unique(output, axis, indicesDataType)
    }

    /** $OpDocBasicUniqueWithCounts
      *
      * @group BasicOps
      * @return Tuple containing `output`, `indices`, and `counts`.
      */
    def uniqueWithCounts[I1: TF : IsIntOrLong](
        axis: Output[I1]
    ): (Output[T], Output[Int], Output[Int]) = {
      Basic.uniqueWithCounts(output, axis, indicesDataType = INT32)
    }

    /** $OpDocBasicUniqueWithCounts
      *
      * @group BasicOps
      * @param  indicesDataType Data type of the returned indices.
      * @return Tuple containing `output`, `indices`, and `counts`.
      */
    def uniqueWithCounts[I1: TF : IsIntOrLong, I2: TF : IsIntOrLong](
        axis: Output[I1],
        indicesDataType: DataType[I2]
    ): (Output[T], Output[I2], Output[I2]) = {
      Basic.uniqueWithCounts(output, axis, indicesDataType)
    }

    /** $OpDocBasicListDiff
      *
      * @group BasicOps
      * @param  other One-dimensional tensor containing the values to remove.
      * @return Tuple containing `output` and `indices`, from the method description.
      */
    def listDiff(other: Output[T]): (Output[T], Output[Int]) = {
      Basic.listDiff(output, other, INT32)
    }

    /** $OpDocBasicListDiff
      *
      * @group BasicOps
      * @param  other           One-dimensional tensor containing the values to remove.
      * @param  indicesDataType Data type to use for the output indices of this op.
      * @return Tuple containing `output` and `indices`, from the method description.
      */
    def listDiff[I: TF : IsIntOrLong](
        other: Output[T],
        indicesDataType: DataType[I]
    ): (Output[T], Output[I]) = {
      Basic.listDiff(output, other, indicesDataType)
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
    def gather[I: TF : IsIntOrLong](
        indices: Output[I],
        axis: Output[I] = null
    ): Output[T] = {
      Basic.gather(output, indices, axis)
    }

    /** $OpDocBasicGatherND
      *
      * @group BasicOps
      * @param  indices Tensor containing indices to gather.
      * @return Result as a new tensor which contains the values from `input` gathered from indices given by `indices`,
      *         with shape `indices.shape(::-1) + input.shape(indices.shape(-1)::)`.
      */
    def gatherND[I: TF : IsIntOrLong](indices: Output[I]): Output[T] = {
      Basic.gatherND(output, indices)
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
    )(implicit ev: IsDecimal[T]): Output[T] = {
      Basic.checkNumerics(output)
    }

    /** $OpDocBasicOneHot
      *
      * @group BasicOps
      * @param  depth    Scalar tensor defining the depth of the one-hot dimension.
      * @param  onValue  Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] = i`.
      *                  Defaults to the value `1` with type `dataType`.
      * @param  offValue Scalar tensor defining the value to fill in the output `i`th value, when `indices[j] != i`.
      *                  Defaults to the value `0` with type `dataType`.
      * @param  axis     Axis to fill. Defaults to `-1`, representing the last axis.
      * @tparam R Data type of the output tensor. If not provided, the function will attempt to assume the data type
      *           of `onValue` or `offValue`, if one or both are passed in. If none of `onValue`, `offValue`, or
      *           `dataType` are provided, `dataType` will default to the `FLOAT32` data type.
      * @return Created op output.
      */
    def oneHot[R: TF](
        depth: Output[Int],
        onValue: Output[R] = null,
        offValue: Output[R] = null,
        axis: Int = -1
    )(implicit ev: IsIntOrLongOrUByte[T]): Output[R] = {
      Basic.oneHot[R, T](output, depth, onValue, offValue, axis)
    }

    //endregion Tensor Ungrouped Ops

    //region Tensor Broadcasting Ops

    /** $OpDocBasicBroadcastTo
      *
      * @group BasicOps
      * @param  shape Shape to broadcast the provided tensor to.
      * @return Created op output.
      */
    def broadcastTo[I: TF : IsIntOrLong](
        shape: Output[I]
    ): Output[T] = {
      Basic.broadcastTo(output, shape)
    }

    //endregion Tensor Broadcasting Ops

    //region Tensor Gradient Ops

    /** $OpDocBasicStopGradient
      *
      * @group BasicOps
      * @return Result as a new tensor which has the same value as this tensor.
      */
    def stopGradient(): Output[T] = {
      Basic.stopGradient(output)
    }

    /** $OpDocBasicPreventGradient
      *
      * @group BasicOps
      * @param  message Message to print along with the error.
      * @return Result as a new tensor which has the same value as this tensor.
      */
    def preventGradient(message: String = ""): Output[T] = {
      Basic.preventGradient(output, message)
    }

    //endregion Tensor Gradient Ops
  }
}
