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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64}
import org.platanios.tensorflow.jni.generated.tensors.{Sparse => NativeTensorOpsSparse}

/** Represents a sparse tensor.
  *
  * TensorFlow represents a sparse tensor as three separate dense tensors: `indices`, `values`, and `denseShape`. In
  * Scala, the three tensors are collected into a [[SparseTensor]] class for ease of use.  If you have separate
  * `indices`, `values`, and `denseShape` tensors, wrap them in a `SparseTensor` object before passing to the
  * relevant sparse tensor manipulation.
  *
  * Concretely, the sparse tensor `SparseTensor(indices, values, denseShape)` comprises the following components,
  * where `N` and `rank` are the number of values and number of dimensions in the [[SparseTensor]], respectively:
  *
  *   - `indices`: Two-dimensional [[INT64]] tensor with shape `[N, rank]`, which specifies the indices of the elements
  *     in the sparse tensor that have nonzero values (elements are zero-indexed). For example,
  *     `indices = [[1, 3], [2, 4]]` specifies that the elements with indexes `[1, 3]` and `[2, 4]` have nonzero
  *     values.
  *   - `values`: One-dimensional tensor of any type, with shape `[N]`, which supplies the values for each element in
  *     `indices`. For example, given `indices = [[1, 3], [2, 4]]`, the parameter `values = [18, 3.6]` specifies that
  *      element `[1, 3]` of the sparse tensor has a value of `18`, and element `[2, 4]` of the tensor has a value of
  *      `3.6`.
  *   - `denseShape`: One-dimensional [[INT64]] tensor with shape `[rank]`, which specifies the dense shape of the
  *     sparse tensor.  For example, `denseShape = [3, 6]` specifies a two-dimensional 3x6 tensor,
  *     `denseShape = [2, 3, 4]` specifies a three-dimensional 2x3x4 tensor, and `denseShape = [9]` specifies a
  *     one-dimensional tensor with 9 elements.
  *
  * The corresponding dense tensor, `dense`, satisfies:
  * {{{
  *   dense.shape == denseShape
  *   dense(indices(i)) = values(i) // Using a somewhat loose notation with respect to indexing.
  * }}}
  *
  * IMPORTANT NOTE: By convention, `indices` should be sorted in row-major order (or equivalently lexicographic order
  * on `indices(i)`). This is not enforced when `SparseTensor` objects are constructed, but most ops assume correct
  * ordering. If the ordering of sparse tensor `st` is wrong, a fixed version can be obtained by calling
  * `sparseReorder(st)`.
  *
  * For example, the sparse tensor `SparseTensor(indices = [[0, 0], [1, 2]], values = [1, 2], denseShape = [3, 4])`,
  * represents the dense tensor `[[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]`.
  *
  * @param  indices    Two-dimensional [[INT32]] tensor with shape `[N, rank]`.
  * @param  values     One-dimensional tensor with shape `[N]`.
  * @param  denseShape One-dimensional [[INT32]] tensor with shape `[rank]`.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class SparseTensor[+D <: DataType](
    indices: Tensor[INT64],
    values: Tensor[D],
    denseShape: Tensor[INT64]
) extends TensorLike[D] {
  Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
  Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

  /** Data type of this sparse op output. */
  override val dataType: D = values.dataType

  /** Shape of this sparse tensor. */
  override val shape: Shape = Shape(denseShape.cast(INT32).entriesIterator.toSeq: _*)

  /** Device on which this sparse op output will be placed. */
  override val device: String = values.device

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  override def toTensor: Tensor[D] = toTensor()

  /** Converts this sparse tensor to a dense tensor.
    *
    * The constructed op builds a tensor `dense` with shape `input.denseShape`, such that:
    * {{{
    *   // If input.indices is scalar:
    *   dense(i) ==> (i == input.indices ? input.values : defaultValue)
    *
    *   // If input.indices is a vector, then for each i:
    *   dense(input.indices(i)) ==> input.values(i)
    *
    *   // If input.indices is an n by d matrix, then for each i in [0, n):
    *   dense(input.indices(i)(0), ..., input.indices(i)(d-1)) ==> input.values(i)
    * }}}
    *
    * All other values in `dense` are set to `defaultValue`. If `input.values` is a scalar, then all sparse indices
    * are set to that single value.
    *
    * `input.indices` should be sorted in lexicographic order and they must not contain any repeats. If
    * `validateIndices` is `true`, then these properties are checked during execution.
    *
    * @param  defaultValue    Scalar tensor with the same data type as `input.values`, containing the value set for
    *                         indices that are not specified in `input.indices`.
    * @param  validateIndices If `true`, the indices in `input.indices` are checked to make sure that they are sorted in
    *                         lexicographic order and that there are no repeats.
    * @return Result as a new tensor, with the same data type as `input.values` and shape `input.denseShape`.
    */
  def toTensor[DD >: D <: DataType](
      defaultValue: Tensor[DD] = Tensor.zeros(dataType, Shape()),
      validateIndices: Boolean = true
  ): Tensor[DD] = {
    Tensor.fromNativeHandle[DD](NativeTensorOpsSparse.sparseToDense(
      executionContext.value.nativeHandle, indices.nativeHandle, denseShape.nativeHandle, values.nativeHandle,
      defaultValue.nativeHandle, validateIndices))
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  @throws[UnsupportedOperationException]
  override def toTensorIndexedSlices: TensorIndexedSlices[D] = {
    throw new UnsupportedOperationException(s"Cannot convert sparse tensor '$this' to tensor indexed slices.")
  }

  override def toString: String = {
    s"SparseTensor(values = $values, indices = $indices, denseShape = $denseShape, device = $device)}"
  }
}
