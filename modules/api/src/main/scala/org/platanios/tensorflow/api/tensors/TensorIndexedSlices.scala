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
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.ops.Math
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64}

/** Sparse representation of a set of tensor slices at given indices.
  *
  * This class if a simple wrapper for a pair (or a set of three) of [[Tensor]] objects:
  *   - `indices`: A one-dimensional integer [[Tensor]] with shape `[D0]`.
  *   - `values`: An [[Tensor]] of any data type, with shape `[D0, D1, ..., Dn]`.
  *   - `denseShape`: Optionally, an integer [[Tensor]] with shape `[LARGE0, D1, ..., Dn]`.
  *
  * An [[TensorIndexedSlices]] is typically used to represent a subset of a larger [[Output]], `dense`, of shape
  * `[LARGE0, D1, ..., Dn]`, where `LARGE0 >> D0`. The values in `indices` are the indices in the first dimension of
  * the slices that have been extracted from the larger tensor.
  *
  * The dense [[Tensor]], `dense`, represented by [[TensorIndexedSlices]], `slices`, has:
  * {{{
  *   dense(slices.indices(i), ::, ::, ...) = slices.values(i, ::, ::, ...)
  * }}}
  *
  * The [[TensorIndexedSlices]] class is used primarily in the definition of gradients for operations that have
  * sparse gradients, such as `gather`.
  *
  * Note that this is different than [[SparseTensor]] which uses multi-dimensional indices and scalar values.
  *
  * @param  indices    Indices along the first dimension of the corresponding dense [[Tensor]].
  * @param  values     Values corresponding to the provided indices.
  * @param  denseShape Shape of the corresponding dense [[Tensor]].
  *
  * @author Emmanouil Antonios Platanios
  */
final case class TensorIndexedSlices[+D <: DataType](
    indices: Tensor[INT64],
    values: Tensor[D],
    denseShape: Tensor[INT64] = null
) extends TensorLike[D] {
  /** Data type of these tensor indexed slices. */
  override val dataType: D = values.dataType

  /** Shape of these tensor indexed slices. */
  override val shape: Shape = Shape(denseShape.cast(INT32).entriesIterator.toSeq: _*)

  /** Device on which these tensor indexed slices will be placed. */
  override val device: String = values.device

  /** Returns the [[Tensor]] that this [[TensorLike]] object represents. */
  @throws[IllegalStateException]
  override def toTensor: Tensor[D] = {
    if (denseShape != null)
      throw new IllegalStateException(
        s"Conversion of 'TensorIndexedSlices', '$this', " +
            s"which has no dense shape information available, is not possible.")
    if (denseShape.prod().scalar > 100000000)
      Tensor.logger.warn(
        "Converting large (> 100000000 elements) tensor indexed slices object to a tensor " +
            "(may consume too much memory).")
    Math.unsortedSegmentSum(data = values, segmentIndices = indices, segmentsNumber = denseShape(0).cast(INT32))
  }

  /** Returns an [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    *
    * @return [[TensorIndexedSlices]] that has the same value as this [[TensorLike]].
    */
  override def toTensorIndexedSlices: TensorIndexedSlices[D] = this

  override def toString: String = {
    s"TensorIndexedSlices(values = $values, indices = $indices, denseShape = $denseShape, device = $device)}"
  }
}
