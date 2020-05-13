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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types.{IsIntOrLong, IsNumeric, IsReal, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.tensors.Tensor

/** Contains functions for constructing ops related to sparse tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Sparse {
  // TODO: [OPS] Add gradients and missing ops. There exist ops we can use such as `SparseAddGrad`.

  /** $OpDocSparseSparseAdd
    *
    * @group SparseOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  threshold Sparsity threshold.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def add[T: TF : IsNumeric, TR: TF : IsReal](
      x: SparseOutput[T],
      y: SparseOutput[T],
      threshold: Output[TR],
      name: String = "SparseAdd"
  ): SparseOutput[T] = {
    Op.Builder[(SparseOutput[T], SparseOutput[T], Output[TR]), SparseOutput[T]](
      opType = "SparseAdd",
      name = name,
      input = (x, y, threshold)
    ).build().output
  }

  /** $OpDocSparseSparseDenseAdd
    *
    * @group SparseOps
    * @param  x    Sparse input tensor.
    * @param  y    Dense input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def denseAdd[T: TF : IsNumeric](
      x: SparseOutput[T],
      y: Output[T],
      name: String = "SparseDenseAdd"
  ): Output[T] = {
    Op.Builder[(SparseOutput[T], Output[T]), SparseOutput[T]](
      opType = "SparseTensorDenseAdd",
      name = name,
      input = (x, y)
    ).build().output
  }

  /** $OpDocSparseReorder
    *
    * @group SparseOps
    * @param  sparseInput Sparse input tensor.
    * @param  name        Name prefix for the created ops.
    * @return Sparse tensor with the same shape as `sparseInput`, but in canonical ordering.
    */
  def reorder[T: TF](
      sparseInput: SparseOutput[T],
      name: String = "SparseReorder"
  ): SparseOutput[T] = {
    Op.nameScope(name) {
      val (reorderedIndices, reorderedValues) = Op.Builder[SparseOutput[T], (Output[Long], Output[T])](
        opType = "SparseReorder",
        name = name,
        input = sparseInput
      ).build().output
      if (sparseInput.shape.isFullyDefined)
        SparseOutput(reorderedIndices, reorderedValues, sparseInput.shape.toOutput.toLong)
      else
        SparseOutput(reorderedIndices, reorderedValues, Basic.identity(sparseInput.denseShape))
    }
  }

  /** $OpDocSparseMerge
    *
    * @group SparseOps
    * @param  sparseIndices Sparse tensors containing the sparse indices.
    * @param  sparseValues  Sparse tensor of any type.
    * @param  depths        Sequence of tensors containing the new sizes for the last dimensions, with
    *                       `0 <= sparseIndices(i).values < depths(i)`, for all `i`.
    * @param  alreadySorted Boolean that indicates whether the per-batch values in `sparseValues` are already sorted.
    *                       If so, sorting is skipped.
    * @param  name          Name prefix for the created ops.
    * @return Sparse tensor with the same shape as `sparseInput`, but in canonical ordering.
    */
  @throws[InvalidArgumentException]
  def merge[T: TF, I: TF: IsIntOrLong](
      sparseIndices: Seq[SparseOutput[I]],
      sparseValues: SparseOutput[T],
      depths: Seq[Tensor[Long]],
      alreadySorted: Boolean = false,
      name: String = "SparseMerge"
  ): SparseOutput[T] = {
    if (sparseIndices.length != depths.length)
      throw InvalidArgumentException("The provided 'sparseIndices' and 'depths' must have the same length.")
    Op.nameScope(name) {
      val indices = sparseIndices.map(_.values.toLong.expandDims(axis = 1))

      // To obtain the new indices, we slice off the last dimension of the sparse indices,
      // and then we tack on the indices.
      val indicesColumnsToPreserve = sparseIndices.head.indices(::, 0 :: -1)

      val result = SparseOutput(
        indices = Basic.concatenate(indicesColumnsToPreserve +: indices, axis = 1),
        values = sparseValues.values,
        denseShape = Basic.concatenate(sparseIndices.head.denseShape(0 :: -1) +: depths.map(_.toOutput), axis = 0))

      if (alreadySorted)
        result
      else
        reorder(result)
    }
  }
}

object Sparse extends Sparse {
  /** @define OpDocSparseSparseAdd
    *   The `sparseAdd` op adds two sparse tensors, producing another sparse tensor.
    *
    *   The input sparse tensor objects' indices are assumed ordered in standard lexicographic order. If this is not the
    *   case, before this op, add a `sparseReorder` op to restore index ordering.
    *
    *   By default, if two values sum to zero at some index, the output sparse tensor would still include that
    *   particular location in its index, storing a zero in the corresponding value slot. To override this, callers can
    *   specify `threshold`, indicating that if the sum has a magnitude strictly smaller than `threshold`, its
    *   corresponding value and index would then not be included. In particular, `threshold == 0` (default) means that
    *   everything is kept and actual thresholding happens only for positive values.
    *
    * @define OpDocSparseSparseDenseAdd
    *   The `sparseDenseAdd` op adds a dense tensor to a sparse tensor, producing a dense tensor.
    *
    *   The input sparse tensor's indices are not required to be ordered in any particular way.
    * 
    * @define OpDocSparseReorder
    *   The `sparse.reorder` op reorders a sparse tensor into the canonical, row-major ordering.
    *   
    *   Note that by convention, all sparse ops preserve the canonical ordering along increasing dimension number. The 
    *   only time ordering can be violated is during manual manipulation of the indices and values to add entries. 
    *   Reordering does not affect the shape of the sparse tensor. For example, if `sparseInput` has shape `[4, 5]` and 
    *   `indices` / `values`:
    *   {{{
    *     [0, 3]: b
    *     [0, 1]: a
    *     [3, 1]: d
    *     [2, 0]: c
    *   }}}
    *   then the output will be a sparse tensor with shape `[4, 5]` and `indices` / `values`:
    *   {{{
    *     [0, 1]: a
    *     [0, 3]: b
    *     [2, 0]: c
    *     [3, 1]: d
    *   }}}
    * 
    * @define OpDocSparseMerge
    *   The `sparse.merge` op combines a batch of feature indices and values into a single sparse tensor.
    * 
    *   The most common use case for this op occurs when feature indices and their corresponding values are stored in 
    *   `Example` protos on disk. The `parseExample` op will return a batch of indices and a batch of values, and this 
    *   op joins them into a single logical sparse tensor which has the following properties:
    * 
    *     - `indices` is equivalent to `sparseIndices.indices` with the last dimension discarded 
    *        and replaced by `sparseIndices.values`.
    *     - `values` is simply `sparseValues.values`.
    *     - If `sparseIndices.denseShape = [D0, D1, ..., Dn, K]`, then `output.shape = [D0, D1, ..., Dn, depths]`.
    * 
    *   For example, consider the following feature vectors:
    *   {{{
    *     vector1 = [-3, 0, 0, 0, 0, 0]
    *     vector2 = [ 0, 1, 0, 4, 1, 0]
    *     vector3 = [ 5, 0, 0, 9, 0, 0]
    *   }}}
    *   These might be stored sparsely in the following `Example` protos by storing only the feature indices (column 
    *   number if the vectors are treated as a single matrix) of the non-zero elements and their corresponding values:
    *   {{{
    *     examples = [Example(features={
    *                     "ids": Feature(int64_list=Int64List(value=[0])),
    *                     "values": Feature(float_list=FloatList(value=[-3]))}),
    *                 Example(features={
    *                     "ids": Feature(int64_list=Int64List(value=[1, 4, 3])),
    *                     "values": Feature(float_list=FloatList(value=[1, 1, 4]))}),
    *                 Example(features={
    *                     "ids": Feature(int64_list=Int64List(value=[0, 3])),
    *                     "values": Feature(float_list=FloatList(value=[5, 9]))})]
    *   }}}
    *   The result of calling `parseExample` on these examples will produce a map with entries for `indices` and 
    *   `values`. Passing those two objects to this op along with `depths = Seq(6)`, will produce a sparse tensor that
    *   sparsely represents all three instances. Namely, the `indices` property will contain the coordinates of the 
    *   non-zero entries in the feature matrix (the first dimension is the row number in the matrix, i.e., the index 
    *   within the batch, and the second dimension is the column number, i.e., the feature id); `values` will contain 
    *   the actual values. `shape` will be the shape of the original matrix, i.e., `(3, 6)`. For our example above, the 
    *   output will be equal to:
    *   {{{
    *     SparseOutput(
    *       indices = [[0, 0], [1, 1], [1, 3], [1, 4], [2, 0], [2, 3]],
    *       values = [-3, 1, 4, 1, 5, 9],
    *       denseShape = [3, 6])
    *   }}}
    *   This method generalizes to higher-dimensions by simply providing a sequence for both the `sparseIndices` as well 
    *   as the `depths`. In this case, the resulting sparse tensor has the following properties:
    * 
    *     - `indices` is equivalent to `sparseIndices.head.indices` with the last dimension discarded 
    *        and replaced by `sparseIndices(0).values`, `sparseIndices(1).values`, ....
    *     - `values` is simply `sparseValues.values`.
    *     - If `sparseIndices.denseShape = [D0, D1, ..., Dn, K]`, then `output.shape = [D0, D1, ..., Dn] + depths`.
    */
  private[ops] trait Documentation
}
