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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.types.{IsNumeric, IsReal, TF}

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
  def sparseAdd[T: TF : IsNumeric, TR: TF : IsReal](
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
  def sparseDenseAdd[T: TF : IsNumeric](
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
}

object Sparse extends Sparse {
  private[ops] trait Implicits {
    implicit def outputConvertibleToSparseOps[OC, T: TF](
        value: OC
    )(implicit f: OC => SparseOutput[T]): SparseOps[T] = {
      new SparseOps(f(value))
    }

    implicit class SparseOps[T: TF](val sparseOutput: SparseOutput[T]) {
      def +(other: SparseOutput[T])(implicit ev: IsNumeric[T]): SparseOutput[T] = {
        add(other, threshold = 0)
      }

      def +(other: Output[T])(implicit ev: IsNumeric[T]): Output[T] = {
        addDense(other)
      }

      /** $OpDocSparseSparseAdd
        *
        * @group SparseOps
        * @param  other     Tensor to add to the current one.
        * @param  threshold Sparsity threshold.
        * @param  name      Name for the created op.
        * @return Created op output.
        */
      def add[TR: TF : IsReal](
          other: SparseOutput[T],
          threshold: Output[TR],
          name: String = "SparseAdd"
      )(implicit ev: IsNumeric[T]): SparseOutput[T] = {
        Sparse.sparseAdd(sparseOutput, other, threshold, name)
      }

      /** $OpDocSparseSparseDenseAdd
        *
        * @group SparseOps
        * @param  other Dense tensor to add to the current one.
        * @param  name  Name for the created op.
        * @return Created op output.
        */
      def addDense(
          other: Output[T],
          name: String = "SparseDenseAdd"
      )(implicit ev: IsNumeric[T]): Output[T] = {
        Sparse.sparseDenseAdd(sparseOutput, other, name)
      }
    }
  }

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
    */
  private[ops] trait Documentation
}
