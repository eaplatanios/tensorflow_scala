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

package org.platanios.tensorflow.api.ops

/** Contains functions for constructing ops related to sparse tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Sparse {
  /** $OpDocSparseSparseAdd
    *
    * @group SparseOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  threshold Sparsity threshold.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def sparseAdd(x: SparseOutput, y: SparseOutput, threshold: Output = 0, name: String = "SparseAdd"): SparseOutput = {
    val result = Op.Builder("SparseAdd", name)
        .addInput(x.indices)
        .addInput(x.values)
        .addInput(x.denseShape)
        .addInput(y.indices)
        .addInput(y.values)
        .addInput(y.denseShape)
        .addInput(threshold)
        .build().outputs
    SparseOutput(result(0), result(1), result(2))
  }

  /** $OpDocSparseSparseDenseAdd
    *
    * @group SparseOps
    * @param  x    Sparse input tensor.
    * @param  y    Dense input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sparseDenseAdd(x: SparseOutput, y: Output, name: String = "SparseDenseAdd"): Output = {
    Op.Builder("SparseTensorDenseAdd", name)
        .addInput(x.indices)
        .addInput(x.values)
        .addInput(x.denseShape)
        .addInput(y)
        .build().outputs(0)
  }
}

private[api] object Sparse extends Sparse {
  private[ops] trait Implicits {
    implicit def sparseOutputToNNOps(value: SparseOutput): SparseOps = SparseOps(value)
  }

  case class SparseOps private[ops](sparseOutput: SparseOutput) {
    def +(other: SparseOutput): SparseOutput = add(other)
    def +(other: Output): Output = addDense(other)

    /** $OpDocSparseSparseAdd
      *
      * @group SparseOps
      * @param  other     Tensor to add to the current one.
      * @param  threshold Sparsity threshold.
      * @param  name      Name for the created op.
      * @return Created op output.
      */
    def add(other: SparseOutput, threshold: Output = 0, name: String = "SparseAdd"): SparseOutput = {
      Sparse.sparseAdd(sparseOutput, other, threshold, name)
    }

    /** $OpDocSparseSparseDenseAdd
      *
      * @group SparseOps
      * @param  other Dense tensor to add to the current one.
      * @param  name  Name for the created op.
      * @return Created op output.
      */
    def addDense(other: Output, name: String = "SparseDenseAdd"): Output = {
      Sparse.sparseDenseAdd(sparseOutput, other, name)
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
