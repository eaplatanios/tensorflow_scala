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

/** Contains functions for constructing ops related to manipulating sparse tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Sparse {
  /** Creates an op that converts a sparse tensor to a dense tensor.
    *
    * The op builds a tensor `dense` with shape `input.denseShape`, such that:
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
    * All other values in `dense` are set to `defaultValue`. If `input.values` is a scalar, then all sparse indices are
    * set to that single value.
    *
    * `input.indices` should be sorted in lexicographic order and they must not contain any repeats. If
    * `validateIndices` is `true`, then these properties are checked during execution.
    *
    * @param  input           Sparse tensor to convert.
    * @param  defaultValue    Scalar tensor with the same data type as `input.values`, containing the value set for
    *                         indices that are not specified in `input.indices`.
    * @param  validateIndices If `true`, the indices in `input.indices` are checked to make sure that they are sorted in
    *                         lexicographic order and that there are no repeats.
    * @param  name            Name for the created op.
    * @return Created op output, with the same data type as `input.values` and shape `input.denseShape`.
    */
  def sparseToDense(
      input: SparseOutput, defaultValue: Output = 0, validateIndices: Boolean = true,
      name: String = "SparseToDense"): Output = {
    Op.Builder(opType = "SparseToDense", name = name)
        .addInput(input.indices)
        .addInput(input.denseShape)
        .addInput(input.values)
        .addInput(defaultValue)
        .setAttribute("validate_indices", validateIndices)
        .build().outputs(0)
  }
}

object Sparse extends Sparse
