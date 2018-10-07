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

import org.platanios.tensorflow.api.types._

/** Contains functions for constructing ops related to sets.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Sets {
  /** $OpDocSetsSetSize
    *
    * @group SetOps
    * @param  input           Input tensor with indices sorted in row-major order.
    * @param  validateIndices Boolean indicator specifying whether to validate the order and range of the indices of
    *                         `input`.
    * @param  name            Name for the created op.
    * @return Tensor containing the set sizes.
    */
  def setSize[T: TF : IsIntOrUInt](
      input: SparseOutput[T],
      validateIndices: Boolean = true,
      name: String = "SetSize"
  ): Output[Int] = {
    Op.Builder[SparseOutput[T], Output[Int]](
      opType = "SetSize",
      name = name,
      input = input
    ).setAttribute("validate_indices", validateIndices)
        .build().output
  }

  /** $OpDocSetsSetIntersection
    *
    * @group SetOps
    * @param  a               First input tensor.
    * @param  b               Second input tensor.
    * @param  validateIndices Boolean indicator specifying whether to validate the order and range of the indices of
    *                         `input`.
    * @param  name            Name for the created op.
    * @return Sparse tensor containing the result of the operation.
    */
  def setIntersection[A, B, T](
      a: A,
      b: B,
      validateIndices: Boolean = true,
      name: String = "SetIntersection"
  )(implicit
      ev: SetOps.Aux[A, B, T],
      evSupported: TF[T]
  ): SparseOutput[T] = {
    ev.applyOperation(a, b, "intersection", validateIndices, name)
  }

  /** $OpDocSetsSetDifference
    *
    * @group SetOps
    * @param  a               First input tensor.
    * @param  b               Second input tensor.
    * @param  aMinusB         Boolean value specifying whether to subtract `b` from `a`, or vice-versa.
    * @param  validateIndices Boolean indicator specifying whether to validate the order and range of the indices of
    *                         `input`.
    * @param  name            Name for the created op.
    * @return Sparse tensor containing the result of the operation.
    */
  def setDifference[A, B, T](
      a: A,
      b: B,
      aMinusB: Boolean = true,
      validateIndices: Boolean = true,
      name: String = "SetDifference"
  )(implicit
      ev: SetOps.Aux[A, B, T],
      evSupported: TF[T]
  ): SparseOutput[T] = {
    if (aMinusB)
      ev.applyOperation(a, b, "a-b", validateIndices, name)
    else
      ev.applyOperation(a, b, "b-a", validateIndices, name)
  }

  /** $OpDocSetsSetUnion
    *
    * @group SetOps
    * @param  a               First input tensor.
    * @param  b               Second input tensor.
    * @param  validateIndices Boolean indicator specifying whether to validate the order and range of the indices of
    *                         `input`.
    * @param  name            Name for the created op.
    * @return Sparse tensor containing the result of the operation.
    */
  def setUnion[A, B, T](
      a: A,
      b: B,
      validateIndices: Boolean = true,
      name: String = "SetUnion"
  )(implicit
      ev: SetOps.Aux[A, B, T],
      evSupported: TF[T]
  ): SparseOutput[T] = {
    ev.applyOperation(a, b, "union", validateIndices, name)
  }
}

object Sets extends Sets {
  private[Sets] def setOperation[A, B, T](
      a: A,
      b: B,
      operation: String,
      validateIndices: Boolean = true,
      name: String
  )(implicit
      ev: SetOps.Aux[A, B, T],
      evSupported: TF[T]
  ): SparseOutput[T] = {
    ev.applyOperation(a, b, operation, validateIndices, name)
  }

  /** @define OpDocSetsSetSize
    *   The `setSize` op computes the number of unique elements along the last dimension of `input`.
    *
    *   For `input` with rank `n`, the op outputs a tensor with rank `n-1`, and the same 1st `n-1` dimensions as
    *   `input`. Each value is the number of unique elements in the corresponding `[0, ..., n-1]` dimension of `input`.
    *
    * @define OpDocSetsSetIntersection
    *   The `setIntersection` op computes the set intersection of elements in the last dimension of `a` and `b`.
    *
    *   All but the last dimension of `a` and `b` must match.
    *
    *   Note that supported input types are:
    *
    *     - `(a: SparseOutput, b: SparseOutput)`
    *     - `(a: Output, b: SparseOutput)`
    *     - `(a: Output, b: Output)`
    *
    *   For sparse tensors, the indices must be sorted in row-major order.
    *
    *   For example:
    *   {{{
    *     val a = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1))
    *       values = Tensor(1, 2, 3, 4, 5, 6),
    *       denseShape = Shape(2, 2, 2))
    *     val b = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1),
    *         Tensor(1, 1, 2),
    *         Tensor(1, 1, 3))
    *       values = Tensor(1, 4, 5, 6, 7, 8),
    *       denseShape = Shape(2, 2, 4))
    *     tf.setIntersection(a, b) ==>
    *       SparseTensor(
    *         indices = Tensor(
    *           Tensor(0, 0, 0),
    *           Tensor(1, 0, 0),
    *           Tensor(1, 1, 0),
    *           Tensor(1, 1, 1)),
    *         values = Tensor(1, 4, 5, 6))
    *   }}}
    *
    * @define OpDocSetsSetDifference
    *   The `setDifference` op computes the set difference of elements in the last dimension of `a` and `b`.
    *
    *   All but the last dimension of `a` and `b` must match.
    *
    *   Note that supported input types are:
    *
    *     - `(a: SparseOutput, b: SparseOutput)`
    *     - `(a: Output, b: SparseOutput)`
    *     - `(a: Output, b: Output)`
    *
    *   For sparse tensors, the indices must be sorted in row-major order.
    *
    *   For example:
    *   {{{
    *     val a = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1))
    *       values = Tensor(1, 2, 3, 4, 5, 6),
    *       denseShape = Shape(2, 2, 2))
    *     val b = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 0, 1),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1),
    *         Tensor(1, 1, 2),
    *         Tensor(1, 1, 3))
    *       values = Tensor(1, 3, 2, 4, 5, 5, 6, 7, 8),
    *       denseShape = Shape(2, 2, 4))
    *     tf.setDifference(a, b) ==>
    *       SparseTensor(
    *         indices = Tensor(
    *           Tensor(0, 0, 0),
    *           Tensor(0, 0, 1)),
    *         values = Tensor(2, 3))
    *   }}}
    *
    * @define OpDocSetsSetUnion
    *   The `setUnion` op computes the set union of elements in the last dimension of `a` and `b`.
    *
    *   All but the last dimension of `a` and `b` must match.
    *
    *   Note that supported input types are:
    *
    *     - `(a: SparseOutput, b: SparseOutput)`
    *     - `(a: Output, b: SparseOutput)`
    *     - `(a: Output, b: Output)`
    *
    *   For sparse tensors, the indices must be sorted in row-major order.
    *
    *   For example:
    *   {{{
    *     val a = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1))
    *       values = Tensor(1, 2, 3, 4, 5, 6),
    *       denseShape = Shape(2, 2, 2))
    *     val b = SparseOutput(
    *       indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 0, 1),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1),
    *         Tensor(1, 1, 2),
    *         Tensor(1, 1, 3))
    *       values = Tensor(1, 3, 2, 4, 5, 5, 6, 7, 8),
    *       denseShape = Shape(2, 2, 4))
    *     tf.setDifference(a, b) ==>
    *       SparseTensor(
    *         indices = Tensor(
    *         Tensor(0, 0, 0),
    *         Tensor(0, 0, 1),
    *         Tensor(0, 0, 2),
    *         Tensor(0, 1, 0),
    *         Tensor(1, 0, 0),
    *         Tensor(1, 0, 1),
    *         Tensor(1, 1, 0),
    *         Tensor(1, 1, 1),
    *         Tensor(1, 1, 2),
    *         Tensor(1, 1, 3))
    *       values = Tensor(1, 2, 3, 2, 3, 4, 5, 5, 6, 7, 8))
    *   }}}
    */
  private[ops] trait Documentation
}

/** Type trait specifying the supported types for set operation inputs. */
trait SetOps[A] {
  type ArgType
  type Type

  @inline def applyOperation(
      a: A,
      b: ArgType,
      operation: String,
      validateIndices: Boolean = true,
      name: String
  ): SparseOutput[Type]
}

object SetOps {
  type Aux[A, B, E] = SetOps[A] {
    type ArgType = B
    type Type = E
  }

  implicit def outputOutputSetOps[T: TF : IsIntOrUInt]: SetOps.Aux[Output[T], Output[T], T] = {
    new SetOps[Output[T]] {
      override type ArgType = Output[T]
      override type Type = T

      @inline override def applyOperation(
          a: Output[T],
          b: Output[T],
          operation: String,
          validateIndices: Boolean,
          name: String
      ): SparseOutput[T] = {
        Op.Builder[(Output[T], Output[T]), SparseOutput[T]](
          opType = "DenseToDenseSetOperation",
          name = name,
          input = (a, b)
        ).setAttribute("set_operation", operation)
            .setAttribute("validate_indices", validateIndices)
            .build().output
      }
    }
  }

  implicit def outputSparseOutputSetOps[T: TF : IsIntOrUInt]: SetOps.Aux[Output[T], SparseOutput[T], T] = {
    new SetOps[Output[T]] {
      override type ArgType = SparseOutput[T]
      override type Type = T

      @inline override def applyOperation(
          a: Output[T],
          b: SparseOutput[T],
          operation: String,
          validateIndices: Boolean,
          name: String
      ): SparseOutput[T] = {
        Op.Builder[(Output[T], SparseOutput[T]), SparseOutput[T]](
          opType = "DenseToSparseSetOperation",
          name = name,
          input = (a, b)
        ).setAttribute("set_operation", operation)
            .setAttribute("validate_indices", validateIndices)
            .build().output
      }
    }
  }

  implicit def sparseOutputSparseOutputSetOps[T: TF : IsIntOrUInt]: SetOps.Aux[SparseOutput[T], SparseOutput[T], T] = {
    new SetOps[SparseOutput[T]] {
      override type ArgType = SparseOutput[T]
      override type Type = T

      @inline override def applyOperation(
          a: SparseOutput[T],
          b: SparseOutput[T],
          operation: String,
          validateIndices: Boolean,
          name: String
      ): SparseOutput[T] = {
        Op.Builder[(SparseOutput[T], SparseOutput[T]), SparseOutput[T]](
          opType = "SparseToSparseSetOperation",
          name = name,
          input = (a, b)
        ).setAttribute("set_operation", operation)
            .setAttribute("validate_indices", validateIndices)
            .build().output
      }
    }
  }
}
