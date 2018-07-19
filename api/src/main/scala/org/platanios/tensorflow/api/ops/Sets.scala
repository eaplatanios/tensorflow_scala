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

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types._

/** Contains functions for constructing ops related to sets.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Sets {
  /** $OpDocSetsSetSize
    *
    * @group SetOps
    * @param  input           Input tensor with indices sorted in row-major order.
    * @param  validateIndices Boolean indicator specifying whether to validate the order and range of the indices of
    *                         `input`.
    * @param  name            Name for the created op.
    * @return `INT32` tensor containing the set sizes.
    * @throws InvalidDataTypeException If `input` has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  def setSize(
      input: SparseOutput,
      validateIndices: Boolean = true,
      name: String = "SetSize"
  ): Output = {
    if (!Sets.supportedDataTypes.contains(input.dataType))
      throw InvalidDataTypeException(s"Unsupported data type: ${input.dataType}.")
    Op.Builder("SetSize", name)
        .addInput(input.indices)
        .addInput(input.values)
        .addInput(input.denseShape)
        .setAttribute("validate_indices", validateIndices)
        .build().outputs(0)
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
    * @throws InvalidDataTypeException If any of the input tensors has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  def setIntersection[A, B](
      a: A,
      b: B,
      validateIndices: Boolean = true,
      name: String = "SetIntersection"
  )(implicit ev: SetOps.Aux[A, B]): SparseOutput = {
    if (!Sets.supportedDataTypes.contains(ev.dataTypeA(a)))
      throw InvalidDataTypeException(s"Unsupported data type for 'a': ${ev.dataTypeA(a)}.")
    if (!Sets.supportedDataTypes.contains(ev.dataTypeB(b)))
      throw InvalidDataTypeException(s"Unsupported data type for 'b': ${ev.dataTypeB(b)}.")
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
    * @throws InvalidDataTypeException If any of the input tensors has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  def setDifference[A, B](
      a: A,
      b: B,
      aMinusB: Boolean = true,
      validateIndices: Boolean = true,
      name: String = "SetDifference"
  )(implicit ev: SetOps.Aux[A, B]): SparseOutput = {
    if (!Sets.supportedDataTypes.contains(ev.dataTypeA(a)))
      throw InvalidDataTypeException(s"Unsupported data type for 'a': ${ev.dataTypeA(a)}.")
    if (!Sets.supportedDataTypes.contains(ev.dataTypeB(b)))
      throw InvalidDataTypeException(s"Unsupported data type for 'b': ${ev.dataTypeB(b)}.")
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
    * @throws InvalidDataTypeException If any of the input tensors has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  def setUnion[A, B](
      a: A,
      b: B,
      validateIndices: Boolean = true,
      name: String = "SetUnion"
  )(implicit ev: SetOps.Aux[A, B]): SparseOutput = {
    if (!Sets.supportedDataTypes.contains(ev.dataTypeA(a)))
      throw InvalidDataTypeException(s"Unsupported data type for 'a': ${ev.dataTypeA(a)}.")
    if (!Sets.supportedDataTypes.contains(ev.dataTypeB(b)))
      throw InvalidDataTypeException(s"Unsupported data type for 'b': ${ev.dataTypeB(b)}.")
    ev.applyOperation(a, b, "union", validateIndices, name)
  }
}

private[api] object Sets extends Sets {
  private[Sets] val supportedDataTypes: Set[DataType] = Set(INT8, INT16, INT32, INT64, UINT8, UINT16, STRING)

  private[Sets] def setOperation[A, B](
      a: A,
      b: B,
      operation: String,
      validateIndices: Boolean = true,
      name: String
  )(implicit ev: SetOps.Aux[A, B]): SparseOutput = {
    ev.applyOperation(a, b, operation, validateIndices, name)
  }

  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("SetSize")
    GradientsRegistry.registerNonDifferentiable("DenseToDenseSetOperation")
    GradientsRegistry.registerNonDifferentiable("DenseToSparseSetOperation")
    GradientsRegistry.registerNonDifferentiable("SparseToSparseSetOperation")
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
  @inline def dataTypeA(a: A): DataType
  @inline def dataTypeB(b: ArgType): DataType
  @inline def applyOperation(
      a: A,
      b: ArgType,
      operation: String,
      validateIndices: Boolean = true,
      name: String
  ): SparseOutput
}

object SetOps {
  type Aux[A, B] = SetOps[A] {type ArgType = B}

  implicit val outputOutputSetOps: SetOps.Aux[Output, Output] = new SetOps[Output] {
    override type ArgType = Output
    @inline override def dataTypeA(a: Output): DataType = a.dataType
    @inline override def dataTypeB(b: Output): DataType = b.dataType
    @inline override def applyOperation(
        a: Output,
        b: Output,
        operation: String,
        validateIndices: Boolean,
        name: String
    ): SparseOutput = {
      val result = Op.Builder("DenseToDenseSetOperation", name)
          .addInput(a)
          .addInput(b)
          .setAttribute("set_operation", operation)
          .setAttribute("validate_indices", validateIndices)
          .build().outputs
      SparseOutput(result(0), result(1), result(2))
    }
  }

  implicit val outputSparseOutputSetOps: SetOps.Aux[Output, SparseOutput] = new SetOps[Output] {
    override type ArgType = SparseOutput
    @inline override def dataTypeA(a: Output): DataType = a.dataType
    @inline override def dataTypeB(b: SparseOutput): DataType = b.dataType
    @inline override def applyOperation(
        a: Output,
        b: SparseOutput,
        operation: String,
        validateIndices: Boolean,
        name: String
    ): SparseOutput = {
      val result = Op.Builder("DenseToSparseSetOperation", name)
          .addInput(a)
          .addInput(b.indices)
          .addInput(b.values)
          .addInput(b.denseShape)
          .setAttribute("set_operation", operation)
          .setAttribute("validate_indices", validateIndices)
          .build().outputs
      SparseOutput(result(0), result(1), result(2))
    }
  }

  implicit val sparseOutputSparseOutputSetOps: SetOps.Aux[SparseOutput, SparseOutput] = new SetOps[SparseOutput] {
    override type ArgType = SparseOutput
    @inline override def dataTypeA(a: SparseOutput): DataType = a.dataType
    @inline override def dataTypeB(b: SparseOutput): DataType = b.dataType
    @inline override def applyOperation(
        a: SparseOutput,
        b: SparseOutput,
        operation: String,
        validateIndices: Boolean,
        name: String
    ): SparseOutput = {
      val result = Op.Builder("SparseToSparseSetOperation", name)
          .addInput(a.indices)
          .addInput(a.values)
          .addInput(a.denseShape)
          .addInput(b.indices)
          .addInput(b.values)
          .addInput(b.denseShape)
          .setAttribute("set_operation", operation)
          .setAttribute("validate_indices", validateIndices)
          .build().outputs
      SparseOutput(result(0), result(1), result(2))
    }
  }
}
