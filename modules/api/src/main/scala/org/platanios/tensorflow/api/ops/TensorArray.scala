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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.{DataType, INT64}

/** Class wrapping dynamic-sized, per-time-step, write-once tensor arrays.
  *
  * This class is meant to be used with dynamic iteration primitives such as `whileLoop` and `mapFunction`. It supports
  * gradient back-propagation via special "flow" control flow dependencies.
  *
  * Note that the name of the `TensorArray` (even if passed in) is uniquified automatically. Each time a new
  * `TensorArray` is created at runtime it is assigned its own name for the duration of the run. This avoids name
  * collisions if a `TensorArray` is created within a `whileLoop`.
  *
  * @param  handle                 Tensor handle to the tensor array.
  * @param  flow                   Float scalar tensor for the tensor array, used to control gradient flow.
  * @param  dataType               Data type of the tensor array elements.
  * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all elements
  *                                must have the same shape.
  * @param  elementShape           A [[Shape]] object specifying the shape constraints of each of the elements of the
  *                                tensor array. The shape need not be fully defined.
  * @param  colocateWithFirstWrite Boolean value indicating whether to place the tensor array on the same device as
  *                                the tensor used on its first write call (write operations include `write`,
  *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
  *                                determined by the op creation context available during its initialization.
  * @param  colocationOps          Used to keep track of what ops the tensor array should be colocated with.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorArray private (
    handle: Output,
    flow: Output,
    dataType: DataType[_],
    inferShape: Boolean,
    private[ops] var elementShape: Option[Shape],
    colocateWithFirstWrite: Boolean = true,
    private var colocationOps: Seq[Op] = null
) extends OutputConvertible {
  /** Changes the element shape of the array given a shape to merge with.
    *
    * @param  shape Shape to merge with.
    * @throws InvalidShapeException If the provided shape is not compatible with the current element shape.
    */
  @throws[InvalidShapeException]
  private def mergeElementShape(shape: Shape): Unit = {
    elementShape match {
      case Some(currentShape) =>
        if (!shape.isCompatibleWith(currentShape))
          throw InvalidShapeException(s"Expected shape '$currentShape' but got '$shape' (and inferShape = true).")
        elementShape = Some(currentShape.mergeWith(shape))
      case None => if (shape.rank != -1) elementShape = Some(shape)
    }
  }

  /** Returns a tensor array with the same content and properties as this one.
    *
    * @return New [[TensorArray]] object with a flow that ensures the control dependencies from the contexts will become
    *         control dependencies for writes, reads, etc. Use this object for all subsequent operations.
    */
  def identity: TensorArray = {
    TensorArray(handle, Basic.identity(flow), dataType, inferShape, elementShape, colocateWithFirstWrite, colocationOps)
  }

  /** Creates an op that reads an element from this tensor array.
    *
    * @param  index Position to read from, inside the tensor array.
    * @param  name  Name for the created op.
    * @return Tensor in the specified position of the tensor array.
    */
  def read(index: Output, name: String = "TensorArrayRead"): Output = {
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      val value = TensorArray.readOp(handle, index, flow, dataType, name)
      elementShape.foreach(value.setShape)
      value
    }
  }

  /** Creates an op that writes an element to this tensor array.
    *
    * @param  index Position to write to, inside the tensor array.
    * @param  value Tensor to write to the tensor array.
    * @param  name  Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  def write(index: Output, value: Output, name: String = "TensorArrayWrite"): TensorArray = {
    val writeFlow = maybeColocateWith(value.op)(TensorArray.writeOp(handle, index, value, flow, name))
    val returnValue = TensorArray(
      handle, writeFlow, dataType, inferShape, elementShape, colocateWithFirstWrite, colocationOps)
    if (inferShape)
      returnValue.mergeElementShape(value.shape)
    returnValue
  }

  /** Creates an op that gathers specific elements from this tensor array.
    *
    * Note that all elements selected by `indices` must have been written and must have the same shape.
    *
    * @param  indices One-dimensional tensor containing the positions in the tensor array from which to read tensor
    *                 elements.
    * @param  name    Name for the created op.
    * @return Tensor containing the gathered elements, concatenated along a new axis (the new dimension `0`).
    */
  def gather(indices: Output, name: String = "TensorArrayGather"): Output = {
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      val ind = if (indices.rank == 0) indices.expandDims(0) else indices
      val value = TensorArray.gatherOp(handle, ind, flow, dataType, elementShape.getOrElse(Shape.unknown()), name)
      if (elementShape.isDefined)
        value.setShape(Shape(-1 +: elementShape.get.asArray: _*))
      value
    }
  }

  /** Creates an op that scatters the provided elements along indices of this tensor array.
    *
    * Note that `indices` must be a vector and its length must match the first dimension of `value`.
    *
    * @param  indices One-dimensional tensor containing the positions in the tensor array at which to write the tensor
    *                 elements.
    * @param  value   Concatenated tensor to write to the tensor array.
    * @param  name    Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  def scatter(indices: Output, value: Output, name: String = "TensorArrayScatter"): TensorArray = {
    val scatterFlow = maybeColocateWith(value.op) {
      TensorArray.scatterOp(handle, indices, value, flow, name)
    }
    val returnValue = TensorArray(
      handle, scatterFlow, dataType, inferShape, elementShape, colocateWithFirstWrite, colocationOps)
    if (inferShape) {
      val valueShape = scatterFlow.inputs(2).shape
      val shape = if (valueShape != Shape.unknown()) Shape.fromSeq(valueShape.asArray.tail) else valueShape
      returnValue.mergeElementShape(shape)
    }
    returnValue
  }

  /** Creates an op that returns the elements in this tensor array as a stacked tensor.
    *
    * Note that all elements of this tensor array must have been written and must have the same shape.
    *
    * If the elements have rank `R`, then the returned tensor shape will be equal to `R + 1`.
    *
    * @param  name Name for the created op.
    * @return Stacked tensor.
    */
  def stack(name: String = "TensorArrayStack"): Output = {
    Op.createWithNameScope(name, Set(handle.op)) {
      Op.colocateWith(Set(handle.op), ignoreExisting = true) {
        gather(Math.range(Basic.constant(0), size()), name)
      }
    }
  }

  /** Creates an op that unstacks the values of a tensor in this tensor array.
    *
    * If the input value shapes have rank `R`, then the output tensor array will contain elements whose shapes have
    * rank `R - 1`.
    *
    * @param  value Tensor to unstack.
    * @param  name  Name for the created op.
    * @return New tensor array object with flow that ensures the unstack occurs. Use this object for all subsequent
    *         operations.
    */
  def unstack(value: Output, name: String = "TensorArrayUnstack"): TensorArray = {
    scatter(Math.range(Basic.constant(0), Basic.shape(value)(0)), value, name)
  }

  /** Creates an op that concatenates the elements of the tensor array.
    *
    * The op takes `T` elements with shapes `[n0, d0, d1, ...]`, `[n1, d0, d1, ...]`, ..., `[n(T-1), d0, d1, ...]` and
    * concatenates them into a tensor with shape `[n0 + n1 + ... + n(T-1), d0, d1, ...]`.
    *
    * All elements must have been written and must have the same shape, except for their first dimension.
    *
    * @param  name Name for the created op.
    * @return Tensor with all of the elements in the tensor array, concatenated along the first axis.
    */
  def concatenate(name: String = "TensorArrayConcatenate"): Output = {
    val shape = elementShape.map(s => Shape.fromSeq(s.asArray.tail)).getOrElse(Shape.unknown())
    val (value, _) = TensorArray.concatenateOp(handle, flow, dataType, shape, name)
    if (elementShape.isDefined)
      value.setShape(Shape(-1 +: shape.asArray: _*))
    value
  }

  /** Splits the values of a tensor into a tensor array.
    *
    * @param  input   (N+1)-dimensional tensor to split. Must have the same data type as this tensor array.
    * @param  lengths 1-D integer tensor with the lengths to use when splitting `input` along its first dimension.
    * @param  name    Name for the created op.
    * @return Tensor array with flow that ensures the split occurs. Use this object for all subsequent operations.
    */
  def split(input: Output, lengths: Output, name: String = "TensorArraySplit"): TensorArray = {
    Op.createWithNameScope(name, Set(handle.op, input.op, lengths.op)) {
      val splitFlow = maybeColocateWith(input.op)(TensorArray.splitOp(handle, input, lengths.cast(INT64), flow, name))
      val returnValue = TensorArray(
        handle, splitFlow, dataType, inferShape, elementShape, colocateWithFirstWrite, colocationOps)
      if (inferShape) {
        val valueShape = splitFlow.inputs(1).shape
        val lengths = Output.constantValue(splitFlow.inputs(2))
        val shape = {
          if (valueShape.rank != -1 && lengths.isDefined && lengths.get.max() == lengths.get.min())
            Shape.fromSeq(lengths.get(0).scalar.asInstanceOf[Long].toInt +: valueShape.asArray.tail)
          else
            Shape.unknown()
        }
        returnValue.mergeElementShape(shape)
      }
      returnValue
    }
  }

  /** Returns an op that gets the current size of the tensor array.
    *
    * @param  name Name for the created op.
    * @return Created op output, containing the current size of the tensor array.
    */
  def size(name: String = "TensorArraySize"): Output = {
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      TensorArray.sizeOp(handle, flow, name)
    }
  }

  /** Returns a tensor array for storing the gradients of the values stored in this tensor array.
    *
    * If the provided tensor array gradient already exists, then a reference to it is returned.
    *
    * This op locks the size of the original tensor array by disabling its dynamic size flag.
    *
    * ==A Note About the Input `flow`==
    *
    * The handle `flow` forces the execution of the gradient lookup to occur only after certain other operations have
    * occurred. For example, when the forward tensor array is dynamically sized, writes to this tensor array may resize
    * the object. The gradient tensor array is statically sized based on the size of the forward tensor array when this
    * operation executes. Furthermore, the size of the forward tensor array is frozen by this call. As a result, the
    * flow is used to ensure that the call to generate the gradient tensor array only happens after all writes are
    * executed.
    *
    * In the case of dynamically sized tensor arrays, the gradient computation should only be performed on read
    * operations that have themselves been chained via flow to occur only after all writes have executed. That way the
    * final size of the forward tensor array is known when this operation is called.
    *
    * ==A Note About the `source` Attribute==
    *
    * Tensor array gradient calls use an accumulator tensor array object. If multiple gradients are calculated and run
    * in the same session, then the multiple gradient nodes may accidentally flow though the same accumulator tensor
    * array. This double counts and generally breaks the tensor array gradient flow.
    *
    * The solution is to identify which gradient call this particular tensor array gradient is being called from. This
    * is performed by identifying a unique string (e.g. "gradients", "gradients_1", ...) from the input gradient
    * tensor's name. This string is used as a suffix when creating the tensor array gradient object here (the attribute
    * `source`).
    *
    * The attribute `source` is added as a suffix to the forward tensor array's name when performing the
    * creation/lookup, so that each separate gradient calculation gets its own tensor array accumulator.
    *
    * @param  source Gradient source string used to decide which gradient tensor array to return.
    * @param  flow   Float scalar that enforces proper chaining of operations.
    * @param  name   Name for the created gradient op.
    * @return Gradient tensor array.
    */
  private[api] def gradient(
      source: String,
      flow: Output = this.flow,
      name: String = "TensorArrayGradient"
  ): TensorArray = {
    // `TensorArray.gradientOp` requires a flow input when forward tensor arrays are dynamically sized. This forces the
    // creation of the gradient tensor array only once the final forward array's size is fixed.
    Op.createWithNameScope(name, Set(handle.op)) {
      Op.colocateWith(Set(handle.op), ignoreExisting = true) {
        val (gradientHandle, _) = TensorArray.gradientOp(handle, flow, source)
        val gradientFlow = Op.createWith(controlDependencies = Set(gradientHandle.op)) {
          Basic.identity(flow, name = "GradientFlow")
        }
        TensorArray(
          gradientHandle, gradientFlow, dataType, inferShape, elementShape, colocateWithFirstWrite = false)
      }
    }
  }

  /** Converts this tensor array to an output (i.e., dense symbolic tensor), by stacking it. */
  override def toOutput: Output = stack()

  /** Returns an op that deletes this tensor array from its resource container.
    *
    * This enables the user to close and release the resource in the middle of a step/run.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def close(name: String = "TensorArrayClose"): Op = {
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      TensorArray.closeOp(handle, name)
    }
  }

  /** Colocates ops created by `block` with an internal colocation group, if such a group exists, or with `op`. If no
    * internal colocation group is set, this method colocates ops with `op` and sets the internal colocation group to be
    * `op`. */
  private[this] def maybeColocateWith[R](op: Op)(block: => R): R = {
    if (!colocateWithFirstWrite) {
      block
    } else if (colocationOps == null) {
      colocationOps = Seq(op)
      Op.colocateWith(colocationOps.toSet, ignoreExisting = true)(block)
    } else {
      Op.colocateWith(colocationOps.take(1).toSet, ignoreExisting = true)(block)
    }
  }
}

object TensorArray {
  /** Creates a new tensor array.
    *
    * @param  size                   Size of the tensor array.
    * @param  dataType               Data type of the elements in the tensor array.
    * @param  dynamicSize            Boolean value indicating whether writes to the tensor array are allowed to grow in
    *                                size. By default, this is not allowed.
    * @param  clearAfterRead         Boolean value indicating whether to clear the tensors in the array, after being
    *                                read. This disables multiple read semantics but allows early release of memory.
    *                                Defaults to `true`.
    * @param  tensorArrayName        Name to use for the tensor array. Overrides the name used for the temporary tensor
    *                                array resource. If not provided or if an empty string is provided, then the name of
    *                                the created tensor array op is used, which is guaranteed to be unique.
    * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all
    *                                elements must have the same shape.
    * @param  elementShape           Expected shape of the elements in the tensor array, if known. If this shape is not
    *                                fully defined, then gathering zero-sized tensor array elements will cause an error.
    * @param  colocateWithFirstWrite Boolean value indicating whether to place the tensor array on the same device as
    *                                the tensor used on its first write call (write operations include `write`,
    *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
    *                                determined by the op creation context available during its initialization.
    * @param  name                   Name for the created tensor array ops.
    * @return Created tensor array.
    */
  def create(
      size: Output,
      dataType: DataType[_],
      dynamicSize: Boolean = false,
      clearAfterRead: Boolean = true,
      tensorArrayName: String = "",
      inferShape: Boolean = true,
      elementShape: Shape = Shape.unknown(),
      colocateWithFirstWrite: Boolean = true,
      name: String = "TensorArray"
  ): TensorArray = {
    // We construct the tensor array with an empty device. The first write into the tensor array from a tensor with a
    // set device will retroactively set the device value of this op.
    val (handle, flow) = {
      if (colocateWithFirstWrite) {
        Op.createWith(device = null) {
          Op.colocateWith(Set.empty[Op], ignoreExisting = true) {
            TensorArray.createOp(
              size, dataType, elementShape, dynamicSize, clearAfterRead, inferShape, tensorArrayName, name)
          }
        }
      } else {
        TensorArray.createOp(
          size, dataType, elementShape, dynamicSize, clearAfterRead, inferShape, tensorArrayName, name)
      }
    }
    createFromHandle(handle, flow, dataType, inferShape, elementShape, colocateWithFirstWrite)
  }

  /** Creates a tensor array from an existing tensor array handle.
    *
    * @param  handle                 Tensor handle to the tensor array.
    * @param  flow                   Float scalar tensor for the tensor array, used to control gradient flow.
    * @param  dataType               Data type of the elements in the tensor array.
    * @param  inferShape             Boolean value indicating whether shape inference is enabled. If `true`, all
    *                                elements must have the same shape.
    * @param  elementShape           Expected shape of the elements in the tensor array, if known. If this shape is not
    *                                fully defined, then gathering zero-sized tensor array elements will cause an error.
    * @param  colocateWithFirstWrite Boolean value indicating whether to place the tensor array on the same device as
    *                                the tensor used on its first write call (write operations include `write`,
    *                                `unstack`, and `split`). If `false`, the tensor array will be placed on the device
    *                                determined by the op creation context available during its initialization.
    * @return Created tensor array.
    */
  private[api] def createFromHandle(
      handle: Output,
      flow: Output,
      dataType: DataType[_],
      inferShape: Boolean = true,
      elementShape: Shape = Shape.unknown(),
      colocateWithFirstWrite: Boolean = true
  ): TensorArray = {
    // Record the current static shape for the array elements. The element shape is defined either by `elementShape` or
    // the shape of the tensor of the first write. If `inferShape` is `true`, then all writes check for shape equality.
    TensorArray(
      handle = handle,
      flow = flow,
      dataType = dataType,
      inferShape = inferShape,
      elementShape = if (elementShape.rank == -1) None else Some(elementShape),
      colocateWithFirstWrite = colocateWithFirstWrite)
  }

  /** Creates an op that constructs a tensor array with the provided shape.
    *
    * @param  size            Size of the tensor array.
    * @param  dataType        Data type of the elements in the tensor array.
    * @param  elementShape    Expected shape of the elements in the tensor array, if known. If this shape is not fully
    *                         defined, then gathering zero-sized tensor array elements will cause an error.
    * @param  dynamicSize     Boolean value indicating whether writes to the tensor array are allowed to grow in size.
    *                         By default, this is not allowed.
    * @param  clearAfterRead  Boolean value indicating whether to clear the tensors in the array, after being read. This
    *                         disables multiple read semantics but allows early release of memory. Defaults to `true`.
    * @param  inferShape      Boolean value indicating whether shape inference is enabled. If `true`, all elements must
    *                         have the same shape.
    * @param  tensorArrayName Overrides the name used for the temporary tensor array resource. If not provided or if an
    *                         empty string is provided, then the name of the created op is used, which is guaranteed to
    *                         be unique.
    * @param  name            Name for the created op.
    * @return Tuple containing the resource handle to the tensor array and a scalar used to control gradient flow.
    */
  private[TensorArray] def createOp(
      size: Output,
      dataType: DataType[_],
      elementShape: Shape = Shape.unknown(),
      dynamicSize: Boolean = false,
      clearAfterRead: Boolean = true,
      inferShape: Boolean = true,
      tensorArrayName: String = "",
      name: String = "TensorArray"
  ): (Output, Output) = {
    val outputs = Op.Builder(opType = "TensorArrayV3", name = name)
        .addInput(size)
        .setAttribute("dtype", dataType)
        .setAttribute("element_shape", elementShape)
        .setAttribute("dynamic_size", dynamicSize)
        .setAttribute("clear_after_read", clearAfterRead)
        .setAttribute("identical_element_shapes", inferShape)
        .setAttribute("tensor_array_name", tensorArrayName)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that reads an element from the provided tensor array.
    *
    * @param  handle Tensor array handle.
    * @param  index  Position to read from, inside the tensor array.
    * @param  flow   Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name   Name for the created op.
    * @return Tensor in the specified position of the tensor array.
    */
  private[TensorArray] def readOp(
      handle: Output,
      index: Output,
      flow: Output,
      dataType: DataType[_],
      name: String = "TensorArrayRead"
  ): Output = {
    Op.Builder(opType = "TensorArrayReadV3", name = name)
        .addInput(handle)
        .addInput(index)
        .addInput(flow)
        .setAttribute("dtype", dataType)
        .build().outputs(0)
  }

  /** Creates an op that writes an element to the provided tensor array.
    *
    * @param  handle Tensor array handle.
    * @param  index  Position to write to, inside the tensor array.
    * @param  value  Tensor to write to the tensor array.
    * @param  flow   Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name   Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  private[TensorArray] def writeOp(
      handle: Output,
      index: Output,
      value: Output,
      flow: Output,
      name: String = "TensorArrayWrite"
  ): Output = {
    Op.Builder(opType = "TensorArrayWriteV3", name = name)
        .addInput(handle)
        .addInput(index)
        .addInput(value)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that gathers specific elements from the provided tensor array.
    *
    * Note that all elements selected by `indices` must have the same shape.
    *
    * @param  handle   Tensor array handle.
    * @param  indices  Positions in the tensor array from which to read tensor elements.
    * @param  flow     Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  dataType Data type of the tensor that is returned.
    * @param  shape    Expected shape of the elements in the tensor array, if known. If this shape is not fully defined,
    *                  then gathering zero-sized tensor array elements will cause an error.
    * @param  name     Name for the created op.
    * @return Tensor containing the gathered elements, concatenated along a new axis (the new dimension `0`).
    */
  private[TensorArray] def gatherOp(
      handle: Output,
      indices: Output,
      flow: Output,
      dataType: DataType[_],
      shape: Shape = Shape.unknown(),
      name: String = "TensorArrayGather"
  ): Output = {
    Op.Builder(opType = "TensorArrayGatherV3", name = name)
        .addInput(handle)
        .addInput(indices)
        .addInput(flow)
        .setAttribute("dtype", dataType)
        .setAttribute("element_shape", shape)
        .build().outputs(0)
  }

  /** Creates an op that scatters the provided elements along indices of the provided tensor array.
    *
    * Note that `indices` must be a vector and its length must match the first dimension of `value`.
    *
    * @param  handle  Tensor array handle.
    * @param  indices Positions in the tensor array at which to write the tensor elements.
    * @param  value   Concatenated tensor to write to the tensor array.
    * @param  flow    Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name    Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  private[TensorArray] def scatterOp(
      handle: Output,
      indices: Output,
      value: Output,
      flow: Output,
      name: String = "TensorArrayScatter"
  ): Output = {
    Op.Builder(opType = "TensorArrayScatterV3", name = name)
        .addInput(handle)
        .addInput(indices)
        .addInput(value)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that concatenates the elements of the tensor array.
    *
    * The op takes `T` elements with shapes `[n0, d0, d1, ...]`, `[n1, d0, d1, ...]`, ..., `[n(T-1), d0, d1, ...]` and
    * concatenates them into a tensor with shape `[n0 + n1 + ... + n(T-1), d0, d1, ...]`.
    *
    * All elements must have the same shape, except for their first dimension.
    *
    * @param  handle    Tensor array handle.
    * @param  flow      Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  dataType  Data type of the tensor that is returned.
    * @param  shapeTail Expected shape of the elements in the tensor array, except for their first dimension, if known.
    *                   If this shape is not fully defined, then concatenating zero-sized tensor array elements will
    *                   cause an error.
    * @param  name      Name for the created op.
    * @return Tuple containing a tensor with all of the elements in the tensor array, concatenated along the first axis,
    *         and a vector with the row sizes of the original `T` elements in the output tensor. In the example above,
    *         this would be the values `n1, n2, ..., n(T-1)`.
    */
  private[TensorArray] def concatenateOp(
      handle: Output,
      flow: Output,
      dataType: DataType[_],
      shapeTail: Shape = Shape.unknown(),
      name: String = "TensorArrayConcatenate"
  ): (Output, Output) = {
    val outputs = Op.Builder(opType = "TensorArrayConcatV3", name = name)
        .addInput(handle)
        .addInput(flow)
        .setAttribute("dtype", dataType)
        .setAttribute("element_shape_except0", shapeTail)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that splits the data from the input value into tensor array elements.
    *
    * Assuming that `lengths` takes on values `[n0, n1, ..., n(T-1)]` and that `value` has shape
    * `[n0 + n1 + ... + n(T-1), d0, d1, ...]`, this splits `values` into a tensor array with `T` tensors.
    *
    * Tensor array index `t` will be the sub-tensor of values with starting position
    * `[n0 + n1 + ... + n(t-1), 0, 0, ...]` and size `nt * d0 * d1 * ...`.
    *
    * @param  handle  Tensor array handle.
    * @param  value   Concatenated tensor to write to the tensor array.
    * @param  lengths Vector of lengths, specifying how to split the rows of `value` into the tensor array.
    * @param  flow    Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name    Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  private[TensorArray] def splitOp(
      handle: Output,
      value: Output,
      lengths: Output,
      flow: Output,
      name: String = "TensorArraySplit"
  ): Output = {
    Op.Builder(opType = "TensorArraySplitV3", name = name)
        .addInput(handle)
        .addInput(value)
        .addInput(lengths)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that gets the current size of the tensor array.
    *
    * @param  handle Tensor array handle.
    * @param  flow   Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name   Name for the created op.
    * @return Created op output, containing the current size of the tensor array.
    */
  private[TensorArray] def sizeOp(handle: Output, flow: Output, name: String = "TensorArraySize"): Output = {
    Op.Builder(opType = "TensorArraySizeV3", name = name)
        .addInput(handle)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that constructs a tensor array for storing the gradients of values in the provided tensor array
    * handle.
    *
    * If the provided tensor array gradient already exists, then a reference to it is returned.
    *
    * This op locks the size of the original tensor array by disabling its dynamic size flag.
    *
    * ==A Note About the Input `flow`==
    *
    * The handle `flow` forces the execution of the gradient lookup to occur only after certain other operations have
    * occurred. For example, when the forward tensor array is dynamically sized, writes to this tensor array may resize
    * the object. The gradient tensor array is statically sized based on the size of the forward tensor array when this
    * operation executes. Furthermore, the size of the forward tensor array is frozen by this call. As a result, the
    * flow is used to ensure that the call to generate the gradient tensor array only happens after all writes are
    * executed.
    *
    * In the case of dynamically sized tensor arrays, the gradient computation should only be performed on read
    * operations that have themselves been chained via flow to occur only after all writes have executed. That way the
    * final size of the forward tensor array is known when this operation is called.
    *
    * ==A Note About the `source` Attribute==
    *
    * Tensor array gradient calls use an accumulator tensor array object. If multiple gradients are calculated and run
    * in the same session, then the multiple gradient nodes may accidentally flow though the same accumulator tensor
    * array. This double counts and generally breaks the tensor array gradient flow.
    *
    * The solution is to identify which gradient call this particular tensor array gradient is being called from. This
    * is performed by identifying a unique string (e.g. "gradients", "gradients_1", ...) from the input gradient
    * tensor's name. This string is used as a suffix when creating the tensor array gradient object here (the attribute
    * `source`).
    *
    * The attribute `source` is added as a suffix to the forward tensor array's name when performing the
    * creation/lookup, so that each separate gradient calculation gets its own tensor array accumulator.
    *
    * @param  handle Handle to the forward tensor array.
    * @param  flow   Float scalar that enforces proper chaining of operations.
    * @param  source Gradient source string used to decide which gradient tensor array to return.
    * @param  name   Name for the created op.
    * @return Tuple containing the resource handle to the gradient tensor array and a scalar used to control gradient
    *         flow.
    */
  private[TensorArray] def gradientOp(
      handle: Output,
      flow: Output,
      source: String,
      name: String = "TensorArrayGrad"
  ): (Output, Output) = {
    val outputs = Op.Builder(opType = "TensorArrayGradV3", name = name)
        .addInput(handle)
        .addInput(flow)
        .setAttribute("source", source)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that deletes the provided tensor array from its resource container.
    *
    * This enables the user to close and release the resource in the middle of a step/run.
    *
    * @param  handle Tensor array handle.
    * @param  name   Name for the created op.
    * @return Created op.
    */
  private[TensorArray] def closeOp(handle: Output, name: String = "TensorArrayClose"): Op = {
    Op.Builder(opType = "TensorArrayCloseV3", name = name)
        .addInput(handle)
        .build()
  }

  private[ops] object Gradients {
    // TODO: [TENSOR_ARRAYS] These ops may be differentiable and there may be latent bugs here.
    GradientsRegistry.registerNonDifferentiable("TensorArray")
    GradientsRegistry.registerNonDifferentiable("TensorArrayGrad")
    GradientsRegistry.registerNonDifferentiable("TensorArraySize")
    GradientsRegistry.registerNonDifferentiable("TensorArrayClose")

    GradientsRegistry.registerNonDifferentiable("TensorArrayV2")
    GradientsRegistry.registerNonDifferentiable("TensorArrayGradV2")
    GradientsRegistry.registerNonDifferentiable("TensorArraySizeV2")
    GradientsRegistry.registerNonDifferentiable("TensorArrayCloseV2")

    GradientsRegistry.registerNonDifferentiable("TensorArrayV3")
    GradientsRegistry.registerNonDifferentiable("TensorArrayGradV3")
    GradientsRegistry.registerNonDifferentiable("TensorArrayGradWithShape")
    GradientsRegistry.registerNonDifferentiable("TensorArraySizeV3")
    GradientsRegistry.registerNonDifferentiable("TensorArrayCloseV3")

    GradientsRegistry.register("TensorArrayRead", tensorArrayReadGradient)
    GradientsRegistry.register("TensorArrayReadV2", tensorArrayReadGradient)
    GradientsRegistry.register("TensorArrayReadV3", tensorArrayReadGradient)

    GradientsRegistry.register("TensorArrayWrite", tensorArrayWriteGradient)
    GradientsRegistry.register("TensorArrayWriteV2", tensorArrayWriteGradient)
    GradientsRegistry.register("TensorArrayWriteV3", tensorArrayWriteGradient)

    GradientsRegistry.register("TensorArrayGather", tensorArrayGatherGradient)
    GradientsRegistry.register("TensorArrayGatherV2", tensorArrayGatherGradient)
    GradientsRegistry.register("TensorArrayGatherV3", tensorArrayGatherGradient)

    GradientsRegistry.register("TensorArrayScatter", tensorArrayScatterGradient)
    GradientsRegistry.register("TensorArrayScatterV2", tensorArrayScatterGradient)
    GradientsRegistry.register("TensorArrayScatterV3", tensorArrayScatterGradient)

    GradientsRegistry.register("TensorArrayConcat", tensorArrayConcatenateGradient)
    GradientsRegistry.register("TensorArrayConcatV2", tensorArrayConcatenateGradient)
    GradientsRegistry.register("TensorArrayConcatV3", tensorArrayConcatenateGradient)

    GradientsRegistry.register("TensorArraySplit", tensorArraySplitGradient)
    GradientsRegistry.register("TensorArraySplitV2", tensorArraySplitGradient)
    GradientsRegistry.register("TensorArraySplitV3", tensorArraySplitGradient)

    /** Identifies which call to `gradients()` created the provided gradient op or op output.
      *
      * Tensor array gradient calls use an accumulator tensor array object. If multiple gradients are calculated and run
      * in the same session, the multiple gradient nodes may accidentally flow through the same accumulator tensor
      * array. This double counting breaks the tensor array gradient flow.
      *
      * The solution is to identify which gradient call this particular tensor array gradient is being called in, by
      * looking at the input gradient tensor's name, and create or lookup an accumulator gradient tensor array
      * associated with this specific call. This resolves any confusion and ensures different gradients from the same
      * forward graph get their own accumulators.
      *
      * @param  opOrOutputName Name of gradient op or op output.
      * @return Unique label associated with the `gradients()` call that is used to create the gradient tensor array.
      */
    private[this] def getGradientSource(opOrOutputName: String): String = {
      val nameParts = opOrOutputName.split("/")
      val gradPosition = nameParts.lastIndexWhere(_.startsWith("Gradient"))
      if (gradPosition == -1)
        throw InvalidArgumentException(
          s"Expected op/tensor name to start with 'Gradient' (excluding scope), but got instead: $opOrOutputName.")
      nameParts.take(gradPosition + 1).mkString("/")
    }

    private[this] def tensorArrayReadGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // Note that the forward flow dependency in the call to `gradient()` is necessary for the case of dynamically
      // sized tensor arrays. When creating the gradient tensor array, the final size of the forward array must be
      // known. For this we need to wait until it has been created by depending on the input flow of the original op.
      Seq(
        null, null,
        TensorArray.createFromHandle(
          op.inputs(0), op.inputs(2), op.dataTypeAttribute("dtype"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(outputGradients.head.name), op.inputs(2))
            .write(op.inputs(1), outputGradients.head.toOutput).flow)
    }

    private[this] def tensorArrayWriteGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val flow = outputGradients.head.toOutput
      Seq(
        null, null,
        TensorArray.createFromHandle(
          op.inputs(0), flow, op.dataTypeAttribute("T"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(flow.name), flow)
            .read(op.inputs(1)),
        flow)
    }

    private[this] def tensorArrayGatherGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // Note that the forward flow dependency in the call to `gradient()` is necessary for the case of dynamically
      // sized tensor arrays. When creating the gradient tensor array, the final size of the forward array must be
      // known. For this we need to wait until it has been created by depending on the input flow of the original op.
      Seq(
        null, null,
        TensorArray.createFromHandle(
          op.inputs(0), op.inputs(2), op.dataTypeAttribute("dtype"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(outputGradients.head.name), op.inputs(2))
            .scatter(op.inputs(1), outputGradients.head.toOutput).flow)
    }

    private[this] def tensorArrayScatterGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val flow = outputGradients.head.toOutput
      Seq(
        null, null,
        TensorArray.createFromHandle(
          op.inputs(0), flow, op.dataTypeAttribute("T"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(flow.name), flow)
            .gather(op.inputs(1)),
        flow)
    }

    private[this] def tensorArrayConcatenateGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // Note that the forward flow dependency in the call to `gradient()` is necessary for the case of dynamically
      // sized tensor arrays. When creating the gradient tensor array, the final size of the forward array must be
      // known. For this we need to wait until it has been created by depending on the input flow of the original op.
      Seq(
        null,
        TensorArray.createFromHandle(
          op.inputs(0), op.inputs(1), op.dataTypeAttribute("dtype"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(outputGradients.head.toOutput.name), op.inputs(1))
            .split(outputGradients.head.toOutput, op.outputs(1)).flow)
    }

    private[this] def tensorArraySplitGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val flow = outputGradients.head.toOutput
      Seq(
        null,
        TensorArray.createFromHandle(
          op.inputs(0), flow, op.dataTypeAttribute("T"), colocateWithFirstWrite = false)
            .gradient(getGradientSource(flow.name), flow)
            .concatenate(),
        null, flow)
    }
  }
}
