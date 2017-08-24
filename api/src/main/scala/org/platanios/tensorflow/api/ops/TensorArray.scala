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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.types.DataType

/** Class wrapping dynamic-sized, per-time-step, write-once tensor arrays.
  *
  * This class is meant to be used with dynamic iteration primitives such as `whileLoop` and `mapFunction`. It supports
  * gradient back-propagation via special "flow" control flow dependencies.
  *
  * Note that the name of the `TensorArray` (even if passed in) is uniquified automatically. Each time a new
  * `TensorArray` is created at runtime it is assigned its own name for the duration of the run. This avoids name
  * collisions if a `TensorArray` is created within a `whileLoop`.
  *
  * @param  handle       Tensor handle to the tensor array.
  * @param  flow         Float scalar tensor for the tensor array, used to control gradient flow.
  * @param  dataType     Data type of the tensor array elements.
  * @param  inferShape   Boolean value indicating whether shape inference is enabled. If `true`, all elements must have
  *                      the same shape.
  * @param  elementShape A [[Shape]] object specifying the shape constraints of each of the elements of the tensor
  *                      array. The shape need not be fully defined.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class TensorArray private(
    handle: Output, flow: Output, dataType: DataType, inferShape: Boolean, private var elementShape: Option[Shape]) {
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
      case None =>
        elementShape = Some(shape)
    }
  }

  /** Returns a tensor array with the same content and properties as this one.
    *
    * @return New [[TensorArray]] object with a flow that ensures the control dependencies from the contexts will become
    *         control dependencies for writes, reads, etc. Use this object for all subsequent operations.
    */
  private[api] def identity: TensorArray = {
    TensorArray(this.handle, Basic.identity(this.flow), this.dataType, this.inferShape, this.elementShape)
  }

  /** Creates an op that reads an element from this tensor array.
    *
    * @param  index Position to read from, inside the tensor array.
    * @param  name  Name for the created op.
    * @return Tensor in the specified position of the tensor array.
    */
  private[api] def read(index: Output, name: String = "TensorArrayRead"): Output = {
    Op.createWith(colocationOps = Set(handle.op)) {
      val value = TensorArray.readOp(handle, index, flow, name)
      elementShape.foreach(value.setShape)
      value
    }
  }

  // TODO: !!! Missing the "maybeColocateWith" functionality.

  /** Creates an op that writes an element to this tensor array.
    *
    * @param  index Position to write to, inside the tensor array.
    * @param  value Tensor to write to the tensor array.
    * @param  name  Name for the created op.
    * @return Output flow of the tensor array, used to enforce proper chaining of operations.
    */
  private[api] def write(index: Output, value: Output, name: String = "TensorArrayWrite"): TensorArray = {
    val writeFlow = Op.createWith(colocationOps = Set(handle.op)) {
      TensorArray.writeOp(handle, index, value, flow, name)
    }
    val returnValue = TensorArray(this.handle, writeFlow, this.dataType, this.inferShape, this.elementShape)
    if (this.inferShape)
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
  private[api] def gather(indices: Output, name: String = "TensorArrayGather"): Output = {
    Op.createWith(colocationOps = Set(handle.op)) {
      val value = TensorArray.gatherOp(handle, indices, flow, dataType, elementShape.getOrElse(Shape.unknown()), name)
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
  private[api] def scatter(indices: Output, value: Output, name: String = "TensorArrayScatter"): TensorArray = {
    val scatterFlow = Op.createWith(colocationOps = Set(handle.op)) {
      TensorArray.scatterOp(handle, indices, value, flow, name)
    }
    val returnValue = TensorArray(this.handle, scatterFlow, this.dataType, this.inferShape, this.elementShape)
    if (this.inferShape) {
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
  private[api] def stack(name: String = "TensorArrayStack"): Output = {
    Op.createWithNameScope(name, Set(handle.op)) {
      Op.createWith(colocationOps = Set(handle.op)) {
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
  private[api] def unstack(value: Output, name: String = "TensorArrayUnstack"): TensorArray = {
    Op.createWithNameScope(name, Set(handle.op, value.op)) {
      // TODO: No colocation with the handle op here? This was also missing from the Python API.
      scatter(Math.range(Basic.constant(0), Basic.shape(value)(0)), value, name)
    }
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
  private[api] def concatenate(name: String = "TensorArrayConcatenate"): Output = {
    Op.createWith(colocationOps = Set(handle.op)) {
      val shape = elementShape.map(s => Shape.fromSeq(s.asArray.tail)).getOrElse(Shape.unknown())
      val (value, _) = TensorArray.concatenateOp(handle, flow, dataType, shape, name)
      if (elementShape.isDefined)
        value.setShape(Shape(-1 +: shape.asArray: _*))
      value
    }
  }

  // def split(value: Output, lengths: Output, name: String = "TensorArraySplit"): TensorArray = {
  //   Op.createWithNameScope(name, Set(handle.op, value.op, lengths.op)) {
  //     val castedLengths = MathOps.cast(lengths, DataType.Int64)
  //     val splitFlow = Op.createWith(colocationOps = Set(handle.op)) {
  //       TensorArray.splitOp(handle, value, castedLengths, flow, name)
  //     }
  //     val returnValue = TensorArray(this.handle, splitFlow, this.dataType, this.inferShape, this.elementShape)
  //     if (this.inferShape) {
  //       val valueShape = splitFlow.inputs(1).shape
  //       val lengths = Op.constantValue(splitFlow.inputs(2))
  //       val shape = {
  //         // TODO: [TENSOR] Need Tensor.max() and Tensor.min() methods.
  //         if (valueShape != Shape.unknown() && lengths != null && lengths.max() == lengths.min())
  //           Shape.fromSeq(lengths(0).scalar.asInstanceOf[Int] +: valueShape.asArray.tail)
  //         else
  //           Shape.unknown()
  //       }
  //       returnValue.mergeElementShape(shape)
  //     }
  //     returnValue
  //   }
  // }

  /** Returns an op that gets the current size of the tensor array.
    *
    * @param  name Name for the created op.
    * @return Created op output, containing the current size of the tensor array.
    */
  private[api] def size(name: String = "TensorArraySize"): Output = {
    Op.createWith(colocationOps = Set(handle.op)) {
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
      source: String, flow: Output = this.flow, name: String = "TensorArrayGradient"): TensorArray = {
    // 'TensorArray.gradientOp' requires a flow input when forward tensor arrays are dynamically sized. This forces the
    // creation of the gradient tensor array only once the final forward array's size is fixed.
    Op.createWithNameScope(name, Set(handle.op)) {
      Op.createWith(colocationOps = Set(handle.op)) {
        val (gradientHandle, _) = TensorArray.gradientOp(handle, flow, source)
        val gradientFlow = Op.createWith(controlDependencies = Set(gradientHandle)) {
          Basic.identity(flow, name = "GradientFlow")
        }
        TensorArray(gradientHandle, gradientFlow, this.dataType, this.inferShape, this.elementShape)
      }
    }
  }

  /** Returns an op that deletes this tensor array from its resource container.
    *
    * This enables the user to close and release the resource in the middle of a step/run.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  private[api] def close(name: String = "TensorArrayClose"): Op = {
    Op.createWith(colocationOps = Set(handle.op)) {
      TensorArray.closeOp(handle, name)
    }
  }
}

private[api] object TensorArray {
  /** Creates a new tensor array.
    *
    * @param  size            Size of the tensor array.
    * @param  dataType        Data type of the elements in the tensor array.
    * @param  dynamicSize     Boolean value indicating whether writes to the tensor array are allowed to grow in size.
    *                         By default, this is not allowed.
    * @param  clearAfterRead  Boolean value indicating whether to clear the tensors in the array, after being read. This
    *                         disables multiple read semantics but allows early release of memory. Defaults to `true`.
    * @param  tensorArrayName Name to use for the tensor array. Overrides the name used for the temporary tensor array
    *                         resource. If not provided or if an empty string is provided, then the name of the created
    *                         tensor array op is used, which is guaranteed to be unique.
    * @param  inferShape      Boolean value indicating whether shape inference is enabled. If `true`, all elements must
    *                         have the same shape.
    * @param  elementShape    Expected shape of the elements in the tensor array, if known. If this shape is not fully
    *                         defined, then gathering zero-sized tensor array elements will cause an error.
    * @param  name            Name for the created tensor array ops.
    * @return Created tensor array.
    */
  private[api] def create(
      size: Output, dataType: DataType, dynamicSize: Boolean = false, clearAfterRead: Boolean = true,
      tensorArrayName: String = "", inferShape: Boolean = true, elementShape: Shape = Shape.unknown(),
      name: String = "TensorArray"): TensorArray = {
    // We construct the tensor array with an empty device. The first write into the tensor array from a tensor with a
    // set device will retroactively set the device value of this op.
    val (handle, flow) = Op.createWith(device = null, controlDependencies = Set.empty[Op]) {
      Op.createWithNameScope(nameScope = name, Set(size.op)) {
        TensorArray.createOp(size, dataType, elementShape, dynamicSize, clearAfterRead, tensorArrayName, name)
      }
    }
    createFromHandle(handle, flow, dataType, inferShape, elementShape)
  }

  /** Creates a tensor array from an existing tensor array handle.
    *
    * @param  handle       Tensor handle to the tensor array.
    * @param  flow         Float scalar tensor for the tensor array, used to control gradient flow.
    * @param  dataType     Data type of the elements in the tensor array.
    * @param  inferShape   Boolean value indicating whether shape inference is enabled. If `true`, all elements must
    *                      have the same shape.
    * @param  elementShape Expected shape of the elements in the tensor array, if known. If this shape is not fully
    *                      defined, then gathering zero-sized tensor array elements will cause an error.
    * @return Created tensor array.
    */
  private[api] def createFromHandle(
      handle: Output, flow: Output, dataType: DataType, inferShape: Boolean = true,
      elementShape: Shape = Shape.unknown()): TensorArray = {
    // Record the current static shape for the array elements. The element shape is defined either by 'elementShape' or
    // by the shape of the tensor of the first write. If 'inferShape' is 'true', then all writes check for shape
    // equality.
    TensorArray(
      handle = handle,
      flow = flow,
      dataType = dataType,
      inferShape = inferShape || elementShape == Shape.unknown(),
      elementShape = if (elementShape == Shape.unknown()) None else Some(elementShape))
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
    * @param  tensorArrayName Overrides the name used for the temporary tensor array resource. If not provided or if an
    *                         empty string is provided, then the name of the created op is used, which is guaranteed to
    *                         be unique.
    * @param  name            Name for the created op.
    * @return Tuple containing the resource handle to the tensor array and a scalar used to control gradient flow.
    */
  private[TensorArray] def createOp(
      size: Output, dataType: DataType, elementShape: Shape = Shape.unknown(), dynamicSize: Boolean = false,
      clearAfterRead: Boolean = true, tensorArrayName: String = "",
      name: String = "TensorArray"): (Output, Output) = {
    val outputs = Op.Builder(opType = "TensorArrayV3", name = name)
        .addInput(size)
        .setAttribute("dtype", dataType)
        .setAttribute("element_shape", elementShape)
        .setAttribute("dynamic_size", dynamicSize)
        .setAttribute("clear_after_read", clearAfterRead)
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
      handle: Output, index: Output, flow: Output, name: String = "TensorArrayRead"): Output = {
    Op.Builder(opType = "TensorArrayReadV3", name = name)
        .addInput(handle)
        .addInput(index)
        .addInput(flow)
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
      handle: Output, index: Output, value: Output, flow: Output,
      name: String = "TensorArrayWrite"): Output = {
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
      handle: Output, indices: Output, flow: Output, dataType: DataType, shape: Shape = Shape.unknown(),
      name: String = "TensorArrayGather"): Output = {
    Op.Builder(opType = "TensorArrayGatherV3", name = name)
        .addInput(handle)
        .addInput(indices)
        .addInput(flow)
        .setAttribute("dtype", dataType)
        .setAttribute("shape", shape)
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
      handle: Output, indices: Output, value: Output, flow: Output,
      name: String = "TensorArrayScatter"): Output = {
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
      handle: Output, flow: Output, dataType: DataType, shapeTail: Shape = Shape.unknown(),
      name: String = "TensorArrayConcatenate"): (Output, Output) = {
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
      handle: Output, value: Output, lengths: Output, flow: Output,
      name: String = "TensorArraySplit"): Output = {
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
      handle: Output, flow: Output, source: String, name: String = "TensorArrayGrad"): (Output, Output) = {
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
}
