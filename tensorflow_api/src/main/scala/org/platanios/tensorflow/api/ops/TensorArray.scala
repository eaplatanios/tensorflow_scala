package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
object TensorArray {
  /** Creates an op that constructs a tensor array with the provided shape.
    *
    * @param  size            Size of the tensor array.
    * @param  dataType        Data type of the elements in the tensor array.
    * @param  shape           Expected shape of the elements in the tensor array, if known. If this shape is not fully
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
  private[this] def create(
      size: Op.Output, dataType: DataType, shape: Shape = Shape.unknown(), dynamicSize: Boolean = false,
      clearAfterRead: Boolean = true, tensorArrayName: String = "",
      name: String = "TensorArray"): (Op.Output, Op.Output) = {
    val outputs = Op.Builder(opType = "TensorArrayV3", name = name)
        .addInput(size)
        .setAttribute("dtype", dataType)
        .setAttribute("element_shape", shape)
        .setAttribute("dynamic_size", dynamicSize)
        .setAttribute("clear_after_read", clearAfterRead)
        .setAttribute("tensor_array_name", tensorArrayName)
        .build().outputs
    (outputs(0), outputs(1))
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
  private[this] def gradient(
      handle: Op.Output, flow: Op.Output, source: String, name: String = "TensorArrayGrad"): (Op.Output, Op.Output) = {
    val outputs = Op.Builder(opType = "TensorArrayGradV3", name = name)
        .addInput(handle)
        .addInput(flow)
        .setAttribute("source", source)
        .build().outputs
    (outputs(0), outputs(1))
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
  private[this] def write(
      handle: Op.Output, index: Op.Output, value: Op.Output, flow: Op.Output,
      name: String = "TensorArrayWrite"): Op.Output = {
    Op.Builder(opType = "TensorArrayWriteV3", name = name)
        .addInput(handle)
        .addInput(index)
        .addInput(value)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that reads an element from the provided tensor array.
    *
    * @param  handle Tensor array handle.
    * @param  index  Position to read from, inside the tensor array.
    * @param  flow   Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name   Name for the created op.
    * @return Tensor in the specified position of the tensor array.
    */
  private[this] def read(
      handle: Op.Output, index: Op.Output, flow: Op.Output, name: String = "TensorArrayRead"): Op.Output = {
    Op.Builder(opType = "TensorArrayReadV3", name = name)
        .addInput(handle)
        .addInput(index)
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
  private[this] def gather(
      handle: Op.Output, indices: Op.Output, flow: Op.Output, dataType: DataType, shape: Shape = Shape.unknown(),
      name: String = "TensorArrayGather"): Op.Output = {
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
  private[this] def scatter(
      handle: Op.Output, indices: Op.Output, value: Op.Output, flow: Op.Output,
      name: String = "TensorArrayScatter"): Op.Output = {
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
  private[this] def concatenate(
      handle: Op.Output, flow: Op.Output, dataType: DataType, shapeTail: Shape = Shape.unknown(),
      name: String = "TensorArrayConcatenate"): (Op.Output, Op.Output) = {
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
  private[this] def split(
      handle: Op.Output, value: Op.Output, lengths: Op.Output, flow: Op.Output,
      name: String = "TensorArraySplit"): Op.Output = {
    Op.Builder(opType = "TensorArraySplitV3", name = name)
        .addInput(handle)
        .addInput(value)
        .addInput(lengths)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that gets the current size of the tensor array.
    *
    * @param  handle  Tensor array handle.
    * @param  flow    Input flow of the tensor array, used to enforce proper chaining of operations.
    * @param  name    Name for the created op.
    * @return Created op output, containing the current size of the tensor array.
    */
  private[this] def size(handle: Op.Output, flow: Op.Output, name: String = "TensorArraySize"): Op.Output = {
    Op.Builder(opType = "TensorArraySizeV3", name = name)
        .addInput(handle)
        .addInput(flow)
        .build().outputs(0)
  }

  /** Creates an op that deletes the provided tensor array from its resource container.
    *
    * This enables the user to close and release the resource in the middle of a step/run.
    *
    * @param  handle Tensor array handle.
    * @param  name   Name for the created op.
    * @return Created op.
    */
  private[this] def close(handle: Op.Output, name: String = "TensorArrayClose"): Op = {
    Op.Builder(opType = "TensorArrayCloseV3", name = name)
        .addInput(handle)
        .build()
  }
}
