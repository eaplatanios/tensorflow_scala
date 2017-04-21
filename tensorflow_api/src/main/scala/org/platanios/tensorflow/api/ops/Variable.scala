package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.api.{DataType, Shape}

/**
  * @author Emmanouil Antonios Platanios
  */
object Variable {
  /** Creates an op that holds a handle to a variable resource.
    *
    * Variables hold state in the form of a tensor that persists across steps. The output of this op is a reference to
    * the tensor state so it may be read or modified.
    *
    * @param  shape      Shape of the variable tensor.
    * @param  dataType   Data type of the elements in the variable tensor.
    * @param  container  If non-empty, the created variable is placed in the given container. Otherwise, a default
    *                    container is used.
    * @param  sharedName If non-empty, the created variable is named in the given bucket with this shared name.
    *                    Otherwise, the op name is used, instead.
    * @param  name       Name for the created variable op.
    * @return Created variable op.
    */
  private def variable(
      shape: Shape, dataType: DataType, container: String = "", sharedName: String = "",
      name: String = "Variable"): Op.Output = {
    Op.Builder(opType = "VarHandleOp", name = name)
        .setAttribute(name = "shape", value = shape)
        .setAttribute(name = "dtype", value = dataType)
        .setAttribute(name = "container", value = container)
        .setAttribute(name = "shared_name", value = sharedName)
        .build().outputs(0)
  }

  /** Creates an op that checks whether a resource handle-based variable has been initialized.
    *
    * The output of the op is a boolean scalar indicating whether the tensor has been initialized.
    *
    * @param  variable Variable being checked that may be uninitialized.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  def isVariableInitialized(variable: Op.Output, name: String = "IsVariableInitialized"): Op.Output = {
    Op.Builder(opType = "VarIsInitializedOp", name = name)
        .addInput(variable)
        .build().outputs(0)
  }

  /** Creates an op that reads the current value of a variable resource.
    *
    * The tensor returned by the op is immutable.
    *
    * The value returned by the op is guaranteed to be influenced by all the writes on which this operation depends
    * directly or indirectly, and to not be influenced by any of the writes which depend directly or indirectly on this
    * op.
    *
    * @param  variable Resource variable whose value is being read.
    * @param  dataType Data type of the elements in the variable tensor.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def readVariable(variable: Op.Output, dataType: DataType, name: String = "ReadVariable"): Op.Output = {
    Op.Builder(opType = "ReadVariableOp", name = name)
        .addInput(variable)
        .setAttribute(name = "dtype", value = dataType)
        .build().outputs(0)
  }

  /** Creates an op that reads the current value of a variable resource, without any memory model.
    *
    * The tensor returned by the op aliases a mutable tensor, and its value can be observed to be different by different
    * op.
    *
    * IMPORTANT NOTE: This method is supposed to be internal and private to the TensorFlow implementation.
    *
    * @param  variable Resource variable whose value is being read.
    * @param  dataType Data type of the elements in the variable tensor.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def unsafeReadVariable(
      variable: Op.Output, dataType: DataType, name: String = "UnsafeReadVariable"): Op.Output = {
    Op.Builder(opType = "_UnsafeReadVariable", name = name)
        .addInput(variable)
        .setAttribute(name = "dtype", value = dataType)
        .build().outputs(0)
  }

  /** Creates an op that deletes the resource represented by the provided variable.
    *
    * All subsequent ops using the variable will result in a `NotFound` error status.
    *
    * @param  variable          Variable to be deleted.
    * @param  ignoreLookupError Boolean value indicating whether to ignore the error occurring when the resource does
    *                           not exist.
    * @param  name              Name for the created op.
    * @return Created op.
    */
  private def destroyVariable(
      variable: Op.Output, ignoreLookupError: Boolean = true, name: String = "DestroyVariable"): Op = {
    Op.Builder(opType = "DestroyResourceOp", name = name)
        .addInput(variable)
        .setAttribute(name = "ignore_lookup_error", value = ignoreLookupError)
        .build()
  }

  /** Creates an op that assigns a value to a variable.
    *
    * Any "readVariable" op with a control dependency on this op is guaranteed to return this value or a subsequent
    * newer value of the variable.
    *
    * @param  variable Variable whose value is being assigned and that may be uninitialized.
    * @param  value    Value to be assigned to the variable.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def assign(variable: Op.Output, value: Op.Output, name: String = "AssignVariable"): Op = {
    Op.Builder(opType = "AssignVariableOp", name = name)
        .addInput(variable)
        .addInput(value)
        .setAttribute(name = "dtype", value = value.dataType)
        .build()
  }

  /** Creates an op that updates a variable value by adding the provided value to it.
    *
    * Any "readVariable" op with a control dependency on this op is guaranteed to return this value or a subsequent
    * newer value of the variable.
    *
    * @param  variable Variable whose value is being assigned and that may be uninitialized.
    * @param  value    Value to be added to the variable.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def assignAdd(variable: Op.Output, value: Op.Output, name: String = "AssignAddVariable"): Op = {
    Op.Builder(opType = "AssignAddVariableOp", name = name)
        .addInput(variable)
        .addInput(value)
        .build()
  }

  /** Creates an op that updates a variable value by subtracting the provided value to it.
    *
    * Any "readVariable" op with a control dependency on this op is guaranteed to return this value or a subsequent
    * newer value of the variable.
    *
    * @param  variable Variable whose value is being assigned and that may be uninitialized.
    * @param  value    Value to be subtracted from the variable.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def assignSub(variable: Op.Output, value: Op.Output, name: String = "AssignSubVariable"): Op = {
    Op.Builder(opType = "AssignSubVariableOp", name = name)
        .addInput(variable)
        .addInput(value)
        .build()
  }

  /** Creates an op that gathers slices from the variable pointed to by `variable` according to `indices`.
    *
    * `indices` must be an integer tensor of any dimension (usually 0-D or 1-D). The op produces an output tensor with
    * shape `indices.shape + variable.shape(1::)`, where:
    * {{{
    *   // Scalar indices
    *   output(::, ---) = variable(indices, ---)
    *
    *   // Vector indices
    *   output(i, ---) = variable(indices(i), ---)
    *
    *   // Higher rank indices
    *   output(i, ..., j, ---) = variable(indices(i, ..., j), ---)
    * }}}
    *
    * @param  variable        Variable to slice.
    * @param  indices         Indices tensor, which must be an `Int32` or `Int64` tensor.
    * @param  validateIndices Boolean value indicating whether to validate the provided indices.
    * @param  name            Name for the created op.
    * @return Created op.
    */
  private def gather(
      variable: Op.Output, indices: Op.Output, validateIndices: Boolean = true,
      name: String = "VariableGather"): Op.Output = {
    if (indices.dataType != DataType.Int32 && indices.dataType != DataType.Int64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable gather op indices. " +
            s"Only 'Int32' and 'Int64' are supported.")
    Op.Builder(opType = "ResourceGather", name = name)
        .addInput(variable)
        .addInput(indices)
        .setAttribute("validate_indices", validateIndices)
        .build().outputs(0)
  }

  /** Creates an op that adds sparse updates to `variable`.
    *
    * The operation computes:
    * {{{
    *   // Scalar indices
    *   variable(::, ---) += updates(indices, ---)
    *
    *   // Vector indices
    *   variable(i, ---) += updates(indices(i), ---)
    *
    *   // Higher rank indices
    *   variable(i, ..., j, ---) += updates(indices(i, ..., j), ---)
    * }}}
    *
    * Duplicate entries are handled correctly: if multiple `indices` reference the same location, their contributions
    * add up.
    *
    * The op requires that `updates.shape = indices.shape + variable.shape(1::)`.
    *
    * @param  variable Variable to be updated.
    * @param  indices  Indices tensor, which must be an `Int32` or `Int64` tensor.
    * @param  updates  Updates tensor, which must have a numeric data type.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def scatterAdd(
      variable: Op.Output, indices: Op.Output, updates: Op.Output, name: String = "ScatterAdd"): Op = {
    if (indices.dataType != DataType.Int32 && indices.dataType != DataType.Int64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable scatter add op indices. " +
            s"Only 'Int32' and 'Int64' are supported.")
    Op.Builder(opType = "ResourceScatterAdd", name = name)
        .addInput(variable)
        .addInput(indices)
        .addInput(updates)
        .build()
  }
}
