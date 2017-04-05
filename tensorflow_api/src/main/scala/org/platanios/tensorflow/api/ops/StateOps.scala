package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.{DataType, Shape}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
object StateOps {
  /** Creates an op that holds state in the form of a tensor that persists across steps. The output of this op is a
    * reference to the tensor state so it may be read or modified.
    *
    * @param  shape      Shape of the variable tensor.
    * @param  dataType   Data type of the elements in the variable tensor.
    * @param  container  If non-empty, the created variable is placed in the given container. Otherwise, a default
    *                    container is used.
    * @param  sharedName If non-empty, the created variable is named in the given bucket with this shared name.
    *                    Otherwise, the op name is used, instead.
    * @param  name       Name for the generated variable op.
    * @param  context    Op creation context.
    * @return Created variable op.
    */
  def variable(
      shape: Shape, dataType: DataType[_], container: String = "", sharedName: String = "", name: String = "Variable")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    Op.Builder(context = context, opType = "VariableV2", name = name)
        .setAttribute(name = "shape", value = shape)
        .setAttribute(name = "dtype", value = dataType)
        .setAttribute(name = "container", value = container)
        .setAttribute(name = "shared_name", value = sharedName)
        .build().output(index = 0)
  }

  /** Creates an op that checks whether a tensor has been initialized. The output of this op is a boolean scalar
    * indicating whether the tensor has been initialized.
    *
    * @param  variable Variable being checked that may be uninitialized.
    * @param  dataType Data type of the elements in the variable tensor.
    * @param  name     Name for the generated variable op.
    * @param  context  Op creation context.
    * @return Created op.
    */
  def isVariableInitialized(
      variable: Op.Output, dataType: DataType[_], name: String = "IsVariableInitialized")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    Op.Builder(context = context, opType = "IsVariableInitialized", name = name)
        .addInput(variable)
        .setAttribute(name = "dtype", value = dataType)
        .build().output(index = 0)
  }

  /** Creates an op that assigns a value to a variable. The output of this op is the input variable, after the
    * assignment is performed. This makes it easier to chain operations that need to use the assigned value.
    *
    * @param  variable      Variable whose value is being assigned and that may be uninitialized.
    * @param  value         Value to be assigned to the variable.
    * @param  validateShape If `true`, the op will validate that the shape of `value` matches the shape of the tensor
    *                       being referenced by `variable`.
    * @param  useLocking    If `true`, the assignment will be protected by a lock. Otherwise, the behavior is undefined,
    *                       but may exhibit less contention.
    * @param  name          Name for the generated variable op.
    * @param  context       Op creation context.
    * @return Created op.
    */
  def assign(
      variable: Op.Output, value: Op.Output, validateShape: Boolean = true, useLocking: Boolean = true,
      name: String = "Assign")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    Op.Builder(context = context, opType = "Assign", name = name)
        .addInput(variable)
        .addInput(value)
        .setAttribute(name = "validate_shape", value = validateShape)
        .setAttribute(name = "use_locking", value = useLocking)
        .build().output(0)
  }

  /** Creates an op that updates a variable value by adding the provided value to it. The output of this op is the input
    * variable, after the assignment is performed. This makes it easier to chain operations that need to use the
    * assigned value.
    *
    * @param  variable   Variable whose value is being assigned and that may be uninitialized.
    * @param  value      Value to be added to the variable.
    * @param  useLocking If `true`, the assignment will be protected by a lock. Otherwise, the behavior is undefined,
    *                    but may exhibit less contention.
    * @param  name       Name for the generated variable op.
    * @param  context    Op creation context.
    * @return Created op.
    */
  def assignAdd(variable: Op.Output, value: Op.Output, useLocking: Boolean = true, name: String = "AssignAdd")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    Op.Builder(context = context, opType = "AssignAdd", name = name)
        .addInput(variable)
        .addInput(value)
        .setAttribute(name = "use_locking", value = useLocking)
        .build().output(0)
  }

  /** Creates an op that updates a variable value by subtracting the provided value from it. The output of this op is
    * the input variable, after the assignment is performed. This makes it easier to chain operations that need to use
    * the assigned value.
    *
    * @param  variable   Variable whose value is being assigned and that may be uninitialized.
    * @param  value      Value to be subtracting to the variable.
    * @param  useLocking If `true`, the assignment will be protected by a lock. Otherwise, the behavior is undefined,
    *                    but may exhibit less contention.
    * @param  name       Name for the generated variable op.
    * @param  context    Op creation context.
    * @return Created op.
    */
  def assignSub(variable: Op.Output, value: Op.Output, useLocking: Boolean = true, name: String = "AssignSub")
      (implicit context: DynamicVariable[OpCreationContext]): Op.Output = {
    Op.Builder(context = context, opType = "AssignSub", name = name)
        .addInput(variable)
        .addInput(value)
        .setAttribute(name = "use_locking", value = useLocking)
        .build().output(0)
  }

  // TODO: Add scatter update ops.
}
