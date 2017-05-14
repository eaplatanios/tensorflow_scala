package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.DataType

/** Contains helper functions for creating slots.
  *
  * A slot is a variable created with the same shape as a primary variable. It is always scoped in the name scope of the
  * primary object and typically has the same device and data type.
  *
  * Slots are typically used as accumulators to track values associated with the primary object (e.g., for optimizers or
  * moving averages).
  *
  * @author Emmanouil Antonios Platanios
  */
object Slot {
  /** Creates a slot initialized with zeros with the same shape as the primary variable.
    *
    * @param  primary             Primary variable.
    * @param  dataType            Data type of the slot variable.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def zeros(
      primary: Variable, name: String, dataType: DataType = null, colocateWithPrimary: Boolean = true): Variable = {
    val inferredDataType = if (dataType == null) primary.dataType else dataType
    // TODO: [VARIABLES] What if the shape is not fully defined?
    if (primary.shape.isFullyDefined) {
      create(primary, ZerosInitializer, name, inferredDataType, primary.shape, colocateWithPrimary)
    } else {
      // TODO: [VARIABLES] Maybe this should use 'primary.initializedValue' instead.
      val initialValue = Basic.zerosLike(primary.value, dataType)
      create(primary, DynamicConstantInitializer(initialValue), name, inferredDataType, null, colocateWithPrimary)
    }
  }

  /** Creates a new slow variable.
    *
    * @param  primary             Primary variable.
    * @param  initializer         Initializer for the slot variable.
    * @param  name                Name of the slot variable.
    * @param  dataType            Data type of the slot variable.
    * @param  shape               Shape of the slot variable. If `null`, then an attempt will be made to infer its value
    *                             from the provided initializer.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def create(
      primary: Variable, initializer: Initializer, name: String, dataType: DataType, shape: Shape = null,
      colocateWithPrimary: Boolean = true): Variable = {
    // Scope the slot name in the namespace of the primary variable. Set "primary.op.name + '/' + name" as the default
    // name, so that the scope name of the slot variable user can be shared when reuse is 'true'. Meanwhile, when reuse
    // is 'false' and the same name has been previously used, the scope name will be made unique by appending an integer
    // to it.
    val inferredShape = {
      if (shape != null)
        shape
      else if (primary.shape.isFullyDefined)
        primary.shape
      else
        initializer.shape
    }
    VariableScope.createWithVariableScope(s"${primary.op.name}/$name", isDefaultName = true) {
      if (colocateWithPrimary)
        Op.colocateWith(Set[Op](primary.op))(createSlotVariable(primary, initializer, "", inferredShape, dataType))
      else
        createSlotVariable(primary, initializer, "", inferredShape, dataType)
    }
  }

  /** Helper function for creating slot variables. */
  private[this] def createSlotVariable(
      primary: Variable, initializer: Initializer, scope: String, shape: Shape,
      dataType: DataType): Variable = {
    // TODO: [VARIABLES] When variables and partitioned variables are merged, makes sure this returns a normal variable.
    // TODO: [VARIABLES] When we support more variable types, match the returned variable type to the primary one.
    val slot = Variable.getVariable(scope, shape, dataType, initializer, trainable = false)
    if (primary.saveSliceInformation != null) {
      // Primary is a partitioned variable, and so we need to also indicate that the slot is also a partitioned
      // variable. Slots have the same partitioning as their primaries. For example, when using the Adam optimizer for a
      // linear model, 'slot.name' could be "linear//weights/Adam:0", while 'primary.op.name' is "linear//weight". We
      // want to get "Adam" as the real slot name, and so we remove "linear//weights/" and ":0".
      val realSlotName = slot.name.substring(primary.op.name.length + 1, slot.name.length - 2)
      slot.saveSliceInformation = Variable.SaveSliceInformation(
        fullName = s"${primary.saveSliceInformation.fullName}/$realSlotName",
        fullShape = primary.saveSliceInformation.fullShape,
        variableOffset = primary.saveSliceInformation.variableOffset,
        variableShape = primary.saveSliceInformation.variableShape)
    }
    slot
  }
}
