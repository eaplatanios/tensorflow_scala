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
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.{DataType, TF}

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
    * @param  name                Slot variable name.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def zeros[T: TF, R: TF](
      primary: Variable[T],
      dataType: DataType[R],
      name: String,
      colocateWithPrimary: Boolean = true
  ): Variable[R] = {
    create(primary, dataType, ZerosInitializer, name, primary.shape, colocateWithPrimary)
  }

  /** Creates a slot initialized with zeros with the same shape as the primary value.
    *
    * @param  primary             Primary value.
    * @param  dataType            Data type of the slot variable.
    * @param  name                Slot variable name.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def zerosForOutput[T: TF, R: TF](
      primary: Output[T],
      dataType: DataType[R],
      name: String,
      colocateWithPrimary: Boolean = true
  ): Variable[R] = {
    // TODO: [SLOTS]: What if `primary`'s shape is not fully defined?
    if (primary.shape.isFullyDefined) {
      createForOutput(primary, dataType, ZerosInitializer, name, primary.shape, colocateWithPrimary)
    } else {
      createForOutput(primary, dataType, ZerosInitializer, name, null, colocateWithPrimary)
    }
  }

  /** Creates a new slow variable.
    *
    * @param  primary             Primary variable.
    * @param  dataType            Data type of the slot variable.
    * @param  initializer         Initializer for the slot variable.
    * @param  name                Slot variable name.
    * @param  shape               Shape of the slot variable. If `null`, then an attempt will be made to infer its value
    *                             from the provided initializer.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def create[T: TF, R: TF](
      primary: Variable[T],
      dataType: DataType[R],
      initializer: Initializer,
      name: String,
      shape: Shape = null,
      colocateWithPrimary: Boolean = true
  ): Variable[R] = {
    // Scope the slot name in the namespace of the primary variable. Set "primary.op.name + '/' + name" as the default
    // name, so that the scope name of the slot variable user can be shared when reuse is 'true'. Meanwhile, when reuse
    // is 'false' and the same name has been previously used, the scope name will be made unique by appending an integer
    // to it.
    val inferredShape = if (shape != null) shape else primary.shape
    VariableScope.scope(s"${primary.op.name}/$name", isDefaultName = true) {
      if (colocateWithPrimary) {
        Op.colocateWith(Set(primary.op), ignoreExisting = true) {
          createSlotVariable(primary, dataType, inferredShape, initializer, "")
        }
      } else {
        createSlotVariable(primary, dataType, inferredShape, initializer, "")
      }
    }
  }

  /** Creates a new slow variable.
    *
    * @param  primary             Primary value.
    * @param  dataType            Data type of the slot variable.
    * @param  initializer         Initializer for the slot variable.
    * @param  name                Name of the slot variable.
    * @param  shape               Shape of the slot variable. If `null`, then an attempt will be made to infer its value
    *                             from the provided initializer.
    * @param  colocateWithPrimary Boolean value indicating whether to colocate the slot variable with the primary
    *                             variable.
    * @return Created slot variable.
    */
  def createForOutput[T: TF, R: TF](
      primary: Output[T],
      dataType: DataType[R],
      initializer: Initializer,
      name: String,
      shape: Shape = null,
      colocateWithPrimary: Boolean = true
  ): Variable[R] = {
    // Scope the slot name in the namespace of the primary value. Set "primary.op.name + '/' + name" as the default
    // name, so that the scope name of the slot variable user can be shared when reuse is 'true'. Meanwhile, when reuse
    // is 'false' and the same name has been previously used, the scope name will be made unique by appending an integer
    // to it.
    val inferredShape = {
      if (shape != null)
        shape
      else
        primary.shape
    }
    VariableScope.scope(s"${primary.op.name}/$name", isDefaultName = true) {
      if (colocateWithPrimary) {
        Op.colocateWith(Set(primary.op), ignoreExisting = true) {
          createSlotVariable(primary, dataType, inferredShape, initializer, "")
        }
      } else {
        createSlotVariable(primary, dataType, inferredShape, initializer, "")
      }
    }
  }

  /** Helper function for creating slot variables. */
  private def createSlotVariable[T: TF, R: TF](
      primary: Variable[T],
      dataType: DataType[R],
      shape: Shape,
      initializer: Initializer,
      scope: String
  ): Variable[R] = {
    val slot = Variable.getVariable[R](scope, shape, initializer, trainable = false)
    if (primary.partitionInformation != null) {
      // Primary is a partitioned variable, and so we need to also indicate that the slot is also a partitioned
      // variable. Slots have the same partitioning as their primaries. For example, when using the Adam optimizer for a
      // linear model, 'slot.name' could be "linear//weights/Adam:0", while 'primary.op.name' is "linear//weight". We
      // want to get "Adam" as the real slot name, and so we remove "linear//weights/" and ":0".
      val realSlotName = slot.name.substring(primary.op.name.length + 1, slot.name.length - 2)
      slot.partitionInformation = Variable.PartitionInformation(
        fullName = s"${primary.partitionInformation.fullName}/$realSlotName",
        fullShape = primary.partitionInformation.fullShape,
        partitionOffsets = primary.partitionInformation.partitionOffsets,
        partitionShape = primary.partitionInformation.partitionShape)
    }
    slot
  }

  /** Helper function for creating slot variables. */
  private def createSlotVariable[T: TF, R: TF](
      primary: Output[T],
      dataType: DataType[R],
      shape: Shape,
      initializer: Initializer,
      scope: String
  ): Variable[R] = {
    Variable.getVariable[R](scope, shape, initializer, trainable = false)
  }
}
