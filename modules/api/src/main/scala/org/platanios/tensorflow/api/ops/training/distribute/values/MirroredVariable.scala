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

package org.platanios.tensorflow.api.ops.training.distribute.values

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.distribute
import org.platanios.tensorflow.api.ops.training.distribute.strategies.DistributionContext
import org.platanios.tensorflow.api.ops.variables.Saveable.VariableSaveable
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}

// TODO: [CHECKPOINTABLE].

/** Holds a map from devices to synchronized variables.
  *
  * '''NOTE:''' We use `MirroredVariable.getUpdateDevice` for the assignment methods to enforce that we are in an
  * `update` scope. The arguments to `update()` are automatically unwrapped and so the `update()` function would
  * normally see regular variables and not mirrored variables. However, the `update()` function can still operate on
  * wrapped mirrored variables through object members, captured arguments, etc. This is more likely in an
  * `updateNonSlot()` scope, which can update several non-slot variables in one call.
  *
  * @param  index           Index map from devices to variables.
  * @param  primaryVariable Primary variable.
  *
  * @author Emmanouil Antonios Platanios
  */
class MirroredVariable protected (
    override val primaryVariable: Variable,
    override val index: Map[DeviceSpecification, Variable]
) extends MirroredValue[Variable](index)
    with DistributedVariable {
  /** Creates an op that assigns the provided value to this variable and returns its value.
    *
    * @param  value Value to assign the variable to.
    * @param  name  Name for created op.
    * @return Variable value read op, after the assignment.
    */
  @throws[UnsupportedOperationException]
  def assign(value: Output, name: String = "Assign")(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assign(value, name)
  }

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  def assignAdd(value: Output, name: String = "AssignAdd")(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assignAdd(value, name)
  }

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  def assignSub(value: Output, name: String = "AssignAdd")(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assignSub(value, name)
  }

  /** Creates an op that applies updates the provided sparse value updates to this variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` used for the update.
    * @param  values  Values to use for updating, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  def assignScatter(
      indices: Output,
      values: Output,
      name: String = "AssignScatter"
  )(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assignScatter(indices, values, name)
  }

  /** Creates an op that adds the provided sparse value to the current value of the variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` being added.
    * @param  values  Values to be added, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  def assignScatterAdd(
      indices: Output,
      values: Output,
      name: String = "AssignScatterAdd"
  )(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assignScatterAdd(indices, values, name)
  }

  /** Creates an op that subtracts the provided sparse value from the current value of the variable and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` being subtracted.
    * @param  values  Values to be subtracted, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  def assignScatterSub(
      indices: Output,
      values: Output,
      name: String = "AssignScatterSub"
  )(implicit context: DistributionContext): Output = {
    get(MirroredVariable.getUpdateDevice).assignScatterSub(indices, values, name)
  }
}

object MirroredVariable {
  /** Validates that we are in an `update` or an `updateNonSlot` scope and returns the current device.
    *
    * This is used in the mirrored variable assignment methods, to make sure that they are only called via an update
    * method (to make sure all components of the variable are being updated in a consistent way).
    *
    * @return The current update device.
    * @throws IllegalStateException If not in an `update` or an `updateNonSlot` scope.
    */
  @throws[IllegalStateException]
  private[MirroredVariable] def getUpdateDevice: String = {
    distribute.currentUpdateDevice match {
      case Some(device) => device
      case None => throw new IllegalStateException("Use 'DistributionStrategy.update()` to modify a mirrored variable.")
    }
  }

  /** Wrapper saveable object that allows mirrored variables to be saved. */
  implicit class MirroredVariableSaveable(variable: MirroredVariable)
      extends VariableSaveable(variable.primaryVariable) {
    override private[api] def restore(restoredTensors: Seq[Output], restoredShapes: Seq[Output] = null): Op = {
      val restoredTensor = {
        if (restoredShapes != null)
          Basic.reshape(restoredTensors.head, restoredShapes.head)
        else
          restoredTensors.head
      }
      // Restore the same value into all variables.
      ControlFlow.group(variable.index.map(p => {
        // Copy the restored tensor to the variable's device.
        Variable.assign(p._2.handle, Op.createWith(device = p._1.toString) {
          Basic.identity(restoredTensor)
        })
      }).toSet)
    }
  }
}
