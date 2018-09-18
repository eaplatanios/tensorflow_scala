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

import org.platanios.tensorflow.api.core.{DeviceSpecification, Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.distribute.strategies.{CrossTowerContext, DistributionContext}
import org.platanios.tensorflow.api.ops.variables.{Variable, VariableLike}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.tensorflow.framework.VariableDef

/** Holds a map from devices to variables.
  *
  * @author Emmanouil Antonios Platanios
  */
trait DistributedVariable extends ProtoSerializable {
  /** Primary variable. */
  val primaryVariable: Variable

  /** Index map from devices to variables. */
  val index: Map[DeviceSpecification, Variable]

  /** Type of this distributed variable (e.g., per-device or mirrored). */
  val distributionType: DistributedValue.Type

  /** Returns the value on the specified device (defaults to the current device, if not provided. */
  def get(device: String = "current")(implicit context: DistributionContext): Variable

  protected val sharedName: String = primaryVariable.name.split(':').head

  /** Graph where this variable is defined. */
  val graph: Graph = primaryVariable.graph

  /** Name of this variable. */
  val name: String = primaryVariable.name

  /** Data type of this variable. */
  val dataType: DataType[_] = primaryVariable.dataType

  /** Shape of this variable. */
  val shape: Shape = primaryVariable.shape

  /** Returns a cached op which reads the last value of this partitioned variable.
    *
    * You can not assign a new value to the returned tensor as it is not a reference to the variable.
    *
    * The returned op output will not inherit the control dependencies from the scope where the value is used, which is
    * equivalent behavior to that of getting the value of a variable.
    *
    * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
    * converting them to tensors.
    */
  def value(implicit context: DistributionContext): Output = get().value

  /** Returns the initializer for this variable. */
  val initializer: Op = {
    ControlFlow.group(index.values.map(_.initializer).toSet, "Initializer")
  }

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  val isInitialized: Output = {
    Math.all(Basic.stack(index.values.map(_.isInitialized).toSeq, name = "IsInitialized"))
  }

  /** Value of the initialized variable. You should use this instead of the variable itself to initialize
    * another variable with a value that depends on the value of this variable.
    *
    * Example:
    * {{{
    *   // Initialize `v` with random values, and then use `initializedValue` to guarantee that `v` has been initialized
    *   // before its value is used to initialize `w`. The random tensor will only be sampled once.
    *   val v = tf.variable("v", FLOAT32, Shape(10, 40), tf.RandomTruncatedNormalInitializer())
    *   val w = tf.variable("w", initializer = tf.ConstantInitializer(v.initializedValue * 2.0))
    * }}}
    */
  def initializedValue(implicit context: DistributionContext): Output = Op.initialization {
    ControlFlow.cond(
      isInitialized,
      () => value,
      () => Op.createWith(controlDependencies = Set(initializer))(value))
  }

  /** Creates an op that reads the value of this variable.
    *
    * This method should be used when there are multiple reads, or when it is desirable to read the value only after
    * some condition is true.
    *
    * The returned value may be different from that of [[value]] depending on the device being used, the control
    * dependencies, etc.
    *
    * @return Created op.
    */
  def read(name: String = "Read")(implicit context: DistributionContext): Output = {
    Op.createWith(graph) {
      // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
      // where the variable is.
      Basic.identity(value)
    }
  }

  /** Creates an op that reads the value of this variable sparsely, using the provided `indices`.
    *
    * This method should be used when there are multiple reads, or when it is desirable to read the value only after
    * some condition is true.
    *
    * @param  indices Indices to use for the sparse read.
    * @param  name    Name for the created op.
    * @return Created op.
    */
  @throws[UnsupportedOperationException]
  def gather(indices: Output, name: String = "Gather")(implicit context: DistributionContext): Output = {
    get().gather(indices, name)
  }

  /** Returns the op of this variable. */
  def op(implicit context: DistributionContext): Op = {
    context match {
      // TODO: [DISTRIBUTE] We want cross-tower code that does some variable.op.X (X = name, graph, or dataType) calls
      // to work (even if the current device is not in this.devices), but other uses of variable.op in a cross-tower
      // context to fail.
      case _: CrossTowerContext => primaryVariable.op
      case _ => get().op
    }
  }

  override def toProto: VariableDef = toProto(null)

  /** Alias for `toVariableDef`. */
  def toProto(exportScope: String): VariableDef = toVariableDef(exportScope)

  /** Convert this object to its corresponding ProtoBuf object.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return ProtoBuf object corresponding to this object.
    */
  def toVariableDef(exportScope: String): VariableDef = primaryVariable.toVariableDef(exportScope)
}
