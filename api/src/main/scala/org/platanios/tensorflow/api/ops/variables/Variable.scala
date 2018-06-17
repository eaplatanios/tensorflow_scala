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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.Op
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import org.tensorflow.framework.{SaveSliceInfoDef, VariableDef}

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.language.postfixOps

/** Variable based on resource handles.
  *
  * See the [[https://www.tensorflow.org/programmers_guide/variables Variables Guide]] for a high-level overview.
  *
  * A variable allows you to maintain state across subsequent calls to `Session.run()`. The variable constructors
  * require an initial value for the variable, which can be a tensor of any type and shape. The initial value defines
  * the type and shape of the variable. After construction, the type and shape of the variable are fixed. The value can
  * be changed using one of the assignment methods.
  *
  * Just like any tensor, variables can be used as inputs for other ops in the graph. Additionally, all the operators
  * overloaded for tensors are carried over to variables, so you can also add nodes to the graph by just doing
  * arithmetic on variables.
  *
  * Unlike the Python API, the Scala API uses resource variables that have well-defined semantics. Each usage of a
  * resource variable in a TensorFlow graph adds a `read` operation to the graph. The tensors returned by a `read`
  * operation are guaranteed to see all modifications to the value of the variable which happen in any operation on
  * which the `read` depends on (either directly, indirectly, or via a control dependency) and guaranteed to not see
  * any modification to the value of the variable from operations that depend on the `read` operation. Updates from
  * operations that have no dependency relationship to the `read` operation might or might not be visible to `read`. For
  * example, if there is more than one assignment to a resource variable in a single `Session.run()` call there is a
  * well-defined value for each operation which uses the variable's value if the assignments and the `read` are
  * connected by edges in the graph.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Variable private (
    override val dataType: DataType,
    private val variableHandle: Output,
    private val initializeOp: Op,
    private val cachedValue: Output,
    private[variables] val graphElement: Output
) extends ProtoSerializable with VariableLike {
  // TODO: _assign_dependencies.

  /** Graph where this variable is defined. */
  override val graph: Graph = variableHandle.graph

  /** Name of this variable. */
  override val name: String = variableHandle.op.name

  /** Device where this variable resides. */
  val device: String = variableHandle.device

  /** Shape of this variable. */
  override val shape: Shape = variableHandle.op.shapeAttribute("shape")

  /** Op corresponding to this variable. */
  val op: Op = variableHandle.op

  /** Op output that holds the variable reference (i.e., handle to the variable).
    *
    * NOTE: You usually do not need to use this field as all ops that need a reference to the variable call it
    * automatically.
    */
  private[api] val handle: Output = variableHandle

  /** Cached op which reads the last value of this variable.
    *
    * You can not assign a new value to the returned tensor as it is not a reference to the variable.
    *
    * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
    * converting them to tensors.
    */
  override val value: Output = {
    if (cachedValue != null) {
      cachedValue
    } else {
      Op.createWith(graph = graph, colocationOps = Set.empty[Op], device = handle.device) {
        // Manually assign reads to the handle's device to avoid log messages
        Op.createWith(device = handle.device)(Variable.readVariable(handle, dataType))
      }
    }
  }

  /** Op responsible for initializing this variable. */
  override val initializer: Op = initializeOp

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  override val isInitialized: Output = {
    Op.createWith(graph)(Variable.isVariableInitialized(handle, name = "IsInitialized"))
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
  override val initializedValue: Output = Op.initialization {
    ControlFlow.cond(isInitialized, () => value, () => Op.createWith(controlDependencies = Set(initializer))(value))
  }

  /** Contains the partition/save-slice information for this variable. */
  private[api] var partitionInformation: Variable.PartitionInformation = _

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
  override def read(name: String = "Read"): Output = {
    Op.createWith(graph) {
      val value = Op.createWith(nameScope = name, device = handle.device) {
        Variable.readVariable(handle, dataType, name)
      }
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
  override def gather(indices: Output, name: String = "Gather"): Output = {
    Op.createWith(graph) {
      val value = Op.createWith(nameScope = name, device = handle.device) {
        Variable.gather(handle, indices, dataType, validateIndices = true, name)
      }
      // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
      // where the variable is.
      Basic.identity(value)
    }
  }

  // /** Evaluates the value of this variable.
  //   *
  //   * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
  //   * variable.
  //   *
  //   * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
  //   * the evaluation.
  //   *
  //   * @param  feeds   Tensors to feed into the session for this evaluation.
  //   * @param  session Optional session to use for the evaluation.
  //   * @return Value of this variable, for this evaluation.
  //   */
  // def evaluate(feeds: Map[Output, Tensor] = Map.empty, session: Session = null): Tensor = {
  //   toOutput.evaluate(feeds, session)
  // }

  //region Assignment Ops

  // TODO: [TF_UPDATE] The following ops are not atomic. Consider making atomic if there is a way to do so without a
  // performance cost for those who don't need it.

  /** Creates an op that assigns the provided value to this variable and returns its value.
    *
    * @param  value Value to assign the variable to.
    * @param  name  Name for created op.
    * @return Variable value read op, after the assignment.
    */
  @throws[UnsupportedOperationException]
  override def assign(value: Output, name: String = "Assign"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assign(handle, value, name))) {
      Variable.readVariable(handle, dataType, name)
    }
  }

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignAdd(value: Output, name: String = "AssignAdd"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assignAdd(handle, value, name))) {
      Variable.readVariable(handle, dataType, name)
    }
  }

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignSub(value: Output, name: String = "AssignAdd"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assignSub(handle, value, name))) {
      Variable.readVariable(handle, dataType, name)
    }
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
  override def assignScatter(indices: Output, values: Output, name: String = "AssignScatter"): Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.scatterUpdate(handle, indices, values, name))) {
      Variable.readVariable(handle, dataType, name)
    }
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
  override def assignScatterAdd(indices: Output, values: Output, name: String = "AssignScatterAdd"): Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.scatterAdd(handle, indices, values, name))) {
      Variable.readVariable(handle, dataType, name)
    }
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
  override def assignScatterSub(indices: Output, values: Output, name: String = "AssignScatterSub"): Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.scatterAdd(handle, indices, -values, name))) {
      Variable.readVariable(handle, dataType, name)
    }
  }

  //endregion Assignment Ops

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
  def toVariableDef(exportScope: String): VariableDef = {
    if (exportScope == null || variableHandle.name.startsWith(exportScope)) {
      val variableDefBuilder = VariableDef.newBuilder()
      variableDefBuilder.setVariableName(Op.stripNameScope(exportScope, handle.name))
      variableDefBuilder.setInitializerName(Op.stripNameScope(exportScope, initializeOp.name))
      if (cachedValue != null)
        variableDefBuilder.setSnapshotName(Op.stripNameScope(exportScope, cachedValue.name))
      variableDefBuilder.setIsResource(true)
      if (partitionInformation != null)
        variableDefBuilder.mergeSaveSliceInfoDef(partitionInformation.toSaveSliceInformationProto(exportScope))
      variableDefBuilder.build()
    } else {
      null
    }
  }

  override def toString: String = op.toString

  override def equals(that: Any): Boolean = that match {
    case that: Variable => this.op == that.op
    case _ => false
  }

  override def hashCode(): Int = op.hashCode()
}

/** Contains helper functions and classes for creating and dealing with [[Variable]] objects. */
private[api] object Variable {
  implicit def variableToOutput(variable: Variable): Output = variable.toOutput

  /** Gets an existing variable with the specified name or creates a new one.
    *
    * This function prefixes the name with the current variable scope and performs variable reuse checks.
    *
    * TODO: Add example.
    *
    * @param  name          Variable name.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
    * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
    *                       shape of the variable from the provided initializer.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    *                       to use by the optimizers.
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
    * @return Requested variable.
    */
  private[api] def getVariable(
      name: String,
      dataType: DataType = null,
      shape: Shape = null,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): Variable = {
    VariableScope.current.getVariable(
      VariableStore.current, name, dataType, shape, initializer, regularizer, trainable, reuse, collections,
      cachingDevice)
  }

  /** Gets an existing partitioned variable with the specified name or creates a new one.
    *
    * This function prefixes the name with the current variable scope and performs variable reuse checks.
    *
    * TODO: Add example.
    *
    * @param  name          Variable name.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
    * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
    *                       shape of the variable from the provided initializer.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  partitioner   Function that accepts a fully defined `Shape` and returns a sequence of integers (i.e., the
    *                       `partitions`). These integers describe how to partition the given variable, along the each
    *                       dimension. That is, `partitions(1) = 3` means that we split the variable into `3` parts
    *                       along dimension `1`. Currently, partitioning along only a single axis is supported.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    *                       to use by the optimizers.
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
    * @return Requested variable.
    */
  private[api] def getPartitionedVariable(
      name: String,
      dataType: DataType = null,
      shape: Shape = null,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      partitioner: Partitioner = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): PartitionedVariable = {
    VariableScope.current.getPartitionedVariable(
      VariableStore.current, name, dataType, shape, initializer, regularizer, partitioner, trainable, reuse,
      collections, cachingDevice)
  }

  /** Gets an existing local variable with the specified name or creates a new one.
    *
    * Local variables are not trainable (i.e., `trainable` argument of [[getVariable]] would be set to `false`) and are
    * added to the graph collection with key [[Graph.Keys.LOCAL_VARIABLES]].
    *
    * This function prefixes the name with the current variable scope and performs variable reuse checks. Please refer
    * to the documentation of [[getVariable]] for more details.
    *
    * @param  name          Variable name.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
    * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
    *                       shape of the variable from the provided initializer.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
    * @return Requested variable.
    */
  private[api] def getLocalVariable(
      name: String,
      dataType: DataType = null,
      shape: Shape = null,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): Variable = {
    VariableScope.current.getVariable(
      VariableStore.current, name, dataType, shape, initializer, regularizer, trainable = false, reuse,
      collections + Graph.Keys.LOCAL_VARIABLES, cachingDevice)
  }

  /** Gets an existing local partitioned variable with the specified name or creates a new one.
    *
    * Local variables are not trainable (i.e., `trainable` argument of [[getVariable]] would be set to `false`) and are
    * added to the graph collection with key [[Graph.Keys.LOCAL_VARIABLES]].
    *
    * This function prefixes the name with the current variable scope and performs variable reuse checks. Please refer
    * to the documentation of [[getPartitionedVariable]] for more details.
    *
    * @param  name          Variable name.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
    * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
    *                       shape of the variable from the provided initializer.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  partitioner   Function that accepts a fully defined `Shape` and returns a sequence of integers (i.e., the
    *                       `partitions`). These integers describe how to partition the given variable, along the each
    *                       dimension. That is, `partitions(1) = 3` means that we split the variable into `3` parts
    *                       along dimension `1`. Currently, partitioning along only a single axis is supported.
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
    * @return Requested variable.
    */
  private[api] def getLocalPartitionedVariable(
      name: String,
      dataType: DataType = null,
      shape: Shape = null,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      partitioner: Partitioner = null,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): PartitionedVariable = {
    VariableScope.current.getPartitionedVariable(
      VariableStore.current, name, dataType, shape, initializer, regularizer, partitioner, trainable = false, reuse,
      collections + Graph.Keys.LOCAL_VARIABLES, cachingDevice)
  }

  /** Creates a variable.
    *
    * @param  initializer   Initializer that creates the tensor that will be used as the initial value of this variable.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value. If it cannot be inferred, then it will default to `FLOAT32`.
    * @param  shape         Shape for the value of the created variable. If `null`, an attempt will be made to infer the
    *                       shape of the variable from the provided initializer.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables to
    *                       use by the optimizers.
    * @param  collections   Set of graph collections keys. The new variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Optional device specification describing where the variable should be cached for reading.
    *                       Defaults to the variable's device. Typical use is to cache on the device where the ops using
    *                       the variable reside, to deduplicate copying through `Switch` and other conditional
    *                       statements.
    * @param  name          Created variable name.
    * @return Created variable.
    */
  private[variables] def apply(
      initializer: Initializer,
      dataType: DataType = null,
      shape: Shape = null,
      trainable: Boolean = true,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null,
      name: String = "Variable"
  ): Variable = {
    val inferredDataType = if (dataType == null) Option(initializer.dataType).getOrElse(FLOAT32) else dataType
    val inferredShape = if (shape == null) initializer.shape else shape
    if (inferredShape == null)
      throw ShapeMismatchException(
        "No shape was provided for the new variable and it could not be inferred from the provided initializer.")
    Op.initialization {
      Op.createWith(nameScope = name) {
        val nameScope = Op.currentNameScope
        val trueName = Op.convertNameScopeToName(nameScope)
        val variableHandle = variable(inferredShape, inferredDataType, sharedName = trueName, name = nameScope)
        val initialValue = Op.createWith(nameScope = "Initializer") {
          Op.colocateWith(Set(variableHandle.op)) {
            initializer(inferredDataType, inferredShape, null)
          }
        }
        val initializeOp = assign(
          variableHandle,
          tryGuardAgainstUninitializedDependencies(name, initialValue), name = "InitializationAssign")
        val (graphElement, cachedValue) = Op.createWith(nameScope = "Read") {
          Op.colocateWith(Set(variableHandle.op)) {
            // Manually assign reads to the handle's device to avoid log messages
            val value = Op.createWith(device = variableHandle.device) {
              readVariable(variableHandle, inferredDataType)
            }
            val cached = {
              if (cachingDevice != null) {
                // Variables may be created in a "createWith(device = ...)" block or a "createWith(colocationOps = ...)"
                // block. At the same time, users would expect the caching device to be independent of this context,
                // and/or would not expect the current device context to be merged with the caching device
                // specification. Therefore, we reset the colocation stack before creating the cached value. Note that
                // resetting the colocation stack will also reset the device stack.
                Op.createWith(colocationOps = Set.empty[Op], deviceFunction = cachingDevice)(Basic.identity(value))
              } else {
                null
              }
            }
            (value, cached)
          }
        }

        val createdVariable = Variable(inferredDataType, variableHandle, initializeOp, cachedValue, graphElement)
        var effectiveCollections = collections
        if (effectiveCollections.isEmpty)
          effectiveCollections += Graph.Keys.GLOBAL_VARIABLES
        if (trainable)
          effectiveCollections += Graph.Keys.TRAINABLE_VARIABLES
        effectiveCollections.foreach(key => createdVariable.graph.addToCollection(createdVariable, key))
        createdVariable
      }
    }
  }

  /** Creates a variable from the provided ProtoBuf object.
    *
    * @param  variableDef ProtoBuf-serialized variable object.
    * @param  importScope Name scope to use for all imported ops.
    * @return Constructed [[Variable]] object.
    */
  def fromProto(variableDef: VariableDef, importScope: String = null): Variable = {
    if (!variableDef.getIsResource)
      throw new IllegalArgumentException("Trying to restore a reference-based variable as a resource-based variable.")

    def prependNameScope(name: String) = if (importScope == null) name else Op.prependNameScope(importScope, name)

    val scope = graphConstructionScope.value
    val handle = scope.graph.getOutputByName(prependNameScope(variableDef.getVariableName))
    val dataType = handle.op.dataTypeAttribute("dtype")
    val initializeOp = scope.graph.getOpByName(prependNameScope(variableDef.getInitializerName))
    val graphElement = try {
      scope.graph.getOutputByName(s"${handle.op.name}/Read/ReadVariable:0")
    } catch {
      // The following handles the default naming of the read ops in the Python API, so that graphs created using the
      // Python API can be loaded in Scala API.
      case _: Throwable => scope.graph.getOutputByName(s"${handle.op.name}/Read/ReadVariableOp:0")
    }
    val cachedValue = {
      if (variableDef.getSnapshotName == null || variableDef.getSnapshotName == "")
        null
      else
        scope.graph.getOutputByName(prependNameScope(variableDef.getSnapshotName))
    }
    val saveSliceInformation = {
      if (variableDef.hasSaveSliceInfoDef)
        PartitionInformation.fromProto(variableDef.getSaveSliceInfoDef)
      else
        null
    }
    val createdVariable = Variable(dataType, handle, initializeOp, cachedValue, graphElement)
    createdVariable.partitionInformation = saveSliceInformation
    createdVariable
  }

  /** Variable getter type, useful for defining custom variable getters and stacking them. */
  trait VariableGetter {
    def apply(
        name: String,
        dataType: DataType = FLOAT32,
        shape: Shape = null,
        initializer: Initializer = null,
        regularizer: Regularizer = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null,
        underlyingGetter: VariableGetter = null
    ): Variable
  }

  /** Adds `getter` to the scope that `block` is executed in.
    *
    * @param  getter Function that specifies custom variable getting behavior. For example, one can specify a custom
    *                variable getter in order to automatically rename the variables, before calling the underlying
    *                getter. The underlying variable getter (i.e., the one which is used by default), is provided as a
    *                last argument to the `getter` function.
    */
  def getter[R](getter: VariableGetter)(block: => R): R = {
    val currentGetters = Op.currentGraph.variableGetters
    currentGetters.withValue(currentGetters.value :+ getter)(block)
  }

  /** Adds `getters` to the scope that `block` is executed in.
    *
    * @param  getters Functions that specify custom variable getting behavior. For example, one can specify a custom
    *                 variable getter in order to automatically rename the variables, before calling the underlying
    *                 getter. The underlying variable getter (i.e., the one which is used by default), is provided as a
    *                 last argument to the `getter` function.
    */
  def getters[R](getters: Seq[VariableGetter])(block: => R): R = {
    val currentGetters = Op.currentGraph.variableGetters
    currentGetters.withValue(currentGetters.value ++ getters)(block)
  }

  /** Returns the variable getters in the current scope. */
  def currentGetters: Seq[VariableGetter] = Op.currentGraph.variableGetters.value

  /** Class that contains partitioning information for a variable that can also be used to save it as a slice.
    *
    * @param  fullName         Name of the full variable, of which the variable is a partition.
    * @param  fullShape        Shape of the full variable, of which the variable is a partition.
    * @param  partitionOffsets Offsets of the partition into the full variable.
    * @param  partitionShape   Shape of the variable.
    */
  private[variables] case class PartitionInformation(
      fullName: String,
      fullShape: Shape,
      partitionOffsets: Array[Int],
      partitionShape: Array[Int]
  ) extends ProtoSerializable {
    if (fullShape.rank != partitionOffsets.length)
      throw new IllegalArgumentException(
        s"The number of offsets provided (${partitionOffsets.length}) does not match the full shape rank (${fullShape.rank}).")
    if (fullShape.asArray.zip(partitionOffsets).exists(p => p._2 < 0 || p._1 <= p._2))
      throw new IllegalArgumentException(
        s"Offset out of bounds exception for offsets '$partitionOffsets' and full shape '$fullShape'.")

    /** Returns the spec string used for saving. */
    def saveSpecString: String = {
      val shapeString = fullShape.asArray.mkString(" ")
      val sliceString = partitionOffsets.zip(partitionShape).map(p => s"${p._1},${p._2}").mkString(":")
      s"$shapeString $sliceString"
    }

    /** Returns the offset when the variable is partitioned along at most one axis.
      *
      * @param  shape Shape of one specific variable partition.
      * @return Integer representing the offset in the dimension along which the variable is partitioned. Returns `0` if
      *         the variable is not being partitioned.
      * @throws ShapeMismatchException If `shape` does not have the same rank as `fullShape`.
      * @throws InvalidShapeException  If the variable is partitioned along more than one axes/dimensions.
      */
    @throws[ShapeMismatchException]
    @throws[InvalidShapeException]
    def singleOffset(shape: Shape): Int = {
      singleSliceAxis(shape).map(partitionOffsets(_)).getOrElse(0)
    }

    /** Returns the slice axis/dimension when the variable is partitioned only along one axis.
      *
      * @param  shape Shape of one specific variable partition.
      * @return Some axis index representing the axis along which the variable is partitioned in, oe `None`, if the
      *         variable does not seem to be partitioned at all.
      * @throws ShapeMismatchException If `shape` does not have the same rank as `fullShape`.
      * @throws InvalidShapeException  If the variable is partitioned along more than one axes/dimensions.
      */
    @throws[ShapeMismatchException]
    @throws[InvalidShapeException]
    def singleSliceAxis(shape: Shape): Option[Int] = {
      if (shape.rank != fullShape.rank)
        throw ShapeMismatchException(
          s"Expected equal rank, but received shape $shape of rank ${shape.rank}, " +
              s"while the full shape $fullShape is of rank ${fullShape.rank}.")
      (0 until shape.rank).foreach(i => {
        if (partitionOffsets(i) + shape(i) > fullShape(i))
          throw InvalidShapeException(
            s"With partition offsets set to [${partitionOffsets.mkString(", ")}], a partition of shape $shape would " +
                s"exceed the full shape $fullShape in dimension $i.")
      })
      val sliceAxes = (0 until shape.rank).filter(i => shape(i) != fullShape(i))
      if (sliceAxes.length > 1)
        throw InvalidShapeException(
          s"Cannot use 'singleSliceAxis()' with shape $shape and full shape $fullShape, since the slice axis could " +
              s"be any one of [${sliceAxes.mkString(", ")}].")
      sliceAxes.headOption
    }

    override def toProto: SaveSliceInfoDef = toSaveSliceInformationProto(null)

    /** Convert this object to a `SaveSliceInfoDef` ProtoBuf object.
      *
      * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope
      *                     will be included in the resulting ProtoBuf object and the export scope will be stripped from
      *                     their names to allow for easy import into new name scopes.
      * @return ProtoBuf object corresponding to this object.
      */
    def toSaveSliceInformationProto(exportScope: String): SaveSliceInfoDef = {
      if (exportScope == null || fullName.startsWith(exportScope)) {
        val saveSliceInfoDefBuilder = SaveSliceInfoDef.newBuilder()
        saveSliceInfoDefBuilder.setFullName(Op.stripNameScope(exportScope, fullName))
        fullShape.asArray.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setFullShape(p._2, p._1.toLong))
        partitionOffsets.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setVarOffset(p._2, p._1.toLong))
        partitionShape.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setVarShape(p._2, p._1.toLong))
        saveSliceInfoDefBuilder.build()
      } else {
        null
      }
    }
  }

  /** Contains helper functions for creating and dealing with [[PartitionInformation]] objects. */
  private[api] object PartitionInformation {
    /** Creates a new [[PartitionInformation]] from the provided ProtoBuf object.
      *
      * @param  saveSliceInfoDef ProtoBuf-serialized variable object.
      * @param  importScope      Name scope to use for all imported ops.
      * @return Constructed [[PartitionInformation]] object.
      */
    def fromProto(saveSliceInfoDef: SaveSliceInfoDef, importScope: String = null): PartitionInformation = {
      val fullName = {
        if (importScope == null)
          saveSliceInfoDef.getFullName
        else
          Op.prependNameScope(importScope, saveSliceInfoDef.getFullName)
      }
      PartitionInformation(
        fullName,
        Shape.fromSeq((0 until saveSliceInfoDef.getFullShapeCount).map(saveSliceInfoDef.getFullShape(_).toInt)),
        (0 until saveSliceInfoDef.getVarOffsetCount).map(saveSliceInfoDef.getVarOffset(_).toInt).toArray,
        (0 until saveSliceInfoDef.getVarShapeCount).map(saveSliceInfoDef.getVarShape(_).toInt).toArray)
    }
  }

  /** Returns the set of global variables in the current graph.
    *
    * Global variables are variables that are shared across machines in a distributed environment. The `Variable()`
    * constructor and the function `getVariable()` automatically add new variables to the graph collection with key
    * `Graph.Keys.GLOBAL_VARIABLES`. This convenience function returns the contents of that collection.
    *
    * An alternative to global variables are local variables.
    */
  def globalVariables: Set[Variable] = Op.currentGraph.globalVariables

  /** Returns the set of local variables in the current graph.
    *
    * Local variables (or per-process variables), are usually not saved/restored to/from checkpoints and are used for
    * temporary or intermediate values. For example, they can be used as counters for metrics computations or number of
    * epochs this machine has read data. This convenience function returns the contents of that collection.
    *
    * An alternative to local variables are global variables.
    */
  def localVariables: Set[Variable] = Op.currentGraph.localVariables

  /** Returns the set of metric variables in the current graph.
    *
    * Metric variables are usually not saved/restored to/from checkpoints and are used for temporary or intermediate
    * values used for computing metrics (e.g., streaming metrics). This convenience function returns the contents of
    * that collection.
    */
  def metricVariables: Set[Variable] = Op.currentGraph.metricVariables

  /** Creates an op that initializes the provided variables.
    *
    * After you launch the graph in a session, you can run the returned op to initialize all the variables in
    * `variables`. This op runs all the initializers of the variables in `variables`, in parallel.
    *
    * Calling `initializer` is equivalent to passing the list of initializers to [[ControlFlow.group]].
    *
    * If `variables` is empty, the method still returns an op that can be run. That op has no effect (i.e., it is a
    * [[ControlFlow.noOp]]).
    *
    * @param  variables Set of variables to initialize.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def initializer(variables: Set[Variable], name: String = "VariablesInitializer"): Op = {
    if (variables != null && variables.nonEmpty)
      ControlFlow.group(variables.map(_.initializer), name)
    else
      ControlFlow.noOp(name)
  }

  /** Creates an op that returns a tensor containing the names of all uninitialized variables in `variables`.
    *
    * If all variables have been initialized, then an empty tensor is returned.
    *
    * @param  variables Variables to check. If not provided, the set of all global and local variables in the current
    *                   graph will be used.
    * @param  name      Name for the created op.
    * @return Created op output, which contains the names of the handles of all variables which have not yet been
    *         initialized.
    */
  def uninitializedVariables(
      variables: Set[Variable] = globalVariables ++ localVariables,
      name: String = "UninitializedVariables"
  ): Output = {
    // Run all operations on the CPU.
    Op.createWith(nameScope = name, device = "/CPU:0") {
      if (variables.isEmpty) {
        // Return an empty tensor so we only need to check for the returned tensor size being 0 as an indication of
        // model readiness.
        Basic.constant(Tensor(STRING))
      } else {
        // Get a 1-D boolean tensor listing whether each variable is initialized.
        val variablesMask = Math.logicalNot(Basic.stack(variables.map(_.isInitialized).toSeq))
        // Get a 1-D string tensor containing all the variable names.
        val variablesList = variables.map(_.handle.name).toSeq
        val variableNames = Basic.constant(Tensor(variablesList.head, variablesList.tail: _*))
        // Return a 1-D tensor containing the names of all uninitialized resources.
        Basic.booleanMask(variableNames, variablesMask)
      }
    }
  }

  /** Attempts to guard against dependencies on uninitialized variables.
    *
    * This method replaces references to variables in `initialValue` with references to the variable's initialized
    * values. The initialized values are essentially conditional TensorFlow graphs that return a variable's value if it
    * is initialized, or its `initialValue` if it hasn't been initialized. This replacement is done on a best effort
    * basis:
    *
    *   - If the `initialValue` graph contains cycles, we do not do any replacements for that graph.
    *   - If the variables that `initialValue` depends on are not present in the global/local variable collections, we
    *     do not replace them.
    *
    * In this cases, it is up to the caller to ensure that the `initialValue` graph uses initialized variables or that
    * they guard access to variables using their `initializedValue` method.
    *
    * @param  variableName Variable name.
    * @param  initialValue Initial value for the variable.
    * @return Initial value to use, with some of its dependencies potentially replaced.
    */
  private[Variable] def tryGuardAgainstUninitializedDependencies(
      variableName: String,
      initialValue: Output
  ): Output = {
    /** Detects cycles in the dependencies of `initialValue`. */
    def hasCycle(op: Op, path: mutable.Set[String]): Boolean = {
      path.contains(op.name) || {
        path.add(op.name)
        op.inputs.exists(i => hasCycle(i.op, path))
      } || {
        val exists = op.controlInputs.exists(i => hasCycle(i.op, path))
        if (!exists)
          path.remove(op.name)
        exists
      }
    }

    // Do not modify the initial value if it contains any cyclic dependencies.
    if (hasCycle(initialValue.op, mutable.Set.empty[String]))
      initialValue
    else
      safeInitialValueFromOutput(variableName, initialValue, mutable.Map.empty[String, Op])
  }

  /** Replaces dependencies on variables with their initialized values.
    *
    * @param  initialValue Initial value to replace.
    * @param  opCache      Map used to memoize the results so as to avoid creating redundant operations.
    * @return Output compatible with `output`. Any inputs that lead to variable values will be replaced with a
    *         corresponding graph that uses the variable's initialized values. This is done on a best-effort basis. If
    *         no modifications need to be made then `output` will be returned unchanged.
    */
  private[Variable] def safeInitialValueFromOutput(
      variableName: String,
      initialValue: Output,
      opCache: mutable.Map[String, Op]
  ): Output = {
    val newOp = opCache.get(initialValue.op.name) match {
      case Some(op) => op
      case None =>
        val op = {
          val opType = initialValue.op.opType
          if (opType == "IsVariableInitialized" || opType == "VarIsInitializedOp" || opType == "ReadVariableOp") {
            initialValue.op
          } else if (opType == "Variable" || opType == "VariableV2" || opType == "VarHandleOp") {
            // TODO: Fix handling of resource variables.
            // Attempt to find the initialized_value of any variable handles.
            findInitializedValueForVariable(initialValue.op) match {
              case Some(initializedValue) => initializedValue.op
              case None => initialValue.op
            }
          } else {
            // Recursively build initializer expressions for the inputs.
            var modified = false
            val newOpInputs = initialValue.op.inputs.map(opInput => {
              val newOpInput = safeInitialValueFromOutput(variableName, opInput, opCache)
              modified ||= newOpInput != opInput
              newOpInput
            })

            // If at least one input was modified, replace the op.
            if (modified) {
              val newOpType = if (opType != "RefSwitch") opType else "Switch"
              val newOpName = s"${initialValue.op.name}_$variableName".replace(":", "_")
              val opBuilder = Op.Builder(newOpType, newOpName)
              newOpInputs.foreach(opBuilder.addInput)
              initialValue.op.toNodeDef.getAttrMap.asScala.foreach(attribute => {
                opBuilder.setAttribute(attribute._1, attribute._2)
              })
              opBuilder.build()
            } else {
              initialValue.op
            }
          }
        }
        opCache.update(initialValue.op.name, op)
        op
    }
    newOp.outputs(initialValue.index)
  }

  /** Finds the initialized value for a variable op, if an initialized value exists.
    *
    * To do so, this method looks up the variable op in the graph global and local variables collections.
    *
    * @param  variableOp Variable op.
    * @return Option containing the initialized value for the variable, or `None`, if no such value could be found.
    */
  private[Variable] def findInitializedValueForVariable(variableOp: Op): Option[Output] = {
    val variables = variableOp.graph.getCollection(Graph.Keys.GLOBAL_VARIABLES) ++
        variableOp.graph.getCollection(Graph.Keys.LOCAL_VARIABLES)
    variables.find(v => v.name == variableOp.name || v.name == variableOp.outputs.head.name).map(_.initializedValue)
  }

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
  private[ops] def variable(
      shape: Shape,
      dataType: DataType,
      container: String = "",
      sharedName: String = "",
      name: String = "Variable"
  ): Output = {
    Op.Builder(opType = "VarHandleOp", name = name)
        .setAttribute("shape", shape)
        .setAttribute("dtype", dataType)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
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
  def isVariableInitialized(variable: Output, name: String = "IsVariableInitialized"): Output = {
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
  private[ops] def readVariable(variable: Output, dataType: DataType, name: String = "ReadVariable"): Output = {
    Op.Builder(opType = "ReadVariableOp", name = name)
        .addInput(variable)
        .setAttribute("dtype", dataType)
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
  private[ops] def unsafeReadVariable(
      variable: Output,
      dataType: DataType,
      name: String = "UnsafeReadVariable"
  ): Output = {
    Op.Builder(opType = "_UnsafeReadVariable", name = name)
        .addInput(variable)
        .setAttribute("dtype", dataType)
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
  private[ops] def destroyVariable(
      variable: Output,
      ignoreLookupError: Boolean = true,
      name: String = "DestroyVariable"
  ): Op = {
    Op.Builder(opType = "DestroyResourceOp", name = name)
        .addInput(variable)
        .setAttribute("ignore_lookup_error", ignoreLookupError)
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
  private[ops] def assign(variable: Output, value: Output, name: String = "AssignVariable"): Op = {
    Op.Builder(opType = "AssignVariableOp", name = name)
        .addInput(variable)
        .addInput(value)
        .setAttribute("dtype", value.dataType)
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
  private[ops] def assignAdd(variable: Output, value: Output, name: String = "AssignAddVariable"): Op = {
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
  private[ops] def assignSub(variable: Output, value: Output, name: String = "AssignSubVariable"): Op = {
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
    * @param  indices         Indices tensor, which must be an `INT32` or `INT64` tensor.
    * @param  dataType        Data type for the created op.
    * @param  validateIndices Boolean value indicating whether to validate the provided indices.
    * @param  name            Name for the created op.
    * @return Created op.
    */
  private[ops] def gather(
      variable: Output,
      indices: Output,
      dataType: DataType = null,
      validateIndices: Boolean = true,
      name: String = "VariableGather"
  ): Output = {
    if (indices.dataType != INT32 && indices.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable gather op indices. " +
            s"Only 'INT32' and 'INT64' are supported.")
    Op.Builder(opType = "ResourceGather", name = name)
        .addInput(variable)
        .addInput(indices)
        .setAttribute("dtype", if (dataType == null) variable.dataType else dataType)
        .setAttribute("validate_indices", validateIndices)
        .build().outputs(0)
  }

  /** Creates an op that applies sparse updates to `variable`.
    *
    * The operation computes:
    * {{{
    *   // Scalar indices
    *   variable(::, ---) = updates(indices, ---)
    *
    *   // Vector indices
    *   variable(i, ---) = updates(indices(i), ---)
    *
    *   // Higher rank indices
    *   variable(i, ..., j, ---) = updates(indices(i, ..., j), ---)
    * }}}
    *
    * Duplicate entries are handled correctly: if multiple `indices` reference the same location, their contributions
    * add up.
    *
    * The op requires that `updates.shape = indices.shape + variable.shape(1::)`.
    *
    * @param  variable Variable to be updated.
    * @param  indices  Indices tensor, which must be an `INT32` or `INT64` tensor.
    * @param  updates  Updates tensor, which must have a numeric data type.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[ops] def scatterUpdate(
      variable: Output,
      indices: Output,
      updates: Output,
      name: String = "ScatterUpdate"
  ): Op = {
    if (indices.dataType != INT32 && indices.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable scatter update op indices. " +
            s"Only 'INT32' and 'INT64' are supported.")
    Op.Builder(opType = "ResourceScatterUpdate", name = name)
        .addInput(variable)
        .addInput(indices)
        .addInput(updates)
        .build()
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
    * @param  indices  Indices tensor, which must be an `INT32` or `INT64` tensor.
    * @param  updates  Updates tensor, which must have a numeric data type.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[ops] def scatterAdd(variable: Output, indices: Output, updates: Output, name: String = "ScatterAdd"): Op = {
    if (indices.dataType != INT32 && indices.dataType != INT64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable scatter add op indices. " +
            s"Only 'INT32' and 'INT64' are supported.")
    Op.Builder(opType = "ResourceScatterAdd", name = name)
        .addInput(variable)
        .addInput(indices)
        .addInput(updates)
        .build()
  }

  private[ops] object Gradients {
    GradientsRegistry.register("ReadVariableOp", readGradient)
    GradientsRegistry.register("ResourceGather", gatherGradient)

    private[this] def readGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(outputGradients.head)
    }

    private[this] def gatherGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // Build appropriately shaped indexed slices.
      // Walk graph back until the original handle is found.
      // TODO: Find a more robust way to get the shape.
      var handle = op.inputs(0)
      while (handle.op.opType != "VarHandleOp")
        handle = handle.op.inputs(0)
      val parametersShape = handle.op.shapeAttribute("shape").toOutput()
      val indices = op.inputs(1)
      val size = Basic.expandDims(Basic.size(indices), 0)
      val valuesShape = Basic.concatenate(Seq(size, parametersShape(1 ::)), 0)
      val values = Basic.reshape(outputGradients.head.toOutput, valuesShape)
      val reshapedIndices = Basic.reshape(indices, size)
      Seq(OutputIndexedSlices(indices = reshapedIndices, values = values, denseShape = parametersShape), null)
    }
  }
}
