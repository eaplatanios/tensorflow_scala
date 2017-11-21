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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.Op
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import org.tensorflow.framework.{SaveSliceInfoDef, VariableDef}

import scala.language.postfixOps
import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
case class Variable private (
    dataType: DataType,
    private val variableHandle: Output,
    private val initializeOp: Op,
    private val cachedValue: Output)
    extends ProtoSerializable {
  /** Graph where this variable is defined. */
  val graph: Graph = variableHandle.graph

  /** Name of this variable. */
  val name: String = variableHandle.op.name

  /** Device where this variable resides. */
  val device: String = variableHandle.device

  /** Shape of this variable. */
  val shape: Shape = variableHandle.op.shapeAttribute("shape")

  /** Op corresponding to this variable. */
  val op: Op = variableHandle.op

  /** Op output that holds the variable reference (i.e., handle to the variable).
    *
    * NOTE: You usually do not need to use this field as all ops that need a reference to the variable call it
    * automatically.
    */
  private[api] val handle: Output = variableHandle

  /** Op responsible for initializing this variable. */
  val initializer: Op = initializeOp

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  val isInitialized: Output = Op.createWith(graph)(Variable.isVariableInitialized(handle, name = "IsInitialized"))

  // /** Returns the value of the initialized variable. You should use this instead of the variable itself to initialize
  //   * another variable with a value that depends on the value of this variable.
  //   *
  //   * TODO: Add example.
  //   */
  // val initializedValue: Output = ??? // TODO: [VARIABLES] [CONTROL_FLOW] !!! We need control flow ops for this.

  /** Returns a cached op which reads the last value of this variable.
    *
    * You can not assign a new value to the returned tensor as it is not a reference to the variable.
    *
    * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
    * converting them to tensors.
    */
  val value: Output = {
    if (cachedValue != null) {
      cachedValue
    } else {
      Op.createWith(graph = graph, colocationOps = Set.empty[Op], device = null) {
        // Manually assign reads to the handle's device to avoid log messages
        Op.createWith(device = handle.device)(Variable.readVariable(handle, dataType))
      }
    }
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
  def read(name: String = "Read"): Output = {
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
  def sparseRead(indices: Output, name: String = "Gather"): Output = {
    Op.createWith(graph) {
      val value = Op.createWith(nameScope = name, device = handle.device) {
        Variable.gather(handle, indices, dataType, validateIndices = true, name)
      }
      // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
      // where the variable is.
      Basic.identity(value)
    }
  }

  /** Evaluates the value of this variable.
    *
    * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
    * variable.
    *
    * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
    * the evaluation.
    *
    * @param  feeds   Tensors to feed into the session for this evaluation.
    * @param  session Optional session to use for the evaluation.
    * @return Value of this variable, for this evaluation.
    */
  def evaluate(feeds: Map[Output, Tensor] = Map.empty, session: Session = null): Tensor = {
    toOutput.evaluate(feeds, session)
  }

  // TODO: [VARIABLE] Add support for slice assignment.

  //region Assignment Ops

  // TODO: [TF_UPDATE] The following ops are not atomic. Consider making atomic if there is a way to do so without a
  // performance cost for those who don't need it.

  /** Creates an op that assigns the provided value to this variable and returns its value.
    *
    * @param  value Value to assign the variable to.
    * @param  name  Name for created op.
    * @return Variable value read op, after the assignment.
    */
  def assign(value: Output, name: String = "Assign"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assign(handle, value, name))) {
      read()
    }
  }

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  def assignAdd(value: Output, name: String = "AssignAdd"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assignAdd(handle, value, name))) {
      read()
    }
  }

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  def assignSub(value: Output, name: String = "AssignAdd"): Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.assignSub(handle, value, name))) {
      read()
    }
  }

  /** Creates an op that adds the provided sparse value to the current value of the variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` being added.
    * @param  values  Values to be added, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  def assignScatterAdd(indices: Output, values: Output, name: String = "AssignScatterAdd"): Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.scatterAdd(handle, indices, values, name))) {
      read()
    }
  }

  /** Creates an op that subtracts the provided sparse value from the current value of the variable and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` being subtracted.
    * @param  values  Values to be subtracted, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  def assignScatterSub(indices: Output, values: Output, name: String = "AssignScatterAdd"): Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(graph = graph, controlDependencies = Set[Op](Variable.scatterAdd(handle, indices, -values, name))) {
      read()
    }
  }

  // Useful operator overloads for the assignment methods:

  def update(value: Output): Unit = assign(value)

  def +=(value: Output): Unit = assignAdd(value)
  def -=(value: Output): Unit = assignSub(value)

  //endregion Assignment Ops

  /** Converts this variable to an op output. This function simply returns an op corresponding to the variable value. */
  def toOutput: Output = value

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
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null): Variable = {
    Op.currentVariableScope.getVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, trainable, reuse, collections,
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
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew, collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    Op.currentVariableScope.getPartitionedVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, partitioner, trainable, reuse,
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
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null): Variable = {
    Op.currentVariableScope.getVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, trainable = false, reuse,
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
      name: String, dataType: DataType = null, shape: Shape, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    Op.currentVariableScope.getPartitionedVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, partitioner, trainable = false, reuse,
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
      initializer: Initializer, dataType: DataType = null, shape: Shape = null, trainable: Boolean = true,
      collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null,
      name: String = "Variable"): Variable = {
    val inferredDataType = if (dataType == null) Option(initializer.dataType).getOrElse(FLOAT32) else dataType
    val inferredShape = if (shape == null) initializer.shape else shape
    if (inferredShape == null)
      throw ShapeMismatchException(
        "No shape was provided for the new variable and it could not be inferred from the provided initializer.")
    Op.createWith(nameScope = name, controlDependencies = Set.empty[Op]) {
      val nameScope = Op.currentNameScope
      val trueName = Op.convertNameScopeToName(nameScope)
      val variableHandle = variable(inferredShape, inferredDataType, sharedName = trueName, name = nameScope)
      val initialValue = Op.createWith(nameScope = "Initializer", colocationOps = Set[Op](variableHandle.op)) {
        initializer(inferredDataType, inferredShape, null)
      }
      val initializeOp = assign(variableHandle, initialValue, name = "InitializationAssign")
      val cachedValue = Op.createWith(nameScope = "Read", colocationOps = Set[Op](variableHandle.op)) {
        val cachedValueOp = {
          if (cachingDevice != null) {
            // Manually assign reads to the handle's device to avoid log messages
            val valueOp = Op.createWith(device = variableHandle.device)(readVariable(variableHandle, inferredDataType))
            // Variables may be created in a "createWith(device = ...)" block or a "createWith(colocationOps = ...)"
            // block. At the same time, users would expect the caching device to be independent of this context, and/or
            // would not expect the current device context to be merged with the caching device specification.
            // Therefore, we reset the colocation stack before creating the cached value. Note that resetting the
            // colocation stack will also reset the device stack.
            Op.createWith(colocationOps = Set.empty[Op], device = null)(Basic.identity(valueOp))
          } else {
            null
          }
        }
        cachedValueOp
      }

      val createdVariable = Variable(inferredDataType, variableHandle, initializeOp, cachedValue)
      var effectiveCollections = collections
      if (effectiveCollections.isEmpty)
        effectiveCollections += Graph.Keys.GLOBAL_VARIABLES
      if (trainable)
        effectiveCollections += Graph.Keys.TRAINABLE_VARIABLES
      effectiveCollections.foreach(key => createdVariable.graph.addToCollection(createdVariable, key))
      createdVariable
    }
  }

  /** Creates a variable from the provided ProtoBuf object.
    *
    * @param  variableDef ProtoBuf-serialized variable object.
    * @param  importScope Name scope to use for all imported ops.
    * @param  context     Op creation context to use while creating the variable.
    * @return Constructed [[Variable]] object.
    */
  def fromProto(variableDef: VariableDef, importScope: String = null)
      (implicit context: DynamicVariable[OpCreationContext]): Variable = {
    if (!variableDef.getIsResource)
      throw new IllegalArgumentException("Trying to restore a reference-based variable as a resource-based variable.")

    def prependNameScope(name: String) = if (importScope == null) name else Op.prependNameScope(importScope, name)

    val variableOp = context.graph.getOutputByName(prependNameScope(variableDef.getVariableName))
    val dataType = variableOp.op.dataTypeAttribute("dtype")
    val initializeOp = context.graph.getOpByName(prependNameScope(variableDef.getInitializerName))
    val cachedValueOp = {
      if (variableDef.getSnapshotName == null || variableDef.getSnapshotName == "")
        null
      else
        context.graph.getOutputByName(prependNameScope(variableDef.getSnapshotName))
    }
    val saveSliceInformation = {
      if (variableDef.hasSaveSliceInfoDef)
        PartitionInformation.fromProto(variableDef.getSaveSliceInfoDef)
      else
        null
    }
    val createdVariable = Variable(dataType, variableOp, initializeOp, cachedValueOp)
    createdVariable.partitionInformation = saveSliceInformation
    createdVariable
  }

  /** Variable getter type, useful for defining custom variable getters and stacking them. */
  trait VariableGetter {
    def apply(
        name: String, dataType: DataType = FLOAT32, shape: Shape = null, initializer: Initializer = null,
        regularizer: Regularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null): Variable
  }

  /** Class that contains partitioning information for a variable that can also be used to save it as a slice.
    *
    * @param  fullName         Name of the full variable, of which the variable is a partition.
    * @param  fullShape        Shape of the full variable, of which the variable is a partition.
    * @param  partitionOffsets Offsets of the partition into the full variable.
    * @param  partitionShape   Shape of the variable.
    */
  private[variables] case class PartitionInformation(
      fullName: String = null, fullShape: Shape = null, partitionOffsets: Array[Int] = null,
      partitionShape: Array[Int] = null) extends ProtoSerializable {
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
      variables: Set[Variable] = globalVariables ++ localVariables, name: String = "UninitializedVariables"): Output = {
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
      shape: Shape, dataType: DataType, container: String = "", sharedName: String = "",
      name: String = "Variable"): Output = {
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
      variable: Output, dataType: DataType, name: String = "UnsafeReadVariable"): Output = {
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
      variable: Output, ignoreLookupError: Boolean = true, name: String = "DestroyVariable"): Op = {
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
      variable: Output, indices: Output, dataType: DataType = null, validateIndices: Boolean = true,
      name: String = "VariableGather"): Output = {
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
      val valuesShape = Basic.concatenate(Array(size, parametersShape(1 ::)), 0)
      val values = Basic.reshape(outputGradients.head.toOutput, valuesShape)
      val reshapedIndices = Basic.reshape(indices, size)
      Seq(OutputIndexedSlices(indices = reshapedIndices, values = values, denseShape = parametersShape), null)
    }
  }
}
