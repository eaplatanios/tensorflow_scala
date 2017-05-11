package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}

import org.tensorflow.framework.{SaveSliceInfoDef, VariableDef}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
case class Variable private(dataType: DataType, variableOp: Op.Output, initializeOp: Op, cachedValueOp: Op.Output)
    extends ProtoSerializable {
  /** Graph where this variable is defined. */
  val graph: Graph = initializeOp.graph

  /** Name of this variable. */
  val name: String = variableOp.op.name

  /** Device where this variable resides. */
  val device: String = variableOp.device

  /** Shape of this variable. */
  val shape: Shape = variableOp.op.shapeAttribute("shape")

  /** Op corresponding to this variable. */
  val op: Op = variableOp.op

  /** Op output that holds the variable reference (i.e., handle to the variable).
    *
    * NOTE: You usually do not need to use this field as all ops that need a reference to the variable call it
    * automatically.
    */
  private[api] val handle: Op.Output = variableOp

  // /** Op responsible for creating/initializing this variable. */
  // val create: Op = initializeOp

  /** Op responsible for initializing this variable. */
  val initializer: Op = initializeOp

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  val isInitialized: Op.Output = Variable.isVariableInitialized(variableOp, name = "IsInitialized")

  // /** Returns the value of the initialized variable. You should use this instead of the variable itself to initialize
  //   * another variable with a value that depends on the value of this variable.
  //   *
  //   * TODO: Add example.
  //   */
  // val initializedValue: Op.Output = ??? // TODO: [CONTROL_FLOW] !!! We need control flow ops for this.

  /** Returns a cached op which reads the last value of this variable.
    *
    * You can not assign a new value to the returned tensor as it is not a reference to the variable.
    *
    * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
    * converting them to tensors.
    */
  val value: Op.Output = {
    if (cachedValueOp ne null) {
      cachedValueOp
    } else {
      Op.createWith(nameScope = name, colocationOps = Set.empty[Op], device = null) {
        // Manually assign reads to the handle's device to avoid log messages
        Op.createWith(device = variableOp.device)(Variable.readVariable(variableOp, dataType))
      }
    }
  }

  /** Contains the save slice information for this variable. */
  private[api] var saveSliceInformation: Variable.SaveSliceInformation = _

  /** Sets the slice information used for saving this variable.
    *
    * @param  saveSliceInformation Save slice information to use.
    */
  private[api] def setSaveSliceInformation(saveSliceInformation: Variable.SaveSliceInformation): Unit = {
    this.saveSliceInformation = saveSliceInformation
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
  def read(name: String = "Read"): Op.Output = {
    val value = Op.createWith(nameScope = this.name, device = variableOp.device) { // TODO: Reset colocation ops?
      Variable.readVariable(variableOp, dataType, name)
    }
    // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
    // where the variable is.
    Basic.identity(value)
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
  def sparseRead(indices: Op.Output, name: String = "Gather"): Op.Output = {
    val value = Op.createWith(nameScope = this.name, device = variableOp.device) { // TODO: Reset colocation ops?
      Variable.gather(variableOp, indices, dataType, validateIndices = true, name)
    }
    // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
    // where the variable is.
    Basic.identity(value)
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
  def evaluate(feeds: Map[Op.Output, Tensor] = Map.empty, session: Session = null): Tensor = {
    toOpOutput.evaluate(feeds, session)
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
  def assign(value: Op.Output, name: String = "Assign"): Op.Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(controlDependencies = Set[Op](Variable.assign(variableOp, value, name))) {
      read()
    }
  }

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  def assignAdd(value: Op.Output, name: String = "AssignAdd"): Op.Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(controlDependencies = Set[Op](Variable.assignAdd(variableOp, value, name))) {
      read()
    }
  }

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  def assignSub(value: Op.Output, name: String = "AssignAdd"): Op.Output = {
    if (value.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${value.dataType}'.")
    Op.createWith(controlDependencies = Set[Op](Variable.assignSub(variableOp, value, name))) {
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
  def assignScatterAdd(indices: Op.Output, values: Op.Output, name: String = "AssignScatterAdd"): Op.Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(controlDependencies = Set[Op](Variable.scatterAdd(variableOp, indices, values, name))) {
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
  def assignScatterSub(indices: Op.Output, values: Op.Output, name: String = "AssignScatterAdd"): Op.Output = {
    if (values.dataType != dataType)
      throw InvalidDataTypeException(s"Expected '$dataType', but got '${values.dataType}'.")
    Op.createWith(controlDependencies = Set[Op](Variable.scatterAdd(variableOp, indices, -values, name))) {
      read()
    }
  }

  // Useful operator overloads for the assignment methods:

  def update(value: Op.Output): Unit = assign(value)

  def +=(value: Op.Output): Unit = assignAdd(value)
  def -=(value: Op.Output): Unit = assignSub(value)

  //endregion Assignment Ops

  /** Converts this variable to an op output. This function simply returns an op corresponding to the variable value. */
  def toOpOutput: Op.Output = value

  override def toProto(exportScope: String = null): VariableDef = {
    if (exportScope == null || variableOp.name.startsWith(exportScope)) {
      val variableDefBuilder = VariableDef.newBuilder()
      variableDefBuilder.setVariableName(Op.stripNameScope(exportScope, variableOp.name))
      variableDefBuilder.setInitializerName(Op.stripNameScope(exportScope, initializeOp.name))
      if (cachedValueOp ne null)
        variableDefBuilder.setSnapshotName(Op.stripNameScope(exportScope, cachedValueOp.name))
      variableDefBuilder.setIsResource(true)
      if (saveSliceInformation != null)
        variableDefBuilder.mergeSaveSliceInfoDef(saveSliceInformation.toProto(exportScope))
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
object Variable {
  /** Creates a variable.
    *
    * @param  initializer   Initializer that generates the op output that will be used as the initial value of this
    *                       variable.
    * @param  shape         Shape for the value of the created variable.
    * @param  dataType      Data type for the value of the created variable. If not provided, its value is inferred from
    *                       the provided initial value.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables to
    *                       use by the optimizers.
    * @param  collections   Set of graph collections keys. The new variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Optional device specification describing where the variable should be cached for reading.
    *                       Defaults to the variable's device. Typical use is to cache on the device where the ops using
    *                       the Variable reside, to deduplicate copying through `Switch` and other conditional
    *                       statements.
    * @param  name          Created variable name.
    * @return Created variable.
    */
  def apply(
      initializer: Initializer, shape: Shape, dataType: DataType, trainable: Boolean = true,
      collections: Set[String] = Set.empty, cachingDevice: OpSpecification => String = null,
      name: String = "Variable"): Variable = {
    Op.createWith(nameScope = name, controlDependencies = Set.empty[Op]) {
      val nameScope = Op.currentNameScope
      val trueName = Op.convertNameScopeToName(nameScope)
      val variableOp = variable(shape, dataType, sharedName = trueName, name = nameScope)
      val initialValue = Op.createWith(nameScope = "Initializer", colocationOps = Set[Op](variableOp.op)) {
        initializer(shape, dataType, null)
      }
      val initializeOp = assign(variableOp, initialValue, name = "InitializationAssign")
      val cachedValueOp = Op.createWith(nameScope = "Read", colocationOps = Set[Op](variableOp.op)) {
        val cachedValueOp = {
          if (cachingDevice != null) {
            // Manually assign reads to the handle's device to avoid log messages
            val valueOp = Op.createWith(device = variableOp.device)(readVariable(variableOp, dataType))
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

      val createdVariable = Variable(dataType, variableOp, initializeOp, cachedValueOp)
      var effectiveCollections = collections
      if (effectiveCollections.isEmpty)
        effectiveCollections += Graph.Keys.GLOBAL_VARIABLES
      if (trainable)
        effectiveCollections += Graph.Keys.TRAINABLE_VARIABLES
      createdVariable.graph.addToCollections(createdVariable, effectiveCollections)
      createdVariable
    }
  }

  /** Creates a variable from the provided ProtoBuf object.
    *
    * @param  protoDef    ProtoBuf object.
    * @param  importScope Name scope to use for all imported ops.
    * @param  context     Op creation context to use while creating the variable.
    * @return Constructed [[Variable]] object.
    */
  def fromProto(protoDef: VariableDef, importScope: String = null)
      (implicit context: DynamicVariable[OpCreationContext]): Variable = {
    if (!protoDef.getIsResource)
      throw new IllegalArgumentException("Trying to restore a reference-based variable as a resource-based variable.")

    def prependNameScope(name: String) = if (importScope == null) name else Op.prependNameScope(importScope, name)

    val variableOp = context.graph.getOpOutputByName(prependNameScope(protoDef.getVariableName))
    val dataType = variableOp.op.dataTypeAttribute("dtype")
    val initializeOp = context.graph.getOpByName(prependNameScope(protoDef.getInitializerName))
    val cachedValueOp = {
      if (protoDef.getSnapshotName == null)
        null
      else
        context.graph.getOpOutputByName(prependNameScope(protoDef.getSnapshotName))
    }
    val saveSliceInformation = {
      if (protoDef.hasSaveSliceInfoDef)
        SaveSliceInformation.fromProto(protoDef.getSaveSliceInfoDef)
      else
        null
    }
    val createdVariable = Variable(dataType, variableOp, initializeOp, cachedValueOp)
    createdVariable.setSaveSliceInformation(saveSliceInformation)
    createdVariable
  }

  // TODO: [VARIABLES] Add PartitionInfo and variable scope support.
  // TODO: [VARIABLES] Add partitioned variables support.

  case class PartitionInfo(fullShape: Shape, offsets: Array[Int])

  /** Class that information on how to save a variable as a slice.
    *
    * This class provides internal support for saving variables as slices of a larger variable. This API is not public
    * and is subject to change.
    *
    * @param  fullName       Name of the full variable, of which the variable is a slice.
    * @param  fullShape      Shape of the full variable, of which the variable is a slice.
    * @param  variableOffset Offset of the variable into the full variable.
    * @param  variableShape  Shape of the variable.
    */
  private[api] case class SaveSliceInformation(
      fullName: String = null, fullShape: Shape = null, variableOffset: Array[Int] = null,
      variableShape: Array[Int] = null) extends ProtoSerializable {
    /** Returns the spec string used for saving. */
    def spec: String = {
      val shapeString = fullShape.asArray.mkString(" ")
      val sliceString = variableOffset.zip(variableShape).map(p => s"${p._1},${p._2}").mkString(":")
      s"$shapeString $sliceString"
    }

    override def toProto(exportScope: String = null): SaveSliceInfoDef = {
      if (exportScope == null || fullName.startsWith(exportScope)) {
        val saveSliceInfoDefBuilder = SaveSliceInfoDef.newBuilder()
        saveSliceInfoDefBuilder.setFullName(Op.stripNameScope(exportScope, fullName))
        fullShape.asArray.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setFullShape(p._2, p._1.toLong))
        variableOffset.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setVarOffset(p._2, p._1.toLong))
        variableShape.zipWithIndex.foreach(p => saveSliceInfoDefBuilder.setVarShape(p._2, p._1.toLong))
        saveSliceInfoDefBuilder.build()
      } else {
        null
      }
    }
  }

  /** Contains helper functions for creating and dealing with [[SaveSliceInformation]] objects. */
  private[api] object SaveSliceInformation {
    /** Creates a new [[SaveSliceInformation]] from the provided ProtoBuf object.
      *
      * @param  protoDef    ProtoBuf object.
      * @param  importScope Name scope to use for all imported ops.
      * @return Constructed [[SaveSliceInformation]] object.
      */
    def fromProto(protoDef: SaveSliceInfoDef, importScope: String = null): SaveSliceInformation = {
      val fullName = {
        if (importScope == null) protoDef.getFullName else Op.prependNameScope(importScope, protoDef.getFullName)
      }
      val fullShape = Shape.fromSeq((0 until protoDef.getFullShapeCount).map(protoDef.getFullShape(_).toInt))
      val variableOffset = (0 until protoDef.getVarOffsetCount).map(protoDef.getVarOffset(_).toInt).toArray
      val variableShape = (0 until protoDef.getVarShapeCount).map(protoDef.getVarShape(_).toInt).toArray
      SaveSliceInformation(fullName, fullShape, variableOffset, variableShape)
    }
  }

  /** Creates an op that initializes the provided variables.
    *
    * @param  variables Set of variables to initialize.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def initializer(variables: Set[Variable], name: String = "Initializer"): Op = {
    if (variables != null && variables.nonEmpty)
      ControlFlow.group(variables.map(_.initializer), name)
    else
      ControlFlow.noOp(name)
  }

  /** Base trait for all variable initializers. */
  trait Initializer {
    def apply(shape: Shape, dataType: DataType, partitionInfo: PartitionInfo): Op.Output = {
      initialValue(shape, dataType, partitionInfo)
    }

    /** Generates an initial value op.
      *
      * @param  shape         Shape for the output tensor.
      * @param  dataType      Data type for the output tensor.
      * @param  partitionInfo [[PartitionInfo]] object holding additional information about how the variable is
      *                       partitioned. May be `null` if the variable is not partitioned.
      * @return Created op output.
      */
    def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInfo): Op.Output
  }

  /** Initializer that sets all elements of the variable tensor to zeros. */
  object ZerosInitializer extends Initializer {
    override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInfo): Op.Output = {
      Basic.zeros(shape, dataType)
    }
  }

  /** Initializer that sets all elements of the variable tensor to ones. */
  object OnesInitializer extends Initializer {
    override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInfo): Op.Output = {
      Basic.ones(shape, dataType)
    }
  }

  /** Initializer that sets the value of the variable to the provided `value`. */
  case class ConstantInitializer(value: Tensor) extends Initializer {
    override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInfo): Op.Output = {
      Basic.constant(value, dataType, shape)
    }
  }

  // TODO: [VARIABLE_INITIALIZERS] RandomUniform/Normal, TruncatedNormal, UniformUnitScaling, Orthogonal.
  // TODO: [VARIABLE_INITIALIZERS] VarianceScaling, Glorot/Xavier Uniform and Normal

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
  private def unsafeReadVariable(
      variable: Op.Output, dataType: DataType, name: String = "UnsafeReadVariable"): Op.Output = {
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
  private def destroyVariable(
      variable: Op.Output, ignoreLookupError: Boolean = true, name: String = "DestroyVariable"): Op = {
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
  private def assign(variable: Op.Output, value: Op.Output, name: String = "AssignVariable"): Op = {
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
    * @param  indices         Indices tensor, which must be an `TFInt32` or `TFInt64` tensor.
    * @param  dataType        Data type for the created op.
    * @param  validateIndices Boolean value indicating whether to validate the provided indices.
    * @param  name            Name for the created op.
    * @return Created op.
    */
  private def gather(
      variable: Op.Output, indices: Op.Output, dataType: DataType = null, validateIndices: Boolean = true,
      name: String = "VariableGather"): Op.Output = {
    if (indices.dataType != TFInt32 && indices.dataType != TFInt64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable gather op indices. " +
            s"Only 'TFInt32' and 'TFInt64' are supported.")
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
    * @param  indices  Indices tensor, which must be an `TFInt32` or `TFInt64` tensor.
    * @param  updates  Updates tensor, which must have a numeric data type.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private def scatterAdd(
      variable: Op.Output, indices: Op.Output, updates: Op.Output, name: String = "ScatterAdd"): Op = {
    if (indices.dataType != TFInt32 && indices.dataType != TFInt64)
      throw InvalidDataTypeException(
        s"Data type '${indices.dataType}' is not supported for the resource variable scatter add op indices. " +
            s"Only 'TFInt32' and 'TFInt64' are supported.")
    Op.Builder(opType = "ResourceScatterAdd", name = name)
        .addInput(variable)
        .addInput(indices)
        .addInput(updates)
        .build()
  }

  object Gradients {
    GradientsRegistry.register("ReadVariableOp", readGradient)

    def readGradient(op: Op, outputGradients: Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
      Seq(outputGradients.head)
    }
  }
}
