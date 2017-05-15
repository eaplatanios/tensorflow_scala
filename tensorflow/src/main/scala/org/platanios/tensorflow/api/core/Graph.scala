package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.exception.{GraphMismatchException, InvalidGraphElementException}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op}
import org.platanios.tensorflow.api.ops.variables.{Variable, VariableScope, VariableStore}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.STRING
import org.platanios.tensorflow.api.{Closeable, ProtoSerializable}
import org.platanios.tensorflow.jni.{Graph => NativeGraph}
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Graph(private[api] var nativeHandle: Long) extends Closeable with ProtoSerializable {
  private[this] object NativeHandleLock

  // TODO: Need to be able to reset and close this session.
  private[api] val defaultSession: Session = Session(this)

  /** Map from native op handle to op object in the Scala side. Used for caching ops that have already been obtained
    * from the native library. */
  private[api] val opsCache: mutable.Map[Long, Op] = mutable.LongMap.empty[Op]

  /** Variable store object of this graph, used to store created variables and keep track of variable scope usages. */
  private[api] val variableStore: VariableStore = VariableStore()

  /** Set that contains the current names in use in this graph. */
  private[this] val namesInUse: mutable.Set[String] = mutable.Set.empty[String]

  /** Marks `name` as a used name in this graph (i.e., increments its usage counter). */
  private[api] def markNameAsUsed(name: String): Unit = namesInUse synchronized {
    namesInUse += name
  }

  /** Returns a unique op name in this graph, based on the provided `name`.
    *
    * @note Operation names are displayed in error messages reported by the TensorFlow runtime, and in various
    *       visualization tools such as TensorBoard.
    * @note You rarely need to call `uniqueName` directly. Most of the time you just need to create
    *       `Op.createWithNameScope(...)` (which is also thread-safe) blocks to generate structured names.
    * @param  name       Name in which to base the generated unique name.
    * @param  markAsUsed If `true`, which is the default, a new unique name is created and marked as in use. If `false`,
    *                    the unique name is returned without actually being marked as used. This is useful when the
    *                    caller simply wants to know what the name to be created will be.
    * @return Unique name.
    */
  private[api] def uniqueName(name: String, markAsUsed: Boolean = true): String = namesInUse synchronized {
    val nameScope = Op.convertNameScopeToName(Op.currentNameScope)
    val fullName = {
      if (nameScope == null || nameScope == "")
        name
      else
        s"$nameScope/$name"
    }
    var count = if (namesInUse.contains(fullName)) 1 else 0
    // Increment the counter for the provided name.
    if (markAsUsed)
      namesInUse += fullName
    if (count > 0) {
      var uniqueName = fullName
      // Make sure the composed name is not already being used.
      while (namesInUse.contains(uniqueName)) {
        uniqueName = s"${fullName}_$count"
        count += 1
      }
      // Mark the composed name as used.
      if (markAsUsed)
        namesInUse += uniqueName
      uniqueName
    } else {
      fullName
    }
  }

  // TODO: Sacrificing type-safety here.
  /** Map from collection name to set of objects in that collection. */
  private[this] val collections: mutable.Map[String, mutable.Set[Any]] = mutable.Map.empty[String, mutable.Set[Any]]

  /** Set of all unfeedable ops in this graph. */
  private[this] val unfeedableOpOutputs: mutable.Set[Op.Output] = mutable.Set.empty

  // TODO: Maybe this should contain ops rather than op outputs.
  /** Set of all unfetchable ops in this graph. */
  private[this] val unfetchableOpOutputs: mutable.Set[Op.Output] = mutable.Set.empty

  /** Adds `value` to the collection with name `key`.
    *
    * @param  value Value to add to the collection.
    * @param  key   Collection name.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def addToCollection(value: Op.Output, key: String): Unit = {
    if (value.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    collections.getOrElseUpdate(key, mutable.Set.empty[Any]) += value
  }

  /** Adds `value` to the collections specified by the names in `keys`.
    *
    * @param  value Value to add to the collections.
    * @param  keys  Collection names.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def addToCollections(value: Op.Output, keys: Set[String]): Unit = {
    keys.foreach(addToCollection(value, _))
  }

  /** Adds `variable` to the collection with name `key`.
    *
    * @param  variable Variable to add to the collection.
    * @param  key      Collection name.
    * @throws GraphMismatchException If the provided variable does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def addToCollection(variable: Variable, key: String): Unit = {
    if (variable.graph != this)
      throw GraphMismatchException("The provided variable does not belong to this graph.")
    collections.getOrElseUpdate(key, mutable.Set.empty[Any]) += variable
  }

  /** Adds `variable` to the collections specified by the names in `keys`.
    *
    * @param  variable Variable to add to the collections.
    * @param  keys     Collection names.
    * @throws GraphMismatchException If the provided variable does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def addToCollections(variable: Variable, keys: Set[String]): Unit = {
    keys.foreach(addToCollection(variable, _))
  }

  /** Adds `store` to the collection with name `key`.
    *
    * @param  store Variable store to add to the collection.
    * @param  key   Collection name.
    */
  private[api] def addToCollection(store: VariableStore, key: String): Unit = {
    collections.getOrElseUpdate(key, mutable.Set.empty[Any]) += store
  }

  /** Adds `store` to the collections specified by the names in `keys`.
    *
    * @param  store Variable store to add to the collections.
    * @param  keys  Collection names.
    */
  private[api] def addToCollections(store: VariableStore, keys: Set[String]): Unit = {
    keys.foreach(addToCollection(store, _))
  }

  /** Adds `scope` to the collection with name `key`.
    *
    * @param  scope Variable scope to add to the collection.
    * @param  key   Collection name.
    */
  private[api] def addToCollection(scope: VariableScope, key: String): Unit = {
    collections.getOrElseUpdate(key, mutable.Set.empty[Any]) += scope
  }

  /** Adds `scope` to the collections specified by the names in `keys`.
    *
    * @param  scope Variable scope to add to the collections.
    * @param  keys  Collection names.
    */
  private[api] def addToCollections(scope: VariableScope, keys: Set[String]): Unit = {
    keys.foreach(addToCollection(scope, _))
  }

  /** Gets the set of objects contained in the collection with name `key`.
    *
    * Note that this method returns an immutable copy of the set.
    *
    * @param  key Collection name.
    * @return Set of objects contained in the collection with name `collection`.
    */
  private[api] def getCollection(key: String): Set[Any] = {
    collections.getOrElse(key, mutable.Set.empty[Any]).toSet[Any]
  }

  /** Gets the set of objects that corresponds to the collection with name `key`.
    *
    * Note that this method returns a reference to the underlying mutable set and so any changes made to that set will
    * be reflected in the corresponding collection.
    *
    * @param  key Collection name.
    * @return Mutable set of objects that corresponds to the collection with name `collection`.
    */
  private[api] def getCollectionReference(key: String): mutable.Set[Any] = {
    collections.getOrElseUpdate(key, mutable.Set.empty[Any])
  }

  /** Returns all the collection keys used in this graph.
    *
    * @return Set containing all the collection keys used in this graph.
    */
  private[api] def getCollectionKeys: Set[String] = collections.filter(_._2.nonEmpty).keys.toSet[String]

  /** Clears the collection with name `key`. This means that the corresponding collection is removed entirely from this
    * graph.
    *
    * @param  key Collection name.
    */
  private[api] def clearCollection(key: String): Unit = {
    collections -= key
  }

  /** Returns the set of global variables in this graph.
    *
    * Global variables are variables that are shared across machines in a distributed environment. The `Variable()`
    * constructor and the function `getVariable()` automatically add new variables to the graph collection with key
    * `Graph.Keys.GLOBAL_VARIABLES`. This convenience function returns the contents of that collection.
    *
    * An alternative to global variables are local variables.
    *
    * @return Set of global variables in this graph.
    */
  def globalVariables: Set[Variable] = {
    getCollection(Graph.Keys.GLOBAL_VARIABLES).map(_.asInstanceOf[Variable])
  }

  /** Returns the set of local variables in this graph.
    *
    * Local variables (or per-process variables), are usually not saved/restored to/from checkpoints and are used for
    * temporary or intermediate values. For example, they can be used as counters for metrics computations or number of
    * epochs this machine has read data. This convenience function returns the contents of that collection.
    *
    * An alternative to local variables are global variables.
    *
    * @return Set of local variables in this graph.
    */
  def localVariables: Set[Variable] = {
    getCollection(Graph.Keys.LOCAL_VARIABLES).map(_.asInstanceOf[Variable])
  }

  /** Returns the subset of `Variable` objects that are used in models for inference (feed forward), in this graph.
    *
    * @return Set of model variables in this graph.
    */
  def modelVariables: Set[Variable] = {
    getCollection(Graph.Keys.MODEL_VARIABLES).map(_.asInstanceOf[Variable])
  }

  /** Returns the set of all variables created with `trainable = true`.
    *
    * When passed `trainable = true`, the `Variable()` constructor automatically adds new variables to the graph
    * collection with key `Graph.Keys.TRAINABLE_VARIABLES`. This convenience function returns the contents of that
    * collection.
    *
    * @return Set of trainable variables in this graph.
    */
  def trainableVariables: Set[Variable] = {
    getCollection(Graph.Keys.TRAINABLE_VARIABLES).map(_.asInstanceOf[Variable])
  }

  /** Returns the set of all the summary `Op.Output`s that have been created in the graph.
    *
    * @return Set of summary op outputs in this graph.
    */
  def summaries: Set[Op.Output] = {
    getCollection(Graph.Keys.SUMMARIES).map(_.asInstanceOf[Op.Output])
  }

  /** Returns the set of all the train `Op`s (i.e., optimizer update ops) that have been created in the graph.
    *
    * @return Set of train ops in this graph.
    */
  def trainOps: Set[Op] = {
    getCollection(Graph.Keys.TRAIN_OP).map(_.asInstanceOf[Op])
  }

  /** Returns an op that initializes all global variables of this graph.
    *
    * For more information, refer to [[globalVariables]] and [[Variable.initializer]].
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): Op = {
    Variable.initializer(globalVariables, name)
  }

  /** Returns an op that initializes all local variables of this graph.
    *
    * For more information, refer to [[localVariables]] and [[Variable.initializer]].
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def localVariablesInitializer(name: String = "LocalVariablesInitializer"): Op = {
    Variable.initializer(localVariables, name)
  }

  /** Returns an op that initializes all model variables of this graph.
    *
    * For more information, refer to [[modelVariables]] and [[Variable.initializer]].
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): Op = {
    Variable.initializer(modelVariables, name)
  }

  /** Returns an op that initializes all trainable variables of this graph.
    *
    * For more information, refer to [[trainableVariables]] and [[Variable.initializer]].
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): Op = {
    Variable.initializer(trainableVariables, name)
  }

  /** Creates an op that lists the names of uninitialized variables.
    *
    * When run, it returns a one-dimensional tensor containing the names of uninitialized variables if there are any, or
    * an empty tensor if there are none.
    *
    * @param  variables Optional set of variables to check. If `null`, the value of `globalVariables ++ localVariables`
    *                   is used. Defaults to `null`.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def reportUninitializedVariables(
      variables: Set[Variable] = null, name: String = "ReportUninitializedVariables"): Op.Output = {
    val actualVariables = if (variables != null) variables else globalVariables ++ localVariables
    Op.createWithNameScope(name) {
      if (actualVariables.isEmpty) {
        // Return an empty tensor so we only need to check for returned tensor size being equal to zero as an indication
        // of the model being ready.
        Basic.constant(Tensor(STRING))
      } else {
        // Get a one-dimensional boolean tensor listing whether each variable is initialized.
        val variablesMask = Math.logicalNot(Basic.stack(variables.map(_.isInitialized).toArray))
        // Get a one-dimensional string tensor containing all the variable names.
        val variableNames = Basic.constant(Tensor(variables.map(v => Tensor(STRING, v.op.name)).toSeq: _*))
        // Return a one-dimensional tensor containing the names of all uninitialized variables.
        Basic.booleanMask(variableNames, variablesMask)
      }
    }
  }

  /** Prevents the feeding of values to the provided op output, while running in a session.
    *
    * @param  opOutput Op output whose feeding is prevented.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def preventFeeding(opOutput: Op.Output): Unit = {
    if (opOutput.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    unfeedableOpOutputs += opOutput
  }

  /** Prevents the fetching of values to the provided op output, while running in a session.
    *
    * @param  opOutput Op output whose fetching is prevented.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def preventFetching(opOutput: Op.Output): Unit = {
    if (opOutput.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    unfetchableOpOutputs += opOutput
  }

  /** Returns `true` if the provided op output is allowed to be fed values, while running in a session.
    *
    * @param  opOutput Op output to check.
    * @return Boolean value indicating whether `opOutput` is allowed to be fed values, while running in a session.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def isFeedable(opOutput: Op.Output): Boolean = {
    if (opOutput.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    !unfeedableOpOutputs.contains(opOutput)
  }

  /** Returns `true` if the provided op output's value is allowed to be fetched, while running in a session.
    *
    * @param  opOutput Op output to check.
    * @return Boolean value indicating whether `opOutput`'s value is allowed to be fetched, while running in a session.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def isFetchable(opOutput: Op.Output): Boolean = {
    if (opOutput.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    !unfetchableOpOutputs.contains(opOutput)
  }

  /** Returns the op with the specified name.
    *
    * @param  name Op name.
    * @return Option containing the op corresponding to that name (`None` if such an op does not exist in this graph).
    */
  private[api] def findOp(name: String): Option[Op] = NativeHandleLock.synchronized {
    val opHandle: Long = NativeGraph.findOp(nativeHandle, name)
    if (opHandle == 0)
      None
    else
      Some(opsCache.getOrElseUpdate(opHandle, Op(this, opHandle)))
  }

  /** Returns all ops of this graph.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    * @return Array containing all ops of this graph.
    */
  def ops: Array[Op] = NativeHandleLock.synchronized {
    NativeGraph.ops(nativeHandle).map(handle => opsCache.getOrElseUpdate(handle, Op(this, handle)))
  }

  /** Returns the op referred to by the provided name, in this graph.
    *
    * If such an op cannot be found, an informative exception is thrown.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    * @param  name Op name.
    * @return Op, from this graph, corresponding to that name.
    */
  @throws[InvalidGraphElementException]
  def getOpByName(name: String): Op = {
    getByName(name = name, allowOp = true, allowOpOutput = false).left.get
  }

  /** Returns the op output referred to by the provided name, in this graph.
    *
    * If such an op output cannot be found, an informative exception is thrown.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    * @param  name Op output name.
    * @return Op output, from this graph, corresponding to that name.
    */
  @throws[InvalidGraphElementException]
  def getOpOutputByName(name: String): Op.Output = {
    getByName(name = name, allowOp = false, allowOpOutput = true).right.get
  }

  /** Returns the [[Op]] or [[Op.Output]] referred to by the provided name, in this graph.
    *
    * This function validates that `name` refers to an element of this graph, and gives an informative error message if
    * it does not. It is the canonical way to get/validate an [[Op]] or [[Op.Output]] from an external argument
    * reference in the Session API. The vast majority of this function is figuring out what an API user might be doing
    * wrong, so that we can give helpful error messages.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    * @param  name          Name of the graph element being looked up.
    * @param  allowOp       Allow ops to be considered for the graph element to return.
    * @param  allowOpOutput Allow op outputs to be considered for the graph element to return.
    * @return Graph element named `name`.
    * @throws InvalidGraphElementException If the provided name cannot be associated with an element of this graph.
    */
  @throws[InvalidGraphElementException]
  private[api] def getByName(
      name: String, allowOp: Boolean = true, allowOpOutput: Boolean = true): Either[Op, Op.Output] = {
    NativeHandleLock.synchronized {
      if (!allowOpOutput && !allowOp)
        throw new IllegalArgumentException("'allowOpOutput' and 'allowOp' cannot both be set to 'false'.")
      if (name.contains(':')) {
        if (allowOpOutput) {
          val nameParts = name.split(':')
          if (nameParts.length != 2 || !nameParts(1).matches("\\d+"))
            throw InvalidGraphElementException(
              s"Name '$name' looks a like an op output name, but it is not a valid one. Op output names must be of " +
                  "the form \"<op_name>:<output_index>\".")
          val opName = nameParts(0)
          val opOutputIndex = nameParts(1).toInt
          val graphOp = findOp(opName) match {
            case Some(o) => o
            case None => throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does not exist in the graph.")
          }
          if (opOutputIndex > graphOp.numOutputs - 1)
            throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does exist in the graph, but it only has ${graphOp.numOutputs} output(s).")
          Right(graphOp.outputs(opOutputIndex))
        } else {
          throw InvalidGraphElementException(
            s"Name '$name' appears to refer to an op output, but 'allowOpOutput' was set to 'false'.")
        }
      } else if (allowOp) {
        findOp(name) match {
          case Some(o) => Left(o)
          case None => throw InvalidGraphElementException(
            s"Name '$name' refers to an op which does not exist in the graph.")
        }
      } else {
        findOp(name) match {
          case Some(_) => throw InvalidGraphElementException(
            s"Name '$name' appears to refer to an op, but 'allowOp' was set to 'false'.")
          case None =>
        }
        throw InvalidGraphElementException(
          s"Name '$name' looks like an (invalid) op name, and not an op output name. Op output names must be of the " +
              "form \"<op_name>:<output_index>\".")
      }
    }
  }

  /** Imports a serialized representation of a graph into the current graph.
    *
    * @param  graphDef               Serialized representation of a graph that will be imported into this graph.
    * @param  nameScope              Optional prefix that will be prepended to all node names in the graph that is
    *                                being imported to this graph.
    * @param  inputsMap              Optional inputs mapping. For each
    *                                `(source_op_name, source_op_output_index) -> destination_op_output` mapping, the
    *                                importer will  set any imported nodes with input named
    *                                `source_op_name:source_op_output_index` to have that input replaced with
    *                                `destination_op_output`. `source_op_name` refers to a node in the graph to be
    *                                imported, whereas `destination_op_output` references a node already existing in
    *                                this graph.
    * @param  controlDependenciesMap Optional control dependencies mapping. For each `source_op_name -> destination_op`
    *                                mapping, the importer will set any imported ops with control input named
    *                                `source_op_name` to have that input replaced with `destination_op`.
    *                                `source_op_name` refers to a node in the graph to be imported, whereas
    *                                `destination_op` references an op already existing in this graph.
    * @param  controlDependencies    Optional control dependencies set. The importer will make sure that the imported
    *                                graph has a control dependency on all ops in this set. All such ops, should
    *                                therefore be defined in this graph.
    */
  def importGraphDef(
      graphDef: GraphDef, nameScope: String = null, inputsMap: Map[(String, Int), Op.Output] = Map.empty,
      controlDependenciesMap: Map[String, Op] = Map.empty, controlDependencies: Set[Op] = Set.empty): Unit = {
    val prefix = {
      if (nameScope == null || nameScope == "")
        ""
      else if (nameScope.endsWith("/"))
        nameScope
      else
        s"$nameScope/"
    }
    val inputsMapSourceOpNames = inputsMap.map(_._1._1).toArray
    val inputsMapSourceOpOutputIndices = inputsMap.map(_._1._2).toArray
    val inputsMapDestinationOpHandles = inputsMap.map(_._2.op.nativeHandle).toArray
    val inputsMapDestinationOpOutputIndices = inputsMap.map(_._2.index).toArray
    val controlDependenciesMapSourceOpNames = controlDependenciesMap.keys.toArray
    val controlDependenciesMapDestinationOpHandles = controlDependenciesMap.map(_._2.nativeHandle).toArray
    val controlDependenciesOpHandles = controlDependencies.map(_.nativeHandle).toArray
    NativeHandleLock.synchronized {
      NativeGraph.importGraphDef(
        nativeHandle, graphDef.toByteArray, prefix, inputsMapSourceOpNames, inputsMapSourceOpOutputIndices,
        inputsMapDestinationOpHandles, inputsMapDestinationOpOutputIndices, controlDependenciesMapSourceOpNames,
        controlDependenciesMapDestinationOpHandles, controlDependenciesOpHandles)
    }
    // TODO: [PERFORMANCE] Make this faster?
    namesInUse synchronized ops.foreach(op => markNameAsUsed(op.name))
  }

  override def toProto: GraphDef = {
    GraphDef.parseFrom(NativeHandleLock.synchronized(NativeGraph.toGraphDef(nativeHandle)))
  }

  private[this] var referenceCount: Int = 0

  def reference: Reference = Reference()

  // Related native objects (such as the TF_Operation object backing an Operation instance)
  // have a validity tied to that of the Graph. The handles to those native objects are not
  // valid after Graph.close() has been invoked.
  //
  // Instances of the Reference class should be used to ensure the Graph has not been closed
  // while dependent handles are in use.
  final case class Reference private() extends Closeable {
    NativeHandleLock.synchronized {
      if (Graph.this.nativeHandle == 0)
        throw new IllegalStateException("close() has been called on the Graph")
      referenceCount += 1
    }

    override def close(): Unit = {
      NativeHandleLock.synchronized {
        if (Graph.this.nativeHandle != 0) {
          referenceCount -= 1
          if (referenceCount == 0)
            NativeHandleLock.notifyAll()
        }
      }
    }

    private[api] def nativeHandle: Long = {
      NativeHandleLock.synchronized {
        if (Graph.this.nativeHandle != 0)
          Graph.this.nativeHandle
        else
          0
      }
    }
  }

  /** Release resources associated with the Graph.
    *
    * Blocks until there are no active [[Session]] instances referring to this Graph. A Graph
    * is not usable after close returns.
    */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      defaultSession.close()
      if (nativeHandle != 0) {
        while (referenceCount > 0) {
          try {
            NativeHandleLock.wait()
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
              // Possible leak of the graph in this case?
              return
          }
        }
        NativeGraph.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }

  // TODO: [GRAPH] Better implementations for equals and hashCode.

  override def equals(that: Any): Boolean = that match {
    case that: Graph => this.nativeHandle == that.nativeHandle
    case _ => false
  }

  override def hashCode(): Int = nativeHandle.hashCode
}

object Graph {
  def apply(): Graph = Graph(nativeHandle = NativeGraph.allocate())

  /** Imports a graph from the provided serialized graph object.
    *
    * @param  graphDef    ProtoBuf-serialized graph object.
    * @param  importScope Name scope to use for all imported ops.
    * @return Constructed [[Graph]] object.
    */
  def fromProto(graphDef: GraphDef, importScope: String = null): Graph = {
    val graph = Graph()
    graph.importGraphDef(graphDef, importScope)
    graph
  }

  // TODO: [DOC] Complete documentation.
  /**
    *
    * @param  graphDef1
    * @param  graphDef2
    * @param  ignoreInternalAttributes Boolean value indicating whether to ignore internal attributes (i.e., attributes
    *                                  whose names start with `'_'`.
    * @return
    */
  private[api] def equalGraphDef(
      graphDef1: GraphDef, graphDef2: GraphDef, ignoreInternalAttributes: Boolean = true): (Boolean, String) = {
    // We intentionally do not check that the versions of the two GraphDefs match so that this function can be used for
    // less brittle golden file tests.
    var graph1Index = mutable.Map.empty[String, NodeDef]
    var index = 0
    while (index < graphDef1.getNodeCount) {
      val nodeDef = graphDef1.getNode(index)
      graph1Index.update(nodeDef.getName, nodeDef)
      index += 1
    }
    graph1Index.toMap
    index = 0
    while (index < graphDef2.getNodeCount) {
      val node2Def = graphDef2.getNode(index)
      if (!graph1Index.contains(node2Def.getName))
        return (false, s"Graph 1 does not contain node '${node2Def.getName}' which graph 2 does.")
      val (equal, difference) = equalNodeDef(graph1Index(node2Def.getName), node2Def, ignoreInternalAttributes)
      if (!equal)
        return (equal, difference)
      graph1Index -= node2Def.getName
      index += 1
    }

    (true, null)
  }

  private[api] def equalNodeDef(
      nodeDef1: NodeDef, nodeDef2: NodeDef, ignoreInternalAttributes: Boolean = true): (Boolean, String) = {
    if (nodeDef1.getName != nodeDef2.getName)
      return (false, s"Node 1 name '${nodeDef1.getName}' does not match node 2 name '${nodeDef2.getName}'.")
    if (nodeDef1.getOp != nodeDef2.getOp)
      return (false, s"Node 1 named '${nodeDef1.getName}' has op '${nodeDef1.getOp}' which does not match node 2's " +
          s"op '${nodeDef2.getOp}'.")
    if (nodeDef1.getDevice != nodeDef2.getDevice)
      return (false, s"Node 1 named '${nodeDef1.getName}' has device '${nodeDef1.getDevice}' which does not match " +
          s"node 2's device '${nodeDef2.getDevice}'.")
    if (nodeDef1.getInputCount != nodeDef2.getInputCount)
      return (false, s"Node 1 named '${nodeDef1.getName}' has '${nodeDef1.getInputCount}' inputs which does not " +
          s"match node 2's '${nodeDef2.getInputCount}' inputs.")

    // Check the inputs
    var firstControlInput = -1
    var index = 0
    while (index < nodeDef1.getInputCount && firstControlInput < 0) {
      val node1Input = nodeDef1.getInput(index)
      val node2Input = nodeDef2.getInput(index)
      if (node1Input.startsWith("^"))
        firstControlInput = index
      else if (node1Input != node2Input)
        return (false, s"Node 1 named '${nodeDef1.getName}' has input $index '$node1Input' which does not match " +
            s"node 2's input '$node2Input'.")
      index += 1
    }

    // Check the control inputs
    if (firstControlInput > 0) {
      var node1ControlInputs = (firstControlInput until nodeDef1.getInputCount).map(nodeDef1.getInput).toSet
      val node2ControlInputs = (firstControlInput until nodeDef2.getInputCount).map(nodeDef2.getInput).toSet
      for (node <- node2ControlInputs) {
        if (!node1ControlInputs.contains(node))
          return (false, s"Node 1 named '${nodeDef1.getName}' does not have control input '$node' that node 2 has.")
        node1ControlInputs -= node
      }
      if (node1ControlInputs.nonEmpty)
        return (false, s"Node 1 named '${nodeDef1.getName}' has control input '${node1ControlInputs.head}' that node 2 " +
            s"does not have.")
    }

    // Check the attributes
    var node1Attributes = nodeDef1.getAttrMap.asScala.filterKeys(k => !ignoreInternalAttributes || !k.startsWith("_"))
    for ((name, value) <- nodeDef2.getAttrMap.asScala) {
      if (!ignoreInternalAttributes || !name.startsWith("_")) {
        if (!node1Attributes.contains(name))
          return (false, s"Node 1 named '${nodeDef1.getName}' does not contain attribute '$name' that node 2 does.")
        // TODO: [PROTO] Implement attr_value_utils and node_def_utils to provide better summaries.
        if (node1Attributes(name).toString != value.toString)
          return (false, s"Node 1 named '${nodeDef1.getName}' has different value for attribute '$name' than node 2.")
        node1Attributes -= name
      }
    }
    if (node1Attributes.nonEmpty)
      return (false, s"Node 1 named '${nodeDef1.getName}' has attribute '${node1Attributes.head}' that node 2 " +
          s"does not have.")

    (true, null)
  }

  /** Contains standard names to use for graph collections.
    *
    * The standard library uses various well-known names to collect and retrieve values associated with a graph. For
    * example, the optimizers default to optimizing the variables collected under `Graph.Keys.TRAINABLE_VARIABLES` if
    * none is specified, but it is also possible to pass an explicit list of variables.
    */
  object Keys {
    /** Key to collect the default collection of `Variable` objects, shared across distributed environment (model
      * variables are subset of these). Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`, and
      * all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`. */
    val GLOBAL_VARIABLES = "variables"

    /** Key to collect the subset of `Variable` objects that are local to each machine. Usually used for temporary
      * variables, like counters.
      * TODO: Note: use `tf.contrib.framework.local_variable` to add to this collection. */
    val LOCAL_VARIABLES = "local_variables"

    /** Key to collect the subset of `Variable` objects that are used in models for inference (feed forward).
      * TODO: Note: use `tf.contrib.framework.model_variable` to add to this collection. */
    val MODEL_VARIABLES = "model_variables"

    /** Key to collect the subset of `Variable` objects that will be trained using an optimizer. */
    val TRAINABLE_VARIABLES = "trainable_variables"

    /** Key to collect the summary `Op.Output` objects that have been created in the graph. */
    val SUMMARIES = "summaries"

    /** Key to collect the `QueueRunner` objects that are used to produce inputs for a computation. */
    val QUEUE_RUNNERS = "queue_runners"

    /** Key to collect table initializer objects. */
    val TABLE_INITIALIZERS = "table_initializer"

    /** Key to collect asset filepaths. An asset represents an external resource like a vocabulary file. */
    val ASSET_FILEPATHS = "asset_filepaths"

    /** Key to collect the subset of `Variable` objects that will also keep moving averages. */
    val MOVING_AVERAGE_VARIABLES = "moving_average_variables"

    /** Key to collect regularization losses at graph construction. */
    val REGULARIZATION_LOSSES = "regularization_losses"

    /** Key to collect concatenated sharded variables. */
    val CONCATENATED_VARIABLES = "concatenated_variables"

    /** Key to collect savers. */
    val SAVERS = "savers"

    /** Key to collect weights. */
    val WEIGHTS = "weights"

    /** Key to collect biases. */
    val BIASES = "biases"

    /** Key to collect activations. */
    val ACTIVATIONS = "activations"

    /** Key to collect update ops. */
    val UPDATE_OPS = "update_ops"

    /** Key to collect losses. */
    val LOSSES = "losses"

    /** Key to collect saveable objects used for checkpoints. */
    val SAVEABLE_OBJECTS = "saveable_objects"

    /** Key to collect all shared resources used by the graph which need to be initialized once per cluster. */
    val RESOURCES = "resources"

    /** Key to collect all shared resources used in this graph which need to be initialized once per session. */
    val LOCAL_RESOURCES = "local_resources"

    /** Key to collect all trainable resource-style variables. */
    val TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

    // Keys to indicate various ops.
    val INIT_OP                 = "init_op"
    val LOCAL_INIT_OP           = "local_init_op"
    val READY_OP                = "ready_op"
    val READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
    val SUMMARY_OP              = "summary_op"
    val GLOBAL_STEP             = "global_step"
    val EVAL_STEP               = "eval_step" // Used to count the number of evaluations performed during a single evaluation run.
    val TRAIN_OP = "train_op"

    // Keys for control flow management.
    val COND_CONTEXT  = "cond_context"
    val WHILE_CONTEXT = "while_context"

    /** Key to collect streaming model ports. */
    val STREAMING_MODEL_PORTS = "streaming_model_ports"
  }
}
