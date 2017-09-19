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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.{GraphMismatchException, InvalidGraphElementException}
import org.platanios.tensorflow.api.ops.{Basic, InstantiatedFunction, Math, Op, Output, Resource}
import org.platanios.tensorflow.api.ops.variables.{Saver, Variable, VariableStore}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.STRING
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.jni.{Function => NativeFunction, Graph => NativeGraph, TensorFlow => NativeLibrary}
import com.google.protobuf.ByteString
import org.tensorflow.framework.CollectionDef.{BytesList, Int64List, NodeList}
import org.tensorflow.framework.MetaGraphDef.MetaInfoDef
import org.tensorflow.framework._
import org.tensorflow.util.SaverDef

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.postfixOps
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
class Graph private[api](private[api] var nativeHandle: Long) extends Closeable with ProtoSerializable {
  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the graph is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** List of functions that will be called right before disposing this graph object. Such functions are usually used to
    * clean up native resources used by this graph. */
  private[api] val cleanupFunctions: mutable.ListBuffer[() => Unit] = mutable.ListBuffer.empty

  /** Adds a cleanup function to this graph. That is, a function that will be called right before disposing this graph
    * object. Such functions are usually used to clean up native resources used by this graph. */
  private[api] def addCleanupFunction(function: () => Unit): Unit = cleanupFunctions.append(function)

  // TODO: [SESSION] Need to be able to reset this session.
  private[api] val defaultSession: Session = Session(this)

  /** Map from native op handle to op object in the Scala side. Used for caching ops that have already been obtained
    * from the native library. */
  private[api] val opsCache: mutable.Map[Long, Op] = mutable.LongMap.empty[Op]

  /** Variable store object of this graph, used to store created variables and keep track of variable scope usages. */
  private[api] val variableStore: VariableStore = VariableStore()

  /** Set that contains the current names in use in this graph. */
  private[this] val namesInUse: mutable.Set[String] = mutable.Set.empty[String]

  /** Marks `name` as a used name in this graph (i.e., increments its usage counter). */
  private[this] def markNameAsUsed(name: String): Unit = namesInUse synchronized {
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

  /** Helper function for processing tensors before using them as inputs for ops placed in this graph. Useful for
    * creating function graphs. */
  private[api] def processOpInput(output: Output): Output = output

  /** Map from function name to function instance, for functions added to this graph. */
  private[this] val functionsMap: mutable.Map[String, InstantiatedFunction[_, _]] = mutable.Map.empty

  /** Returns the set of functions that have been added to this graph. */
  private[api] def functions: Set[InstantiatedFunction[_, _]] = functionsMap.values.toSet

  /** Adds the provided function instance to this graph. */
  private[api] def addFunction(function: InstantiatedFunction[_, _]): Unit = {
    if (!functionsMap.contains(function.name)) {
      NativeFunction.addToGraph(nativeHandle, function.nativeHandle)
      functionsMap.update(function.name, function)
    }
  }

  /** Return the function instance in this graph corresponding to the provided name. If such a function does not exist
    * in this graph, then `None` is returned. */
  private[api] def getFunction(name: String): Option[InstantiatedFunction[_, _]] = functionsMap.get(name)

  /** Map from collection key to set of values in that collection. */
  private[this] val collections: mutable.Map[Graph.Key[_], mutable.Set[_]] = mutable.Map.empty

  // TODO: [GRAPH] Should we keep track of this and the field that follows in a collection? MetaGraphDef.
  /** Set of all unfeedable ops in this graph. */
  private[this] val unfeedableOutputs: mutable.Set[Output] = mutable.Set.empty

  /** Set of all unfetchable ops in this graph. */
  private[this] val unfetchableOps: mutable.Set[Op] = mutable.Set.empty

  /** Removes the specified collection from this graph.
    *
    * @param  key Collection key.
    */
  private[api] def clearCollection[K](key: Graph.Key[K]): Unit = collections -= key

  /** Adds `value` to the collection with name `key`.
    *
    * @param  value Value to add to the collection.
    * @param  key   Collection name.
    */
  private[api] def addToCollection[K](value: K, key: Graph.Key[K]): Unit = {
    collections.getOrElseUpdate(key, mutable.Set.empty[K]).asInstanceOf[mutable.Set[K]].add(value)
  }

  /** Gets the set of values contained in the collection with name `key`.
    *
    * Note that this method returns an immutable copy of the set.
    *
    * @param  key Collection name.
    * @return Set of values contained in the collection with name `collection`.
    */
  private[api] def getCollection[K](key: Graph.Key[K]): Set[K] = {
    collections.getOrElse(key, mutable.Set.empty[K]).asInstanceOf[mutable.Set[K]].toSet[K]
  }

  /** Gets the set of values that corresponds to the collection with name `key`.
    *
    * Note that this method returns a reference to the underlying mutable set and so any changes made to that set will
    * be reflected in the corresponding collection.
    *
    * @param  key Collection name.
    * @return Mutable set of values that corresponds to the collection with name `collection`.
    */
  private[api] def getCollectionReference[K](key: Graph.Key[K]): mutable.Set[K] = {
    collections.getOrElseUpdate(key, mutable.Set.empty[K]).asInstanceOf[mutable.Set[K]]
  }

  /** Gets the random seed of this graph. */
  def randomSeed: Option[Int] = {
    collections.getOrElseUpdate(Graph.Keys.RANDOM_SEEDS, mutable.Set.empty[Int])
        .asInstanceOf[mutable.Set[Int]].headOption
  }

  /** Sets the random seed of this graph to the provided value. */
  def setRandomSeed(value: Int): Unit = {
    collections.update(Graph.Keys.RANDOM_SEEDS, mutable.Set[Int](value))
  }

  /** Returns the set of global variables in this graph.
    *
    * Global variables are variables that are shared across machines in a distributed environment. The `Variable()`
    * constructor and the function `getVariable()` automatically add new variables to the graph collection with key
    * `Graph.Keys.GLOBAL_VARIABLES`. This convenience function returns the contents of that collection.
    *
    * An alternative to global variables are local variables.
    */
  def globalVariables: Set[Variable] = getCollection(Graph.Keys.GLOBAL_VARIABLES)

  /** Returns the set of local variables in this graph.
    *
    * Local variables (or per-process variables), are usually not saved/restored to/from checkpoints and are used for
    * temporary or intermediate values. For example, they can be used as counters for metrics computations or number of
    * epochs this machine has read data. This convenience function returns the contents of that collection.
    *
    * An alternative to local variables are global variables.
    */
  def localVariables: Set[Variable] = getCollection(Graph.Keys.LOCAL_VARIABLES)

  /** Returns the subset of `Variable` objects that are used in models for inference (feed forward), in this grap. */
  def modelVariables: Set[Variable] = getCollection(Graph.Keys.MODEL_VARIABLES)

  /** Returns the set of all variables created with `trainable = true`.
    *
    * When passed `trainable = true`, the `Variable()` constructor automatically adds new variables to the graph
    * collection with key `Graph.Keys.TRAINABLE_VARIABLES`. This convenience function returns the contents of that
    * collection.
    */
  def trainableVariables: Set[Variable] = getCollection(Graph.Keys.TRAINABLE_VARIABLES)

  /** Returns the set of all the summary `Output`s that have been created in the graph. */
  def summaries: Set[Output] = getCollection(Graph.Keys.SUMMARIES)

  /** Returns the set of all shared resources used by the graph which need to be initialized once per cluster. */
  def sharedResources: Set[Resource] = getCollection(Graph.Keys.SHARED_RESOURCES)

  /** Returns the set of all local resources used by the graph which need to be initialized once per cluster. */
  def localResources: Set[Resource] = getCollection(Graph.Keys.LOCAL_RESOURCES)

  /** Returns the set of all the train `Op`s (i.e., optimizer update ops) that have been created in the graph. */
  def trainOps: Set[Op] = getCollection(Graph.Keys.TRAIN_OP)

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
      variables: Set[Variable] = null, name: String = "ReportUninitializedVariables"): Output = {
    val actualVariables = if (variables != null) variables else globalVariables ++ localVariables
    Op.createWithNameScope(name) {
      if (actualVariables.isEmpty) {
        // Return an empty tensor so we only need to check for returned tensor size being equal to zero as an indication
        // of the model being ready.
        Basic.constant(Tensor(STRING))
      } else {
        // Get a one-dimensional boolean tensor listing whether each variable is initialized.
        val variablesMask = Math.logicalNot(Basic.stack(variables.map(_.isInitialized).toSeq))
        // Get a one-dimensional string tensor containing all the variable names.
        val variableNames = Basic.constant(Tensor(variables.map(v => Tensor(STRING, v.op.name)).toSeq))
        // Return a one-dimensional tensor containing the names of all uninitialized variables.
        Basic.booleanMask(variableNames, variablesMask)
      }
    }
  }

  /** Prevents the feeding of values to the provided op output, while running in a session.
    *
    * @param  output Op output whose feeding is prevented.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def preventFeeding(output: Output): Unit = {
    if (output.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    unfeedableOutputs += output
  }

  /** Prevents the fetching of values to the provided op, while running in a session.
    *
    * @param  op Op whose fetching is prevented.
    * @throws GraphMismatchException If the provided op does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def preventFetching(op: Op): Unit = {
    if (op.graph != this)
      throw GraphMismatchException("The provided op does not belong to this graph.")
    unfetchableOps += op
  }

  /** Returns `true` if the provided op output is allowed to be fed values, while running in a session.
    *
    * @param  output Op output to check.
    * @return Boolean value indicating whether `output` is allowed to be fed values, while running in a session.
    * @throws GraphMismatchException If the provided op output does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def isFeedable(output: Output): Boolean = {
    if (output.graph != this)
      throw GraphMismatchException("The provided op output does not belong to this graph.")
    !unfeedableOutputs.contains(output)
  }

  /** Returns `true` if the provided op's value is allowed to be fetched, while running in a session.
    *
    * @param  op Op to check.
    * @return Boolean value indicating whether `op`'s value is allowed to be fetched, while running in a session.
    * @throws GraphMismatchException If the provided op does not belong to this graph.
    */
  @throws[GraphMismatchException]
  private[api] def isFetchable(op: Op): Boolean = {
    if (op.graph != this)
      throw GraphMismatchException("The provided op does not belong to this graph.")
    !unfetchableOps.contains(op)
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
    getByName(name = name, allowOp = true, allowOutput = false).left.get
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
  def getOutputByName(name: String): Output = {
    getByName(name = name, allowOp = false, allowOutput = true).right.get
  }

  /** Returns the [[Op]] or [[Output]] referred to by the provided name, in this graph.
    *
    * This function validates that `name` refers to an element of this graph, and gives an informative error message if
    * it does not. It is the canonical way to get/validate an [[Op]] or [[Output]] from an external argument
    * reference in the Session API. The vast majority of this function is figuring out what an API user might be doing
    * wrong, so that we can give helpful error messages.
    *
    * @note This function may be called concurrently from multiple threads (i.e., it is thread-safe).
    * @param  name        Name of the graph element being looked up.
    * @param  allowOp     Allow ops to be considered for the graph element to return.
    * @param  allowOutput Allow op outputs to be considered for the graph element to return.
    * @return Graph element named `name`.
    * @throws InvalidGraphElementException If the provided name cannot be associated with an element of this graph.
    */
  @throws[InvalidGraphElementException]
  private[api] def getByName(
      name: String, allowOp: Boolean = true, allowOutput: Boolean = true): Either[Op, Output] = {
    NativeHandleLock.synchronized {
      if (!allowOutput && !allowOp)
        throw new IllegalArgumentException("'allowOutput' and 'allowOp' cannot both be set to 'false'.")
      if (name.contains(':')) {
        if (allowOutput) {
          val nameParts = name.split(':')
          if (nameParts.length != 2 || !nameParts(1).matches("\\d+"))
            throw InvalidGraphElementException(
              s"Name '$name' looks a like an op output name, but it is not a valid one. Op output names must be of " +
                  "the form \"<op_name>:<output_index>\".")
          val opName = nameParts(0)
          val outputIndex = nameParts(1).toInt
          val graphOp = findOp(opName) match {
            case Some(o) => o
            case None => throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does not exist in the graph.")
          }
          if (outputIndex > graphOp.numOutputs - 1)
            throw InvalidGraphElementException(
              s"Name '$name' refers to an op output which does not exist in the graph. More specifically, op, " +
                  s"'$opName', does exist in the graph, but it only has ${graphOp.numOutputs} output(s).")
          Right(graphOp.outputs(outputIndex))
        } else {
          throw InvalidGraphElementException(
            s"Name '$name' appears to refer to an op output, but 'allowOutput' was set to 'false'.")
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
    * @param  graphDef               Serialized representation of the graph that will be imported into this graph.
    * @param  importScope            Optional prefix that will be prepended to all node names in the graph that is
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
      graphDef: GraphDef, importScope: String = null, inputsMap: Map[(String, Int), Output] = Map.empty,
      controlDependenciesMap: Map[String, Op] = Map.empty, controlDependencies: Set[Op] = Set.empty): Unit = {
    val prefix = {
      if (importScope == null || importScope == "")
        ""
      else if (importScope.endsWith("/"))
        importScope
      else
        s"$importScope/"
    }
    val inputsMapSourceOpNames = inputsMap.map(_._1._1).toArray
    val inputsMapSourceOutputIndices = inputsMap.map(_._1._2).toArray
    val inputsMapDestinationOpHandles = inputsMap.map(_._2.op.nativeHandle).toArray
    val inputsMapDestinationOutputIndices = inputsMap.map(_._2.index).toArray
    val controlDependenciesMapSourceOpNames = controlDependenciesMap.keys.toArray
    val controlDependenciesMapDestinationOpHandles = controlDependenciesMap.map(_._2.nativeHandle).toArray
    val controlDependenciesOpHandles = controlDependencies.map(_.nativeHandle).toArray
    NativeHandleLock.synchronized {
      NativeGraph.importGraphDef(
        nativeHandle, graphDef.toByteArray, prefix, inputsMapSourceOpNames, inputsMapSourceOutputIndices,
        inputsMapDestinationOpHandles, inputsMapDestinationOutputIndices, controlDependenciesMapSourceOpNames,
        controlDependenciesMapDestinationOpHandles, controlDependenciesOpHandles)
    }
    // TODO: [PERFORMANCE] Make this faster?
    namesInUse synchronized ops.foreach(op => markNameAsUsed(op.name))
  }

  /** Imports a serialized representation of a graph and its meta-information into the current graph.
    *
    * This function takes a [[MetaGraphDef]] protocol buffer as input and it adds all the nodes from its `graph_def`
    * field to the current graph. It also recreates the desired collections stored in that protocol buffer.
    *
    * In combination with [[toMetaGraphDef]], this function can be used to:
    *   - Serialize a graph along with other objects stored in its collections, into a [[MetaGraphDef]].
    *   - Restart training from saved graphs and checkpoints.
    *   - Run inference from saved graphs and checkpoints.
    *
    * @param  metaGraphDef                Serialized representation of the graph and its meta-information, that will be
    *                                     imported into this graph.
    * @param  importScope                 Optional prefix that will be prepended to all node names in the graph that is
    *                                     being imported to this graph.
    * @param  inputsMap                   Optional inputs mapping. For each
    *                                     `(source_op_name, source_op_output_index) -> destination_op_output` mapping,
    *                                     the importer will  set any imported nodes with input named
    *                                     `source_op_name:source_op_output_index` to have that input replaced with
    *                                     `destination_op_output`. `source_op_name` refers to a node in the graph to be
    *                                     imported, whereas `destination_op_output` references a node already existing
    *                                     in this graph.
    * @param  controlDependenciesMap      Optional control dependencies mapping. For each
    *                                     `source_op_name -> destination_op` mapping, the importer will set any imported
    *                                     ops with control input named `source_op_name` to have that input replaced with
    *                                     `destination_op`. `source_op_name` refers to a node in the graph to be
    *                                     imported, whereas `destination_op` references an op already existing in this
    *                                     graph.
    * @param  controlDependencies         Optional control dependencies set. The importer will make sure that the
    *                                     imported graph has a control dependency on all ops in this set. All such ops,
    *                                     should therefore be defined in this graph.
    * @param  clearDevices                Boolean value indicating whether to clear the device information from the
    *                                     returned node definition.
    * @param  unboundInputsCollectionKey  Collection key for looking up unbound inputs.
    * @param  restoreCollectionsPredicate Function that takes as input a graph collection key and returns a boolean
    *                                     value indicating whether or not to load that collection. Note that the
    *                                     collection specified by `unboundInputsCollectionKey` is never loaded.
    *                                     Defaults to a function that returns `true` for all inputs.
    */
  def importMetaGraphDef(
      metaGraphDef: MetaGraphDef, importScope: String = null, inputsMap: Map[(String, Int), Output] = Map.empty,
      controlDependenciesMap: Map[String, Op] = Map.empty, controlDependencies: Set[Op] = Set.empty,
      clearDevices: Boolean = false, unboundInputsCollectionKey: Graph.Key[String] = Graph.Keys.UNBOUND_INPUTS,
      restoreCollectionsPredicate: Graph.Key[_] => Boolean = _ => true): Unit = {
    if (unboundInputsCollectionKey != null) {
      val collectionDef = metaGraphDef.getCollectionDefOrDefault(unboundInputsCollectionKey.name, null)
      if (collectionDef != null) {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 2)
          throw new IllegalArgumentException("The unbound inputs collection is stored with the wrong type.")
        val values = collectionDef.getBytesList.getValueList.asScala.map(_.toStringUtf8).toSet
        if (inputsMap == null || !values.subsetOf(inputsMap.keySet.map(i => s"${i._1}:${i._2}")))
          throw new IllegalArgumentException("Mappings for all unbound inputs need to be provided in the 'inputsMap'.")
      }
    }
    // Gathers the list of nodes we are interested in.
    val inputGraphDefBuilder = GraphDef.newBuilder(metaGraphDef.getGraphDef)
    if (clearDevices) {
      // Remove all the explicit device specifications. This helps make the graph more portable.
      var nodeIndex = 0
      while (nodeIndex < inputGraphDefBuilder.getNodeCount) {
        val nodeDefBuilder = NodeDef.newBuilder(inputGraphDefBuilder.getNode(nodeIndex))
        nodeDefBuilder.setDevice("")
        inputGraphDefBuilder.setNode(nodeIndex, nodeDefBuilder)
        nodeIndex += 1
      }
    }
    importGraphDef(inputGraphDefBuilder.build(), importScope, inputsMap, controlDependenciesMap, controlDependencies)

    // Restore the collections.
    metaGraphDef.getCollectionDefMap.asScala.foreach {
      case (name, collectionDef) =>
        import Graph.Keys._
        val key = Graph.Keys.fromName(name)
        if (restoreCollectionsPredicate(key) && key != UNBOUND_INPUTS)
          key.parseCollectionDef(collectionDef, this, importScope)
    }
  }

  /** Constructs and returns a [[GraphDef]] object, which is a serialized version of this graph.
    *
    * Note that the [[GraphDef]] does not contain any meta-information about the graph (such as collections information,
    * for example). For a serialized representation of the graph that contains such information, please refer to
    * [[Graph.toMetaGraphDef]].
    */
  def toGraphDef: GraphDef = GraphDef.parseFrom(NativeHandleLock.synchronized(NativeGraph.toGraphDef(nativeHandle)))

  /** Constructs and returns a [[MetaGraphDef]] object using the provided arguments.
    *
    * In combination with [[importMetaGraphDef]], this function can be used to:
    *   - Serialize a graph along with other objects stored in its collections, into a [[MetaGraphDef]].
    *   - Restart training from saved graphs and checkpoints.
    *   - Run inference from saved graphs and checkpoints.
    *
    * @param  exportScope                Optional string specifying the name scope to remove. Only the ops within this
    *                                    name scope will be included in the resulting ProtoBuf object and the export
    *                                    scope will be stripped from their names to allow for easy import into new name
    *                                    scopes.
    * @param  metaInfoDef                [[MetaInfoDef]] associated with the [[MetaGraphDef]] that will be constructed.
    * @param  saverDef                   [[SaverDef]] associated with the [[MetaGraphDef]] that will be constructed.
    * @param  collections                Graph collection keys specifying the collections to include in the
    *                                    [[MetaGraphDef]].
    * @param  unboundInputsCollectionKey Collection key for storing unbound inputs. If provided, a string collection
    *                                    with the given name will be added to the returned [[MetaGraphDef]], containing
    *                                    the names of tensors that must be remapped when importing the [[MetaGraphDef]].
    * @param  clearDevices               Boolean value indicating whether to clear the device information from the
    *                                    returned node definitions.
    * @return Constructed [[MetaGraphDef]].
    */
  def toMetaGraphDef(
      exportScope: String = null, metaInfoDef: MetaInfoDef = null, saverDef: SaverDef = null,
      collections: Set[Graph.Key[_]] = Set.empty,
      unboundInputsCollectionKey: Graph.Key[String] = Graph.Keys.UNBOUND_INPUTS,
      clearDevices: Boolean = false): MetaGraphDef = {
    val unboundInputs = mutable.Set.empty[String]
    val graphDef = {
      val originalGraphDef = toGraphDef
      if (exportScope != null || clearDevices) {
        val graphDefBuilder = GraphDef.newBuilder()
        graphDefBuilder.setVersions(originalGraphDef.getVersions)
        originalGraphDef.getNodeList.asScala
            .filter(n => Graph.shouldIncludeNode(n.getName, exportScope))
            .foreach(n => graphDefBuilder.addNode(Graph.processNodeDef(n, exportScope, unboundInputs, clearDevices)))
        graphDefBuilder.build()
      } else {
        originalGraphDef
      }
    }
    if (exportScope != null && unboundInputsCollectionKey != null) {
      // It's possible that not all the inputs are in the export scope. If we would like such information included in
      // the exported graph meta-information, we add them to a special collection.
      clearCollection(unboundInputsCollectionKey)
      unboundInputs.foreach(addToCollection(_, unboundInputsCollectionKey))
    }

    // Create the 'MetaGraphDef' object.
    val metaGraphDefBuilder = MetaGraphDef.newBuilder()
    metaGraphDefBuilder.setGraphDef(graphDef)

    // Add the meta information.
    val metaInfoDefBuilder = if (metaInfoDef == null) MetaInfoDef.newBuilder() else MetaInfoDef.newBuilder(metaInfoDef)
    metaInfoDefBuilder.setTensorflowVersion(NativeLibrary.version)
    metaGraphDefBuilder.mergeMetaInfoDef(metaInfoDefBuilder.build())

    // Add the saver information.
    if (saverDef != null)
      metaGraphDefBuilder.mergeSaverDef(saverDef)

    // Add the collections.
    if (collections != null)
      collections.foreach(key => addCollectionDefToMetaGraphDefBuilder(metaGraphDefBuilder, key, exportScope))

    metaGraphDefBuilder.build()
  }

  /** Adds a collection named `name` in a [[MetaGraphDef.Builder]].
    *
    * Note that if a collection with the same name already exists in the provided `metaGraphDefBuilder`, then that
    * collection will be overwritten by the new one being added.
    *
    * @param  metaGraphDefBuilder [[MetaGraphDef.Builder]] in which to add the collection.
    * @param  key                 Collection key.
    * @param  exportScope         Optional string specifying the name scope to remove. Only the ops within this name
    *                             scope will be included in the resulting ProtoBuf object and the export scope will be
    *                             stripped from their names to allow for easy import into new name scopes.
    * @return Updated [[MetaGraphDef.Builder]].
    */
  private[this] def addCollectionDefToMetaGraphDefBuilder[K](
      metaGraphDefBuilder: MetaGraphDef.Builder, key: Graph.Key[K],
      exportScope: String = null): MetaGraphDef.Builder = {
    metaGraphDefBuilder.putCollectionDef(key.name, key.createCollectionDef(getCollection(key), exportScope))
    metaGraphDefBuilder
  }

  /** Constructs and returns a [[GraphDef]] object, which is a serialized version of this graph.
    *
    * Note that the [[GraphDef]] does not contain any meta-information about the graph (such as collections information,
    * for example). For a serialized representation of the graph that contains such information, please refer to
    * [[Graph.toMetaGraphDef]].
    */
  override def toProto: GraphDef = toGraphDef

  /** Reference counter for this graph instance. */
  private[this] var referenceCount: Int = 0

  /** Returns a new reference to this graph. This method should be used by all classes whose corresponding native
    * objects (such as the `TF_Operation` object backing an [[Op]] instance) have a validity tied to that of the graph.
    *
    * That is because, the handles to those native objects are not valid after [[Graph.close]] has been invoked and the
    * references returned by this method help account for this behavior.
    */
  private[api] def reference: Reference = new Reference()

  /** Helper class for keeping track of references to this graph.
    *
    * Related native objects (such as the `TF_Operation` object backing an [[Op]] instance) have a validity tied to that
    * of the graph. The handles to those native objects are not valid after [[Graph.close]] has been invoked.
    *
    * Instances of the `Reference` class should be used to ensure the graph has not been closed while dependent handles
    * are in use. */
  private[api] final class Reference private[Graph]() extends Closeable {
    val graph: Graph = Graph.this

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

    def nativeHandle: Long = {
      NativeHandleLock.synchronized {
        if (Graph.this.nativeHandle != 0)
          Graph.this.nativeHandle
        else
          0
      }
    }
  }

  /** Releases the native resources associated with this graph instance.
    *
    * This method blocks until there are no active [[Session]] (or other) instances referring to this graph instance. A
    * graph is not usable after this method returns.
    */
  override def close(): Unit = NativeHandleLock.synchronized {
    defaultSession.close()
    if (nativeHandle != 0) {
      while (referenceCount > 0) {
        try {
          NativeHandleLock.wait()
        } catch {
          case _: InterruptedException =>
            Thread.currentThread().interrupt()
            // TODO: Possible leak of the graph in this case?
            return
        }
      }
      cleanupFunctions.foreach(_ ())
      NativeGraph.delete(nativeHandle)
      nativeHandle = 0
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
  /** Constructs and returns an empty new graph. */
  def apply(): Graph = new Graph(nativeHandle = NativeGraph.allocate())

  /** Imports a graph from the provided serialized graph object.
    *
    * @param  graphDef    Serialized representation of the graph that will be imported.
    * @param  importScope Optional prefix that will be prepended to all node names in the graph that is being imported
    *                     to this graph.
    * @return Constructed [[Graph]] object.
    */
  def fromGraphDef(graphDef: GraphDef, importScope: String = null): Graph = {
    val graph = Graph()
    graph.importGraphDef(graphDef, importScope)
    graph
  }

  /** Imports a graph and its meta-information from the provided serialized graph meta-information object.
    *
    * This function takes a [[MetaGraphDef]] protocol buffer as input and it adds all the nodes from its `graph_def`
    * field to a new graph. It also recreates the desired collections stored in that protocol buffer.
    *
    * In combination with [[Graph.toMetaGraphDef]], this function can be used to:
    *   - Serialize a graph along with other objects stored in its collections, into a [[MetaGraphDef]].
    *   - Restart training from saved graphs and checkpoints.
    *   - Run inference from saved graphs and checkpoints.
    *
    * @param  metaGraphDef                Serialized representation of the graph and its meta-information, that will be
    *                                     imported into the new graph.
    * @param  importScope                 Optional prefix that will be prepended to all node names in the graph that is
    *                                     being imported to the new graph.
    * @param  clearDevices                Boolean value indicating whether to clear the device information from the
    *                                     returned node definition.
    * @param  unboundInputsCollectionKey  Collection key for looking up unbound inputs.
    * @param  restoreCollectionsPredicate Function that takes as input a graph collection key and returns a boolean
    *                                     value indicating whether or not to load that collection. Note that the
    *                                     collection specified by `unboundInputsCollectionKey` is never loaded.
    *                                     Defaults to a function that returns `true` for all inputs.
    * @return Constructed [[Graph]] object.
    */
  def fromMetaGraphDef(
      metaGraphDef: MetaGraphDef, importScope: String = null, clearDevices: Boolean = false,
      unboundInputsCollectionKey: Graph.Key[String] = Graph.Keys.UNBOUND_INPUTS,
      restoreCollectionsPredicate: Graph.Key[_] => Boolean = _ => true): Graph = {
    val graph = Graph()
    graph.importMetaGraphDef(
      metaGraphDef, importScope, clearDevices = clearDevices, unboundInputsCollectionKey = unboundInputsCollectionKey,
      restoreCollectionsPredicate = restoreCollectionsPredicate)
    graph
  }

  /** Imports a graph from the provided serialized graph object.
    *
    * @param  graphDef    Serialized representation of the graph that will be imported.
    * @param  importScope Optional prefix that will be prepended to all node names in the graph that is being imported
    *                     to this graph.
    * @return Constructed [[Graph]] object.
    */
  def fromProto(graphDef: GraphDef, importScope: String = null): Graph = fromGraphDef(graphDef, importScope)

  //region MetaGraphDef Helpers

  private[this] val nodeDefNamePrefixRegex: Regex = "^\\^+".r
  private[this] val nodeDefRenameRegex    : Regex = "([\\^]|^)(.*)".r

  /** Returns `true` if a node should be included.
    *
    * @param  name        Node name.
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Boolean value indicating whether the node with the provided name should be included.
    */
  private def shouldIncludeNode(name: String, exportScope: String = null): Boolean = {
    name.startsWith(META_GRAPH_UNBOUND_INPUT_PREFIX) || exportScope == null || name.startsWith(exportScope)
  }

  /** Processes a node definition according the provided arguments and returns a new node definition.
    *
    * @param nodeDef       Node definition to process.
    * @param exportScope   Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                      be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                      names to allow for easy import into new name scopes.
    * @param unboundInputs Set containing unbound input names if they exist.
    * @param clearDevices  Boolean value indicating whether to clear the device information from the returned node
    *                      definition.
    * @return New processed node definition.
    */
  private def processNodeDef(
      nodeDef: NodeDef, exportScope: String = null, unboundInputs: mutable.Set[String] = mutable.Set.empty,
      clearDevices: Boolean = false): NodeDef = {
    val nodeDefBuilder = NodeDef.newBuilder(nodeDef)
    nodeDefBuilder.setName(Op.stripNameScope(exportScope, nodeDef.getName))
    val numberOfInputs = nodeDef.getInputCount
    var inputIndex = 0
    while (inputIndex < numberOfInputs) {
      val input = nodeDef.getInput(inputIndex)
      if (exportScope != null && nodeDefNamePrefixRegex.pattern.matcher(input).replaceAll("").startsWith(exportScope)) {
        // Add a prefix to the unbound name so that they are easily identifiable.
        val newInput = nodeDefRenameRegex.pattern.matcher(input).replaceFirst(s"$$1$META_GRAPH_UNBOUND_INPUT_PREFIX$$2")
        nodeDefBuilder.setInput(inputIndex, newInput)
        unboundInputs += newInput
      } else {
        nodeDefBuilder.setInput(inputIndex, Op.stripNameScope(exportScope, input))
      }
      inputIndex += 1
    }
    val attributes = nodeDef.getAttrMap.asScala
    for ((name, value) <- attributes) {
      if (name == "_class") {
        val values = value.getList.getSList.asScala
            .filter(exportScope == null || _.toStringUtf8.split("@")(1).startsWith(exportScope))
            .map(v => ByteString.copyFromUtf8(Op.stripNameScope(exportScope, v.toStringUtf8)))
        nodeDefBuilder.putAttr(
          name, AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addAllS(values.asJava)).build())
      } else {
        nodeDefBuilder.putAttr(name, AttrValue.newBuilder(value).build())
      }
    }
    if (clearDevices)
      nodeDefBuilder.setDevice("")
    nodeDefBuilder.build()
  }

  /** Copies a graph and its meta-information from `fromGraph` to `toGraph`, according to the provided scopes.
    *
    * @param  fromGraph From/source graph.
    * @param  toGraph   To/destination graph.
    * @param  fromScope From/source name scope. Only ops within this name scope are copied.
    * @param  toScope   To/destination name scope. The copied ops are placed under this name scope in `toGraph`.
    */
  def copyMetaGraph(fromGraph: Graph, toGraph: Graph, fromScope: String, toScope: String): Unit = {
    if (fromGraph == toGraph && fromScope == toScope)
      throw new IllegalArgumentException(
        "The 'fromScope' and the 'toScope' must be different when copying within the same graph.")
    toGraph.importMetaGraphDef(fromGraph.toMetaGraphDef(exportScope = fromScope), importScope = toScope)
  }

  //endregion MetaGraphDef Helpers

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

  /** Key to a graph collection. */
  sealed trait Key[K] {
    /** Name of this collection key. */
    def name: String

    /** Creates a [[CollectionDef]] from the provided set of values.
      *
      * @param  values      Values to serialize in the [[CollectionDef]].
      * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope
      *                     will be included in the resulting ProtoBuf object and the export scope will be stripped from
      *                     their names to allow for easy import into new name scopes.
      * @return Serialized `values` in a new [[CollectionDef]].
      */
    def createCollectionDef(values: Set[K], exportScope: String = null): CollectionDef

    /** Parses a [[CollectionDef]] and adds the collection values to the corresponding collection in the graph.
      *
      * @param  collectionDef [[CollectionDef]] being parsed.
      * @param  graph         Graph in which the collection should be added.
      * @param  importScope   Optional prefix that will be prepended to all node names in the graph that is being
      *                       imported to this graph.
      */
    def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit

    Keys.registry += name -> this
  }

  /** Contains standard names to use for graph collections.
    *
    * The standard library uses various well-known names to collect and retrieve values associated with a graph. For
    * example, the optimizers default to optimizing the variables collected under `Graph.Keys.TRAINABLE_VARIABLES` if
    * none is specified, but it is also possible to pass an explicit list of variables.
    *
    * Note: Whenever a new key is added, appropriate edits need to be made to the [[Keys.fromName]] function.
    */
  object Keys {
    private[api] val registry = mutable.Map.empty[String, Key[_]]

    private[api] def fromName(name: String): Key[_] = {
      registry.getOrElse(name, throw new IllegalArgumentException(s"Cannot find graph collection key named '$name'."))
    }

    /** Key for collections of strings. */
    trait StringCollectionKey extends Key[String] {
      override def createCollectionDef(values: Set[String], exportScope: String = null): CollectionDef = {
        val bytesListBuilder = BytesList.newBuilder()
        values.foreach(s => bytesListBuilder.addValue(ByteString.copyFromUtf8(s)))
        CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 2)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a byte list.")
        collectionDef.getBytesList.getValueList.asScala.foreach(s => graph.addToCollection(s.toStringUtf8, this))
      }
    }

    /** Key for collections of integers. */
    trait IntCollectionKey extends Key[Int] {
      override def createCollectionDef(values: Set[Int], exportScope: String = null): CollectionDef = {
        val int64ListBuilder = Int64List.newBuilder()
        values.foreach(v => int64ListBuilder.addValue(v))
        CollectionDef.newBuilder().setInt64List(int64ListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 3)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as an INT64 list.")
        collectionDef.getInt64List.getValueList.asScala.foreach(v => graph.addToCollection(v.toInt, this))
      }
    }

    /** Key for collections of ops. */
    trait OpCollectionKey extends Key[Op] {
      override def createCollectionDef(values: Set[Op], exportScope: String = null): CollectionDef = {
        val nodeListBuilder = NodeList.newBuilder()
        values.asInstanceOf[Set[Op]]
            .filter(o => Graph.shouldIncludeNode(o.name, exportScope))
            .filter(o => exportScope == null || o.name.startsWith(exportScope)).foreach(o => {
          nodeListBuilder.addValue(Op.stripNameScope(exportScope, o.name))
        })
        CollectionDef.newBuilder().setNodeList(nodeListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 1)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a node list.")
        collectionDef.getNodeList.getValueList.asScala
            .foreach(o => graph.addToCollection(graph.getOpByName(Op.prependNameScope(importScope, o)), this))
      }
    }

    /** Key for collections of op outputs. */
    trait OutputCollectionKey extends Key[Output] {
      override def createCollectionDef(values: Set[Output], exportScope: String): CollectionDef = {
        val nodeListBuilder = NodeList.newBuilder()
        values.asInstanceOf[Set[Output]]
            .filter(o => Graph.shouldIncludeNode(o.name, exportScope))
            .filter(o => exportScope == null || o.name.startsWith(exportScope)).foreach(o => {
          nodeListBuilder.addValue(Op.stripNameScope(exportScope, o.name))
        })
        CollectionDef.newBuilder().setNodeList(nodeListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 1)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a node list.")
        collectionDef.getNodeList.getValueList.asScala
            .foreach(o => graph.addToCollection(graph.getOutputByName(Op.prependNameScope(importScope, o)), this))
      }
    }

    /** Key for collections of variables. */
    trait VariableCollectionKey extends Key[Variable] {
      override def createCollectionDef(values: Set[Variable], exportScope: String = null): CollectionDef = {
        val bytesListBuilder = BytesList.newBuilder()
        values
            .map(_.toProto(exportScope))
            .filter(_ != null)
            .foreach(s => bytesListBuilder.addValue(s.toByteString))
        CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 1)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a byte list.")
        collectionDef.getBytesList.getValueList.asScala
            .foreach(v => graph.addToCollection(Variable.fromProto(VariableDef.parseFrom(v), importScope), this))
      }
    }

    /** Key for collections of savers. */
    trait SaverCollectionKey extends Key[Saver] {
      override def createCollectionDef(values: Set[Saver], exportScope: String = null): CollectionDef = {
        val bytesListBuilder = BytesList.newBuilder()
        values
            .map(_.toProto(exportScope))
            .filter(_ != null)
            .foreach(s => bytesListBuilder.addValue(s.toByteString))
        CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 1)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a byte list.")
        collectionDef.getBytesList.getValueList.asScala
            .foreach(s => graph.addToCollection(Saver.fromProto(SaverDef.parseFrom(s), importScope), this))
      }
    }

    /** Key for collections of resources. */
    trait ResourceCollectionKey extends Key[Resource] {
      override def createCollectionDef(values: Set[Resource], exportScope: String): CollectionDef = {
        val nodeListBuilder = NodeList.newBuilder()
        values.foreach(r => {
          if (Graph.shouldIncludeNode(r.handle.name) &&
              Graph.shouldIncludeNode(r.initializeOp.name) &&
              Graph.shouldIncludeNode(r.isInitialized.name)) {
            nodeListBuilder.addValue(Op.stripNameScope(exportScope, r.handle.name))
            nodeListBuilder.addValue(Op.stripNameScope(exportScope, r.initializeOp.name))
            nodeListBuilder.addValue(Op.stripNameScope(exportScope, r.isInitialized.name))
          }
        })
        CollectionDef.newBuilder().setNodeList(nodeListBuilder.build()).build()
      }

      override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
        val kind = collectionDef.getKindCase.getNumber
        if (kind != 1)
          throw new IllegalArgumentException(s"The '$name' collection should be stored as a node list.")
        collectionDef.getNodeList.getValueList.asScala.grouped(3)
            .foreach(r => {
              graph.addToCollection(
                Resource(
                  graph.getOutputByName(Op.prependNameScope(importScope, r(0))),
                  graph.getOpByName(Op.prependNameScope(importScope, r(1))),
                  graph.getOutputByName(Op.prependNameScope(importScope, r(2)))), this)
            })
      }
    }

    /** Key to collect the graph random seed values. The seed values collection should have only one element
      * representing the graph random seed value. */
    object RANDOM_SEEDS extends IntCollectionKey {
      override def name: String = "random_seeds"
    }

    /** Key to collect the default collection of `Variable` objects, shared across distributed environment (model
      * variables are subset of these). Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`, and
      * all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`. */
    object GLOBAL_VARIABLES extends VariableCollectionKey {
      override def name: String = "variables"
    }

    /** Key to collect the subset of `Variable` objects that are local to each machine. Usually used for temporary
      * variables, like counters. */
    object LOCAL_VARIABLES extends VariableCollectionKey {
      override def name: String = "local_variables"
    }

    /** Key to collect the subset of `Variable` objects that are used in models for inference (feed forward).
      * TODO: Note: use `tf.contrib.framework.model_variable` to add to this collection. */
    object MODEL_VARIABLES extends VariableCollectionKey {
      override def name: String = "model_variables"
    }

    /** Key to collect the subset of `Variable` objects that will be trained using an optimizer. */
    object TRAINABLE_VARIABLES extends VariableCollectionKey {
      override def name: String = "trainable_variables"
    }

    /** Key to collect the summary `Output` objects that have been created in the graph. */
    object SUMMARIES extends OutputCollectionKey {
      override def name: String = "summaries"
    }

    // /** Key to collect the `QueueRunner` objects that are used to produce inputs for a computation. */
    // object QUEUE_RUNNERS extends Key {override def name: String = "queue_runners"}

    // /** Key to collect table initializer objects. */
    // object TABLE_INITIALIZERS extends Key {override def name: String = "table_initializer"}

    // /** Key to collect asset filepaths. An asset represents an external resource like a vocabulary file. */
    // object ASSET_FILEPATHS extends Key {override def name: String = "asset_filepaths"}

    /** Key to collect the subset of `Variable` objects that will also keep moving averages. */
    object MOVING_AVERAGE_VARIABLES extends VariableCollectionKey {
      override def name: String = "moving_average_variables"
    }

    /** Key to collect regularization losses at graph construction. */
    object REGULARIZATION_LOSSES extends OutputCollectionKey {
      override def name: String = "regularization_losses"
    }

    // /** Key to collect concatenated sharded variables. */
    // object CONCATENATED_VARIABLES extends Key {override def name: String = "concatenated_variables"}

     /** Key to collect savers. */
     object SAVERS extends SaverCollectionKey {
       override def name: String = "savers"
     }

    /** Key to collect weights. */
    object WEIGHTS extends VariableCollectionKey {
      override def name: String = "weights"
    }

    /** Key to collect biases. */
    object BIASES extends VariableCollectionKey {
      override def name: String = "biases"
    }

    /** Key to collect activations. */
    object ACTIVATIONS extends OpCollectionKey {
      override def name: String = "activations"
    }

    /** Key to collect update ops. */
    object UPDATE_OPS extends OpCollectionKey {
      override def name: String = "update_ops"
    }

    /** Key to collect losses. */
    object LOSSES extends OutputCollectionKey {
      override def name: String = "losses"
    }

    // /** Key to collect saveable objects used for checkpoints. */
    // object SAVEABLE_OBJECTS extends Key {override def name: String = "saveable_objects"}

    /** Key to collect all shared resources used by the graph which need to be initialized once per cluster. */
    object SHARED_RESOURCES extends ResourceCollectionKey {
      override def name: String = "resources"
    }

    /** Key to collect all local resources used in this graph which need to be initialized once per session. */
    object LOCAL_RESOURCES extends ResourceCollectionKey {
      override def name: String = "local_resources"
    }

    /** Key to collect all trainable resource-style variables. */
    object TRAINABLE_RESOURCE_VARIABLES extends VariableCollectionKey {
      override def name: String = "trainable_resource_variables"
    }

    // Keys to indicate various ops.

    object INIT_OP extends OpCollectionKey {
      override def name: String = "init_op"
    }

    object LOCAL_INIT_OP extends OpCollectionKey {
      override def name: String = "local_init_op"
    }

    object READY_OP extends OpCollectionKey {
      override def name: String = "ready_op"
    }

    object READY_FOR_LOCAL_INIT_OP extends OpCollectionKey {
      override def name: String = "ready_for_local_init_op"
    }

    object SUMMARY_OP extends OpCollectionKey {
      override def name: String = "summary_op"
    }

    object GLOBAL_STEP extends VariableCollectionKey {
      override def name: String = "global_step"
    }

    object EVAL_STEP extends VariableCollectionKey {
      override def name: String = "eval_step"
    }

    object TRAIN_OP extends OpCollectionKey {
      override def name: String = "train_op"
    }

    // Keys for control flow management.
    // object COND_CONTEXT extends Key {override def name: String = "cond_context"}
    // object WHILE_CONTEXT extends Key {override def name: String = "while_context"}

    /** Key to collect streaming model ports. */
    object STREAMING_MODEL_PORTS extends VariableCollectionKey {
      override def name: String = "streaming_model_ports"
    }

    /** Key to collect the unbound inputs when serializing/deserializing graphs. */
    object UNBOUND_INPUTS extends StringCollectionKey {
      override def name: String = "unbound_inputs"
    }
  }
}
