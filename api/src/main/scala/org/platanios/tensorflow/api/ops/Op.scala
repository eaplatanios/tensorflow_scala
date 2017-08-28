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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.{DeviceSpecification, Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.{CreateNewOnly, VariableScope, VariableStore}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.utilities.using
import org.platanios.tensorflow.jni.{Op => NativeOp, Tensor => NativeTensor}

import java.nio.charset.Charset

import scala.collection.mutable
import scala.util.{DynamicVariable, Try}

/** Represents a graph node, or as we shall call it, an operation, that performs computation on tensors.
  *
  * TODO: Add Op.run method and Op.Output.eval method.
  *
  * An `Op` is a symbolic representation of the computation it performs. It is a node in a TensorFlow [[Graph]] that
  * takes zero or more `Op.Output` objects as input, and produces zero or more `Op.Output` objects as output. `Op`
  * objects are constructed by calling op creation functions, such as [[Basic.constant]] or [[Math.matmul]].
  *
  * For example, `val c = MathOps.matmul(a, b)` creates an `Op` of type `"MatMul"` that takes `Op.Output`s `a` and
  * `b` as input, and produces `Op.Output` `c` as output.
  *
  * @note The `Op.Input` class is simply a wrapper around an `Op` meant to represent one of its inputs. Actual op inputs
  *       have type `Op.Output` since they represent outputs of other ops. Currently, `Op.Input` is only useful for
  *       representing consumers of an `Op`'s outputs.
  *
  *       After the graph has been launched in a [[Session]], an `Op` can be executed by using `Session.run`.
  *
  *       TODO: Add `Op.run` use example, once that is supported.
  * @author Emmanouil Antonios Platanios
  */
final case class Op private (graph: Graph, private[api] val nativeHandle: Long) {
  // Update the ops cache of the graph with the current op.
  graph.opsCache.update(nativeHandle, this)

  /** Name of the op. */
  lazy val name: String = using(graph.reference) { _ => NativeOp.name(nativeHandle) }

  /** Type of the op (i.e., the name of the computation performed by the operation). */
  lazy val opType: String = using(graph.reference) { _ => NativeOp.opType(nativeHandle) }

  /** Device in which the op tensors are stored and where all computations for this op are performed. */
  lazy val device: String = using(graph.reference) { _ =>
    val nativeDevice = NativeOp.device(nativeHandle)
    if (nativeDevice == null)
      ""
    else
      nativeDevice
  }

  /** Colocation ops for this op (i.e., ops guaranteed to be placed on the same device). */
  lazy val colocationOps: Set[Op] = using(graph.reference) { _ =>
    Try(NativeOp.getAttrStringList(nativeHandle, COLOCATION_OPS_ATTRIBUTE_NAME))
        .map(_.toSet[String]
                 .filter(_.startsWith(COLOCATION_OPS_ATTRIBUTE_PREFIX))
                 .map(opName => graph.findOp(opName.substring(COLOCATION_OPS_ATTRIBUTE_PREFIX.length)).get))
        .getOrElse(Set.empty[Op])
  }

  /** Number of inputs to this op (i.e., number of tensors fed as input to this op). */
  lazy val numInputs: Int = using(graph.reference) { _ => NativeOp.numInputs(nativeHandle) }

  /** Inputs of this op. Note that these inputs are outputs of other ops and thus have type [[Output]]. */
  lazy val inputs: Array[Output] = (0 until numInputs).map(index => using(graph.reference) { _ =>
    val jniOutput = NativeOp.input(nativeHandle, index)
    val op = graph.opsCache.getOrElseUpdate(
      jniOutput.opHandle,
      Op(graph, jniOutput.opHandle))
    op.outputs(jniOutput.outputIndex)
  }).toArray

  /** Number of control inputs to this op. These are ops that are guaranteed to finish executing before this op starts
    * executing). */
  lazy val numControlInputs: Int = using(graph.reference) { _ => NativeOp.numControlInputs(nativeHandle) }

  /** Control inputs of this op. These are ops that are guaranteed to finish executing before this op starts
    * executing). */
  lazy val controlInputs: Set[Op] = {
    val controlInputHandles = using(graph.reference) { _ => NativeOp.controlInputs(nativeHandle) }
    controlInputHandles.map(handle => graph.opsCache.getOrElseUpdate(handle, Op(graph, handle))).toSet
  }

  /** Number of tensors produced by this operation. */
  lazy val numOutputs: Int = using(graph.reference) { _ => NativeOp.numOutputs(nativeHandle) }

  /** Outputs of this op. */
  lazy val outputs: Array[Output] = (0 until numOutputs).map(i => Output(op = this, index = i)).toArray

  /** Gets the (current) number of control outputs of this op. These are ops that are guaranteed to start executing
    * after this op finishes executing.
    *
    * @note A concurrent modification of the graph can change the number of control outputs of this op.
    * @return Current number of control outputs of this op.
    */
  def numControlOutputs: Int = using(graph.reference) { _ => NativeOp.numControlOutputs(nativeHandle) }

  /** Gets the (current) control outputs of this op. These are ops that are guaranteed to start executing after this op
    * finishes executing.
    *
    * @note A concurrent modification of the graph can change the number of control outputs of this op.
    * @return Current control outputs of this op.
    */
  def controlOutputs: Set[Op] = {
    val controlOutputHandles = using(graph.reference) { _ => NativeOp.controlOutputs(nativeHandle) }
    controlOutputHandles.map(handle => graph.opsCache.getOrElseUpdate(handle, Op(graph, handle))).toSet
  }

  /** Gets the value of a string-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no string attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringAttribute(name: String): String = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrString(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no string attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a string-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no string array attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringArrayAttribute(name: String): Array[String] = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrStringList(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no string array attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a long-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no long attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def longAttribute(name: String): Long = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrInt(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no long attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a float-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no float attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def floatAttribute(name: String): Float = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrFloat(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no float attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a boolean-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no boolean attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def booleanAttribute(name: String): Boolean = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrBool(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no boolean attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a data type-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no data type attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def dataTypeAttribute(name: String): DataType = using(graph.reference) { _ =>
    try {
      DataType.fromCValue(NativeOp.getAttrType(nativeHandle, name))
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no data type attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a tensor-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no tensor attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def tensorAttribute(name: String): Tensor = using(graph.reference) { _ =>
    try {
      Tensor.fromHostNativeHandle(NativeOp.getAttrTensor(nativeHandle, name))
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no tensor attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a shape-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If no shape attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def shapeAttribute(name: String): Shape = using(graph.reference) { _ =>
    try {
      Shape.fromSeq(NativeOp.getAttrShape(nativeHandle, name).map(_.toInt))
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no shape attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  override def toString: String = name

  // TODO: [OP] Better implementations for equals and hashCode.

  override def equals(that: Any): Boolean = that match {
    case that: Op => this.graph == that.graph && this.nativeHandle == that.nativeHandle
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    result = prime * result + graph.hashCode
    result = prime * result + nativeHandle.hashCode
    result
  }
}

final case class OpSpecification(name: String, opType: String)

private[api] final case class OpCreationContext(
    graph: Graph = Graph(), nameScope: String = "", variableScope: VariableScope = VariableScope(reuse = CreateNewOnly),
    device: OpSpecification => String = _ => "", colocationOps: Set[Op] = Set.empty,
    controlDependencies: Set[Op] = Set.empty, attributes: Map[String, Any] = Map.empty, container: String = "") // TODO: !!! Use containers.

object Op {
  private[ops] trait API {
    type Op = ops.Op
    val Op: ops.Op.type = ops.Op

    type OpCreationContext = ops.OpCreationContext
    type OpSpecification = ops.OpSpecification

    def currentGraph: Graph = Op.currentGraph
    def currentNameScope: String = Op.currentNameScope
    def currentVariableScope: VariableScope = Op.currentVariableScope
    def currentVariableStore: VariableStore = Op.currentVariableStore
    def currentDevice: OpSpecification => String = Op.currentDevice
    def currentColocationOps: Set[Op] = Op.currentColocationOps
    def currentControlDependencies: Set[Op] = Op.currentControlDependencies
    def currentAttributes: Map[String, Any] = Op.currentAttributes
    def currentContainer: String = Op.currentContainer

    def currentGraphRandomSeed(opSeed: Option[Int] = None): (Option[Int], Option[Int]) = {
      Op.currentGraphRandomSeed(opSeed)
    }

    def setCurrentGraphRandomSeed(value: Int): Unit = Op.setCurrentGraphRandomSeed(value)

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: OpSpecification => String = _ => "",
        colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
        container: String = null)(block: => R): R = {
      Op.createWith(graph, nameScope, device, colocationOps, controlDependencies, attributes, container)(block)
    }

    def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R): R = {
      Op.createWithNameScope(nameScope, values)(block)
    }

    def colocateWith[R](colocationOps: Set[Op], ignoreExisting: Boolean = false)(block: => R): R = {
      Op.colocateWith(colocationOps, ignoreExisting)(block)
    }

    def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): Op = {
      Op.currentGraph.globalVariablesInitializer(name)
    }

    def localVariablesInitializer(name: String = "LocalVariablesInitializer"): Op = {
      Op.currentGraph.localVariablesInitializer(name)
    }

    def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): Op = {
      Op.currentGraph.modelVariablesInitializer(name)
    }

    def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): Op = {
      Op.currentGraph.trainableVariablesInitializer(name)
    }
  }

  /** Returns the graph of the current op creation context. */
  private[api] def currentGraph(implicit context: DynamicVariable[OpCreationContext]): Graph = context.value.graph

  /** Returns the name scope of the current op creation context. */
  private[api] def currentNameScope(implicit context: DynamicVariable[OpCreationContext]): String = {
    if (context.value.nameScope == "")
      ""
    else
      s"${context.value.nameScope}/"
  }

  /** Returns the variable scope of the current op creation context. */
  private[api] def currentVariableScope(implicit context: DynamicVariable[OpCreationContext]): VariableScope = {
    context.value.variableScope
  }

  /** Returns the variable store of the current op creation context. */
  private[api] def currentVariableStore(implicit context: DynamicVariable[OpCreationContext]): VariableStore = {
    context.value.graph.variableStore
  }

  /** Returns the device of the current op creation context. */
  private[api] def currentDevice(implicit context: DynamicVariable[OpCreationContext]): OpSpecification => String = {
    context.value.device
  }

  /** Returns the colocation ops of the current op creation context. */
  private[api] def currentColocationOps(implicit context: DynamicVariable[OpCreationContext]): Set[Op] = {
    context.value.colocationOps
  }

  /** Returns the control dependencies of the current op creation context. */
  private[api] def currentControlDependencies(implicit context: DynamicVariable[OpCreationContext]): Set[Op] = {
    context.value.controlDependencies
  }

  /** Returns the attributes of the current op creation context. */
  private[api] def currentAttributes(implicit context: DynamicVariable[OpCreationContext]): Map[String, Any] = {
    context.value.attributes
  }

  /** Returns the container of the current op creation context. */
  private[api] def currentContainer(implicit context: DynamicVariable[OpCreationContext]): String = {
    context.value.container
  }

  /** Returns the local seeds an operation should use given an op-specific random seed.
    *
    * Given the op-specific seed, `opSeed`, this helper function returns two seeds derived from graph-level and op-level
    * seeds. Many random operations internally use the two seeds to allow the user to change the seed globally for a
    * graph, or only for specific operations.
    *
    * For details on how the graph-level seed interacts with op seeds, see [[setCurrentGraphRandomSeed]].
    *
    * @param  opSeed Op-specific seed value.
    * @return Tuple of two numbers that should be used for the local seed of this operation.
    */
  private[api] def currentGraphRandomSeed(opSeed: Option[Int] = None): (Option[Int], Option[Int]) = {
    (currentGraph.randomSeed, opSeed) match {
      // Avoid (0, 0) as the C++ ops interpret it as non-determinism, which would be unexpected.
      case (Some(0), Some(0)) => (Some(0), Some(Int.MaxValue))
      case (Some(g), Some(o)) => (Some(g), Some(o))
      case (Some(g), None) => (Some(g), Some(currentGraph.ops.length))
      case (None, Some(o)) => (Some(DEFAULT_GRAPH_RANDOM_SEED), Some(o))
      case (None, None) => (None, None)
    }
  }

  /** Sets the graph-level random seed.
    *
    * Operations that rely on a random seed actually derive it from two seeds: the graph-level and the operation-level
    * seeds. This function sets the graph-level seed.
    *
    * Its interactions with operation-level seeds are as follows:
    *   1. If neither the graph-level nor the operation-level seed is set, a random seed is used for this op.
    *   2. If the graph-level seed is set, but the operation-level seed is not, the system deterministically picks an
    *      operation-level seed in conjunction with the graph-level seed so that it gets a unique random sequence.
    *   3. If the graph-level seed is not set, but the operation-level seed is set, a default graph-level seed and the
    *      specified operation-level seed are used to determine the random sequence.
    *   4. If both the graph-level and the operation-level seed are set, then both seeds are used in conjunction to
    *      determine the random sequence.
    *
    * To generate different sequences across sessions, set neither the graph-level nor the op-level seeds.
    *
    * @param  value Value to set the graph-level random seed to.
    */
  private[api] def setCurrentGraphRandomSeed(value: Int): Unit = currentGraph.setRandomSeed(value)

  /** Creates a context that can be used for creating ops according to the provided options.
    *
    * = General Information =
    *
    * During graph creation, a context is maintained that includes:
    *   - The current graph in which new ops are placed.
    *   - The current name scope used for naming these new ops.
    *   - A device function, used to decide in which device (e.g., CPU) the new ops should be placed and executed.
    *   - A set of colocation ops for the newly constructed ops. This means that the newly created ops will be placed on
    * the same device as these colocation ops.
    *   - A set of ops defining control dependencies for the newly constructed ops. This means that the newly
    * constructed ops are constrained to only execute after the provided set of ops has finished executing.
    *   - A map from op attribute names to values for the newly constructed ops. These attributes will be applied to all
    * newly constructed ops.
    *   - A container name for the newly constructed resource ops. All newly constructed resource ops will be placed in
    * the provided container.
    *
    * Note that all arguments of this function are optional. If they are not provided, then the corresponding option in
    * current op creation context is left unchanged.
    *
    * Care must be taken if concurrency is used while creating the graph because the op creation context is wrapped
    * inside a [[scala.util.DynamicVariable]]. More information on this general issue can be found at
    * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
    *
    * = Argument Specifics =
    *
    * == Graph ==
    *
    * When `createWith(...)` is used with a graph, then all ops created within its code block will be placed in the
    * provided graph.
    *
    * For example:
    * {{{
    *   val g = Graph()
    *   createWith(graph = g) {
    *     val c = constant(5.0)
    *     assert(c.graph == g)
    *   }
    * }}}
    *
    * == Name Scope ==
    *
    * When `createWith(...)` is used with a name scope, the provided name scope is appended to the context name scope,
    * generating a new op creation context. This new context is used for all ops created within the code block provided
    * in the `createWith(...)` function. The `nameScope` argument will be interpreted as follows:
    *   - A string not ending with `"/"` will create a new name scope, in which `nameScope` is appended to the prefix of
    *     all operations created in the provided code block. If `nameScope` has been used before, it will be made unique
    *     by calling `uniqueName(graph = context.graph, name = nameScope)`.
    *   - A string ending with `"/"` will be treated as an "absolute" name scope, which makes it possible to re-enter
    *     existing scopes. Such absolute name scopes can be obtained by using the `currentNameScope` function, from
    *     within the appropriate context.
    *   - A value of `""` will reset the current name scope to the top-level (i.e., empty) name scope.
    *
    * This function checks the provided `nameScope` for validity by checking whether it matches: (i) the regular
    * expression `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*` if the current context name scope is empty (i.e., at the root), or
    * (ii) the regular expression `[A-Za-z0-9_.\\-/]*`, otherwise.
    *
    * For example:
    * {{{
    *   // No name scope used
    *   val c = constant(1.0, name = "C")
    *   assert(c.op.name == "C")
    *   val c1 = constant(2.0, name = "C_1")
    *   assert(c_1.op.name == "C_1")
    *
    *   // Create a name scope called "Nested"
    *   createWith(nameScope = "Nested") {
    *     val nameScope = currentNameScope
    *     val nestedC = constant(3.0, name = "C")
    *     assert(nestedC.op.name == "Nested/C")
    *
    *     // Create a nested name scope called "Inner"
    *     createWith(nameScope = "Inner") {
    *       val nestedInnerC = constant(4.0, name = "C")
    *       assert(nestedInnerC.op.name == "Nested/Inner/C")
    *     }
    *
    *     // Create a nested name scope called "Inner_1"
    *     createWith(nameScope = "Inner_1") {
    *       val nestedInner1C = constant(5.0, name = "C")
    *       assert(nestedInner1C.op.name == "Nested/Inner_1/C")
    *
    *       createWith(nameScope = nameScope) {
    *         val nestedC1 = constant(6.0, name = "C_1")
    *         assert(nestedC1.op.name == "Nested/C_1")
    *
    *         // Reset the name scope using ""
    *         createWith(nameScope = "") {
    *           val c2 = constant(7.0, name = "C_2")
    *           assert(c2.op.name == "C_2")
    *         }
    *       }
    *     }
    *   }
    * }}}
    *
    * == Device ==
    *
    * When `createWith(...)` is used with a device, the `device` argument needs to be a function taking an
    * [[OpSpecification]] as input and returning a string representation of the device where the corresponding op should
    * be placed. This function is invoked every time a new op is created within the provided code block. If the function
    * returns `null` for some op, then all subsequent invocations of `createWith(device = ...)` in the provided code
    * block will be ignored. Note that, if the device implicit conversion function is within scope, then a `String`
    * value (or `null`) can be used directly for the `device` field. In this case, the value provided will be used as
    * the device for all newly create ops in the provided code block. For information about the valid syntax of device
    * name strings, see the documentation in
    * [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
    *
    * Note that the device scope may be overridden by op wrappers or other library code. For example, a variable
    * assignment op must be colocated with the corresponding variable. Incompatible device scopes will be ignored.
    *
    * For example:
    * {{{
    *   // Specifying which device to use
    *   createWith(device = "/GPU:0") {
    *     // All ops constructed in this code block will be placed in GPU 0
    *     val gpu0C = constant(7.0)
    *     assert(gpu0C.device == "/device:GPU:0")
    *
    *     // Reset the device being used
    *     createWith(device = null) {
    *       // All ops constructed in this code block will have no assigned device
    *       val c = constant(8.0)
    *       assert(c.device == "")
    *     }
    *   }
    *
    *   // Using a device function
    *   def matmulOnGPU(opSpecification: OpSpecification): String = {
    *     if (opSpecification.opType == "MatMul")
    *       "/GPU:0"
    *     else
    *       "/CPU:0"
    *   }
    *
    *   createWith(device = matmulOnGPU) {
    *     // All ops of type "MatMul" constructed in this code block will be placed on GPU 0. All other operations will
    *     // be placed on CPU 0.
    *     val c = constant(9.0)
    *     assert(c.device == "/device:CPU:0")
    *     val m = matmul(c, constant(10.0))
    *     assert(m.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * == Colocation Ops ==
    *
    * When `createWith(...)` is used with a set of colocation ops, then all ops created within its code block will be
    * placed on the same device as the provided colocation ops. Note that if a set of colocation ops already exists in
    * the current op creation context (e.g., as the result of nesting multiple `createWith(colocationOps = ...)` calls),
    * then the new set of colocation ops will be the union of the two sets. If provided an empty colocation ops set,
    * then the new set of colocation ops will also be empty (i.e., it is being reset).
    *
    * Note that using a non-empty set of colocation ops resets any existing device constraints. In other words,
    * colocation ops override any other device placement specification.
    *
    * For example:
    * {{{
    *   val a = createWith(device = "/CPU:0")(constant(1.0))
    *   val b = createWith(device = "/GPU:0")(constant(1.0))
    *   assert(a.colocationOps === Set.empty[Op])
    *   assert(b.colocationOps === Set.empty[Op])
    *   val c = createWith(colocationOps = Set(a))(constant(1.0))
    *   assert(c.colocationOps === Set[Op](a))
    *   createWith(colocationOps = Set[Op](b)) {
    *     val d = constant(1.0)
    *     assert(d.colocationOps === Set[Op](b))
    *     createWith(colocationOps = Set[Op](a, d)) {
    *       val e = constant(1.0)
    *       assert(e.colocationOps === Set[Op](a, b, d))
    *       createWith(colocationOps = Set.empty[Op]) {
    *         val f = constant(1.0)
    *         assert(f.colocationOps === Set.empty[Op])
    *       }
    *     }
    *   }
    * }}}
    *
    * == Control Dependencies ==
    *
    * When `createWith(...)` is used with a set of control dependencies, then all ops created within its code block will
    * be dependent on the control dependency ops. This means that they will be guaranteed to execute only after all of
    * the control dependencies ops have finished executing. Note that if a set of control dependencies already exists in
    * the current op creation context (e.g., as the result of nesting multiple `createWith(controlDependencies = ...)`
    * calls), then the new set of control dependencies will be the union of the two sets. Furthermore, if an empty set
    * is provided, then the control dependencies are cleared, instead of taking the union with the current control
    * dependencies.
    *
    * For example:
    * {{{
    *   val a = constant(1.0)
    *   val b = constant(1.0)
    *   createWith(controlDependencies = Set(a)) {
    *     val c = constant(1.0)
    *     assert(c.controlInputs.toSet == Set(a))
    *     createWith(controlDependencies = Set(b, c)) {
    *       val d = constant(1.0)
    *       assert(d.controlInputs.toSet == Set(a, b, c))
    *       createWith(controlDependencies = Set()) {
    *         createWith(controlDependencies = Set(d)) {
    *           val e = constant(1.0)
    *           assert(e.controlInputs.toSet == Set(d))
    *         }
    *       }
    *     }
    *   }
    *   assert(a.controlOutputs.toSet == Set(c, d))
    *   assert(b.controlOutputs.toSet == Set(d))
    *   assert(c.controlOutputs.toSet == Set())
    *   assert(d.controlOutputs.toSet == Set(e))
    *   assert(e.controlOutputs.toSet == Set())
    * }}}
    *
    * Note that transitive dependencies are eliminated (e.g., if `a` depends on `b` and `c`, and `b` depends on `c`,
    * then the dependency of `a` on `c` is ignored) in order not to add redundant control dependencies to the graph.
    *
    * == Attributes ==
    *
    * When `createWith(...)` is used with a set of attributes, then all ops created within its code block will have
    * those attributes set to the provided values when constructed. Note that if a map from attribute names to values
    * already exists in the current op creation context, then the two maps are merged. If a name exists in both, then
    * the provided value overrides the existing one, otherwise, the union of the two maps is used. Note that if the
    * value for an attribute in the provided map is set to `null`, then that attribute name-value pair is completely
    * removed from the op creation context.
    *
    * For example:
    * {{{
    *   val a = constant(1.0)
    *   assert(a.stringAttribute("_a") == null)
    *   createWith(attributes = Map("_a" -> "foo")) {
    *     val b = constant(1.0)
    *     assert(b.stringAttribute("_a") == "foo")
    *     createWith(attributes = Map("_a" -> "bar")) {
    *       val c = constant(1.0)
    *       assert(c.stringAttribute("_a") == "bar")
    *       createWith(attributes = Map("_a" -> null)) {
    *         val d = constant(1.0)
    *         assert(d.stringAttribute("_a") == null)
    *       }
    *     }
    *   }
    * }}}
    *
    * == Container ==
    *
    * Stateful operations, such as variables and queues, can maintain their states on devices so that they can be shared
    * by multiple processes. A resource container is a string name under which these stateful operations are tracked.
    * These resources can be released or cleared with `Session.reset`. // TODO: [SESSION] Add that function reference.
    *
    * When `createWith(...)` is used with a container, then all resource ops created within its code block will be
    * placed in the provided container. A new value for the container always overrides the previous value, except if
    * `null`, meaning that the previous value is used. The default root container name is `""`.
    *
    * TODO: [VARIABLE] Add example when we have support for variables.
    *
    * == Combining Arguments ==
    *
    * Multiple arguments can be provided to change several aspects of the current op creation scope.
    *
    * For example:
    * {{{
    *   // Changing graph, name scope, and device to use for new ops.
    *   createWith(graph = g, nameScope = "Nested", device = "/GPU:0") {
    *     val c = constant(11.0, name = "C")
    *     assert(c.graph == g)
    *     assert(c.op.name == "Nested/C")
    *     assert(c.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * @param  graph               Graph to use as default for new ops.
    * @param  nameScope           Name scope to use.
    * @param  device              Device function to use.
    * @param  colocationOps       Colocation ops to use.
    * @param  controlDependencies Control dependencies to use.
    * @param  attributes          Attributes to use.
    * @param  container           Container to use for resources.
    * @param  block               Code block to run using the provided options.
    * @param  context             Current op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    * @throws IllegalNameException If the provided name scope does not pass the regular expression validity checks.
    */
  @throws[IllegalNameException]
  private[api] def createWith[R](
      graph: Graph = null, nameScope: String = null, device: OpSpecification => String = _ => "",
      colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
      container: String = null)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    // TODO: Move this to a separate scope class.
    // TODO: !!! The order of the updates matters here so let's make sure everything is fine.
    var updatedContext = context.value
    val newGraph: Graph = mergeGraph(graph, updatedContext)
    updatedContext = updatedContext.copy(graph = newGraph)
    val newNameScope: String = mergeNameScope(nameScope, updatedContext)
    updatedContext = updatedContext.copy(nameScope = newNameScope)
    val newDevice: OpSpecification => String = mergeDevice(device, updatedContext)
    updatedContext = updatedContext.copy(device = newDevice)
    val newColocationOps: Set[Op] = mergeColocationOps(colocationOps, updatedContext)
    updatedContext = updatedContext.copy(colocationOps = newColocationOps)
    val newControlDependencies: Set[Op] = mergeControlDependencies(controlDependencies, updatedContext)
    updatedContext = updatedContext.copy(controlDependencies = newControlDependencies)
    val newAttributes: Map[String, Any] = mergeAttributes(attributes, updatedContext)
    updatedContext = updatedContext.copy(attributes = newAttributes)
    val newContainer: String = mergeContainer(container, updatedContext)
    updatedContext = updatedContext.copy(container = newContainer)
    context.withValue(updatedContext)(block)
  }

  /** Creates a context that can be used for creating ops.
    *
    * This function validates that the provided `values` are all defined in the same graph, makes that the graph used
    * by the op creation context it defines, and also "pushes" the provided `nameScope` in the op creation context. More
    * details on the op creation context can be found in the documentation of the public API [[createWith]] function of
    * this library.
    *
    * @param  nameScope Name scope to use.
    * @param  values    Input values to obtain the default graph from.
    * @param  block     Code block to run using the provided options.
    * @param  context   Current op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    * @throws GraphMismatchException If any two of the values provided lie in different graphs.
    */
  @throws[GraphMismatchException]
  private[api] def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R)
      (implicit context: DynamicVariable[OpCreationContext]): R = {
    if (values.nonEmpty) {
      val newGraph: Graph = mergeGraph(getGraphFromInputs(values), context)
      val newNameScope: String = mergeNameScope(nameScope, context.copy(graph = newGraph))
      context.withValue(context.copy(graph = newGraph, nameScope = newNameScope))(block)
    } else {
      val newNameScope: String = mergeNameScope(nameScope, context)
      context.withValue(context.copy(nameScope = newNameScope))(block)
    }
  }

  /** Creates a context that can be used for creating ops and placing them on the same device as `colocationOps`.
    *
    * Details on the op creation context can be found in the documentation of the public API [[createWith]] function of
    * this library.
    *
    * @param  colocationOps  Colocation ops to use.
    * @param  ignoreExisting Boolean value indicating whether to ignore the colocation ops in the current context.
    * @param  block          Code block to run using the provided options.
    * @param  context        Current op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def colocateWith[R](colocationOps: Set[Op], ignoreExisting: Boolean = false)
      (block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    val newColocationOps: Set[Op] = {
      if (ignoreExisting)
        colocationOps
      else
        mergeColocationOps(colocationOps, context)
    }
    context.withValue(context.copy(colocationOps = newColocationOps))(block)
  }

  /** Merges a graph to the provided op creation context graph and returns the graph to use when specifying the updated
    * op creation context. The merging rules are specified in the documentation of the [[createWith]] function.
    *
    * @param  graph   Graph to merge.
    * @param  context Op creation context whose graph needs to be updated.
    * @return Graph to use for the new op creation context.
    */
  private[this] def mergeGraph(graph: Graph, context: OpCreationContext): Graph = {
    if (graph == null) context.graph else graph
  }

  /** Merges a name scope to the provided op creation context name scope and returns the name scope to use when
    * specifying the updated op creation context. The merging rules are specified in the documentation of the
    * [[createWith]] function.
    *
    * @param  nameScope Name scope to merge.
    * @param  context   Op creation context whose name scope needs to be updated.
    * @return Name scope to use for the new op creation context.
    */
  private[this] def mergeNameScope(nameScope: String, context: OpCreationContext): String = {
    if (nameScope == null) {
      context.nameScope
    } else {
      // Check whether the provided name scope is valid.
      // If the root name scope is being set, then stricter checks are performed on it (i.e., op naming checks). This
      // makes sure the name scope does not start with any illegal characters (e.g., '_', '-', '\', and '/').
      if ((context.nameScope == "" && nameScope != "" && !checkName(nameScope))
          || (context.nameScope != "" && !checkNameScope(nameScope)))
        throw IllegalNameException(s"Illegal name scope '$nameScope'.")
      if (nameScope == "")
        ""
      else if (nameScope.endsWith("/"))
        convertNameScopeToName(nameScope)
      else
        context.graph.uniqueName(nameScope)
    }
  }

  /** Merges a device to the provided op creation context device and returns the device to use when specifying the
    * updated op creation context. The merging rules are specified in the documentation of the [[createWith]] function.
    *
    * @param  device  Device to merge.
    * @param  context Op creation context whose device needs to be updated.
    * @return Device to use for the new op creation context.
    */
  private[this] def mergeDevice(
      device: OpSpecification => String = _ => "", context: OpCreationContext): OpSpecification => String = {
    val oldContextDevice = context.device
    opSpecification => {
      val oldDeviceSpecString = oldContextDevice(opSpecification)
      val newDeviceSpecString = if (device != null) device(opSpecification) else null
      // Check if the device has been reset or has to be reset for all subsequent nested scopes
      if (oldDeviceSpecString == null || newDeviceSpecString == null) {
        null
      } else {
        val oldDeviceSpec = DeviceSpecification.fromString(oldDeviceSpecString)
        val newDeviceSpec = DeviceSpecification.fromString(newDeviceSpecString)
        DeviceSpecification.merge(oldDeviceSpec, newDeviceSpec).toString
      }
    }
  }

  /** Merges a set of colocation ops to the provided op creation context set of colocation ops and returns the
    * set of colocation ops to use when specifying the updated op creation context. The merging rules are
    * specified in the documentation of the [[createWith]] function.
    *
    * @param  colocationOps Set of colocation ops to merge.
    * @param  context       Op creation context whose colocation ops need to be updated.
    * @return Set of colocation ops to use for the new op creation context.
    */
  private[this] def mergeColocationOps(colocationOps: Set[Op], context: OpCreationContext): Set[Op] = {
    if (colocationOps == null)
      context.colocationOps
    else if (colocationOps.isEmpty)
      Set.empty[Op]
    else
      context.colocationOps ++ colocationOps
  }

  /** Merges a set of control dependencies to the provided op creation context set of control dependencies and returns
    * the set of control dependencies to use when specifying the updated op creation context. The merging rules are
    * specified in the documentation of the [[createWith]] function.
    *
    * @param  controlDependencies Set of control dependencies to merge.
    * @param  context             Op creation context whose control dependencies needs to be updated.
    * @return Set of control dependencies to use for the new op creation context.
    */
  private[this] def mergeControlDependencies(controlDependencies: Set[Op], context: OpCreationContext): Set[Op] = {
    if (controlDependencies == null)
      context.controlDependencies
    else if (controlDependencies == Set.empty[Op])
      controlDependencies
    else
      context.controlDependencies ++ controlDependencies
  }

  /** Merges a set of attributes to the provided op creation context set of attributes and returns the set of attributes
    * to use when specifying the updated op creation context. The merging rules are specified in the documentation of
    * the [[createWith]] function.
    *
    * @param  attributes Set of attributes to merge.
    * @param  context    Op creation context whose attributes needs to be updated.
    * @return Set of attributes to use for the new op creation context.
    */
  private[this] def mergeAttributes(attributes: Map[String, Any], context: OpCreationContext): Map[String, Any] = {
    if (attributes == null)
      context.attributes
    else if (attributes == Map.empty[String, Any])
      attributes.filter(attribute => attribute._2 != null)
    else {
      var mergedMap = Map[String, Any](context.attributes.toSeq: _*)
      attributes.foreach(attribute => {
        if (attribute._2 == null && mergedMap.contains(attribute._1))
          mergedMap -= attribute._1
        else if (attribute._2 != null)
          mergedMap += attribute
      })
      mergedMap
    }
  }

  /** Merges a container to the provided op creation context container and returns the container to use when specifying
    * the updated op creation context. The merging rules are specified in the documentation of the [[createWith]]
    * function.
    *
    * @param  container Container to merge.
    * @param  context   Op creation context whose container needs to be updated.
    * @return Container to use for the new op creation context.
    */
  private[this] def mergeContainer(container: String, context: OpCreationContext): String = {
    if (container == null)
      context.container
    else
      container
  }

  /** Checks whether the provided string is a valid op name.
    *
    * @param  name String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[this] def checkName(name: String): Boolean = {
    VALID_OP_NAME_REGEX.pattern.matcher(name).matches
  }

  /** Checks whether the provided string is a valid name scope for creating ops.
    *
    * @param  nameScope String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[this] def checkNameScope(nameScope: String): Boolean = {
    VALID_NAME_SCOPE_REGEX.pattern.matcher(nameScope).matches
  }

  /** Converts the provided name scope to a valid op name, by removing a trailing `"/"` if there exists one.
    *
    * @param  nameScope Name scope to convert.
    * @return Name obtained from the provided name scope.
    */
  private[api] def convertNameScopeToName(nameScope: String): String = {
    if (nameScope.endsWith("/"))
      nameScope.substring(0, nameScope.length - 1)
    else
      nameScope
  }

  /** Asserts that two ops are defined in the same graph. If they are not, a [[GraphMismatchException]] is thrown.
    *
    * @param  op1 First op.
    * @param  op2 Second op.
    * @throws GraphMismatchException If the two ops lie in different graphs.
    */
  @throws[GraphMismatchException]
  private[ops] def assertSameGraph(op1: Op, op2: Op): Unit = {
    if (op1.graph != op2.graph)
      throw GraphMismatchException(s"'$op1' and '$op2' must be defined in the same graph.")
  }

  /** Returns the appropriate graph to use for the given inputs.
    *
    * This function provides a consistent algorithm for choosing the graph in which an op should be constructed in:
    *
    *   1. If the argument `graph` is provided and is not set to `null`, the function validates that all `inputs` are
    * defined in that graph.
    *   2. Otherwise, we attempt to select a graph from the first op in `inputs` and validate that all other `inputs`
    * are also defined in the same graph.
    *
    * @param  inputs Inputs.
    * @param  graph  Graph to use. If `null`, the graph is inferred from `inputs`.
    * @return The appropriate graph to use for the given inputs.
    * @throws GraphMismatchException If any two of the inputs lie in different graphs, or if `graph` is not `null` and
    *                                at least one of the `inputs` is not defined in it.
    */
  @throws[GraphMismatchException]
  private[ops] def getGraphFromInputs(inputs: Set[Op], graph: Graph = null): Graph = {
    val returnGraph = if (graph == null) inputs.head.graph else graph
    inputs.foreach(i => {
      if (graph == null)
        assertSameGraph(inputs.head, i)
      else if (i.graph != returnGraph)
        throw GraphMismatchException(s"'$i' is not defined in the passed-in graph.")
    })
    returnGraph
  }

  //region ProtoBuf Helper Functions

  private[api] def stripNameScope(nameScope: String, name: String): String = {
    if (nameScope != null && nameScope != "")
      name.replaceFirst(s"([\\^]|loc:@|^)$nameScope[\\/]+(.*)", "$1$2")
    else
      name
  }

  private[api] def prependNameScope(nameScope: String, name: String): String = {
    if (nameScope != null && nameScope != "")
      name.replaceFirst("([\\^]|loc:@|^)(.*)", "$1" + nameScope + "/$2")
    else
      name
  }

  //endregion ProtoBuf Helper Functions

  private[ops] final case class Builder(opType: String, name: String)
      (implicit context: DynamicVariable[OpCreationContext]) {
    if (!checkName(name))
      throw IllegalNameException(s"Illegal op name '$name'.")

    private val graph: Graph = context.graph

    private var built         : Boolean           = false
    // TODO: [OP] Avoid using this extra input functions sequence.
    private var inputFunctions: Seq[Long => Unit] = Seq.empty
    private var inputs        : Seq[Output]       = Seq.empty
    private var inputLists    : Seq[Seq[Output]]  = Seq.empty
    private var device        : Option[String]    = None
    private var attributes    : Map[String, Any]  = Map.empty

    /** Prunes control dependencies from the provided set, given that the op for which these control dependencies are
      * specified uses `op` as direct or indirect (through other ops) input or control input. This eliminates redundant
      * control dependencies due to transitive dependencies (e.g., if `a` depends on `b` and `c`, and `b` depends on
      * `c`, then the dependency of `a` on `c` is pruned).
      *
      * @param  controlDeps  Current set of control dependencies for the op that is being built.
      * @param  op           Op that is a direct or indirect (through other ops) input or control input, for the op that
      *                      is being built.
      * @param  processedOps Already processed ops (provided for efficiency purposes so that we do not go through them
      *                      a second time).
      */
    private[this] def pruneControlDependencies(
        controlDeps: mutable.Set[Op], op: Op, processedOps: mutable.Set[Op] = mutable.Set.empty[Op]): Unit = {
      if (processedOps.contains(op)) {
        // Prune op that is already used as input to the dependant op
        controlDeps -= op
        processedOps += op
        // Prune transitive control dependencies
        op.inputs.foreach(input => pruneControlDependencies(controlDeps, input.op, processedOps))
        op.controlInputs.foreach(pruneControlDependencies(controlDeps, _, processedOps))
      }
    }

    def build(): Op = graph.synchronized {
      using(graph.reference) { r =>
        if (built)
          throw OpBuilderUsedException("This op builder has already been used to built an op and cannot be re-used.")
        // TODO: [OP] Using just "this.name" here feels kind of awkward.
        device = Option(context.device(OpSpecification(name = this.name, opType = opType)))
        val name = {
          // If a name ends with a "/" then it is a name scope and we use it as-is, after removing the trailing "/".
          if (this.name.endsWith("/"))
            convertNameScopeToName(this.name)
          else
            graph.uniqueName(this.name)
        }
        val nativeHandle: Long = NativeOp.allocate(r.nativeHandle, opType, name)
        inputFunctions.foreach(_(nativeHandle))
        val controlDependencies: mutable.Set[Op] = mutable.Set(context.controlDependencies.toSeq: _*)
        inputs.foreach(input => pruneControlDependencies(controlDependencies, input.op))
        inputLists.foreach(_.foreach(input => pruneControlDependencies(controlDependencies, input.op)))
        controlDependencies.foreach(op => NativeOp.addControlInput(nativeHandle, op.nativeHandle))
        device.foreach(NativeOp.setDevice(nativeHandle, _))
        context.colocationOps.foreach(op => NativeOp.colocateWith(nativeHandle, op.nativeHandle))
        mergeAttributes(context.attributes)
        setAttributes(nativeHandle)
        // TODO: Set the "container" attribute when necessary. Need a way to check for statefulness.
        val operation = Op(graph, NativeOp.finish(nativeHandle))
        built = true
        operation
      }
    }

    private def mergeAttributes(attributes: Map[String, Any]): Unit = {
      attributes.foreach(this.attributes += _)
    }

    private def setAttributes(nativeHandle: Long): Unit = {
      attributes.foreach(attribute => {
        attribute._2 match {
          case value: String =>
            NativeOp.setAttrString(nativeHandle, attribute._1, encodeString(value))
          case value: Array[String] =>
            NativeOp.setAttrStringList(nativeHandle, attribute._1, value.map(encodeString))
          case value: Long =>
            NativeOp.setAttrInt(nativeHandle, attribute._1, value)
          case value: Array[Long] =>
            NativeOp.setAttrIntList(nativeHandle, attribute._1, value)
          case value: Float =>
            NativeOp.setAttrFloat(nativeHandle, attribute._1, value)
          case value: Array[Float] =>
            NativeOp.setAttrFloatList(nativeHandle, attribute._1, value)
          case value: Boolean =>
            NativeOp.setAttrBool(nativeHandle, attribute._1, value)
          case value: Array[Boolean] =>
            NativeOp.setAttrBoolList(nativeHandle, attribute._1, value)
          case value: DataType =>
            NativeOp.setAttrType(nativeHandle, attribute._1, value.cValue)
          case value: Array[DataType] =>
            NativeOp.setAttrTypeList(nativeHandle, attribute._1, value.map(_.cValue))
          case value: Tensor =>
            val handle = value.resolve()
            NativeOp.setAttrTensor(nativeHandle, attribute._1, handle)
            NativeTensor.delete(handle)
          case value: Array[Tensor] =>
            val handles = value.map(_.resolve())
            NativeOp.setAttrTensorList(nativeHandle, attribute._1, handles)
            handles.foreach(NativeTensor.delete)
          case value: Shape =>
            NativeOp.setAttrShape(nativeHandle, attribute._1, value.asArray.map(_.toLong), value.rank)
          case value: Array[Shape] =>
            NativeOp.setAttrShapeList(
              nativeHandle, attribute._1, value.map(_.asArray.map(_.toLong)), value.map(_.rank), value.length)
          case _ =>
            throw new IllegalArgumentException(s"Unsupported attribute type for attribute named '${attribute._1}.'")
        }
      })
    }

    private def encodeString(value: String): Array[Byte] = value.getBytes(Charset.forName("UTF-8"))

    def addInput(input: Output): Builder = {
      this.inputFunctions :+= {
        (nativeHandle: Long) => NativeOp.addInput(nativeHandle, input.op.nativeHandle, input.index)
      }
      this.inputs :+= input
      this
    }

    def addInputList(inputList: Seq[Output]): Builder = {
      this.inputFunctions :+= {
        (nativeHandle: Long) =>
          NativeOp.addInputList(nativeHandle, inputList.map(_.nativeHandle).toArray, inputList.map(_.index).toArray)
      }
      this.inputLists :+= inputList
      this
    }

    def setDevice(device: String): Builder = {
      this.device = Some(device)
      this
    }

    def setAttribute(name: String, value: String): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[String]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Long): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Long]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Float): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Float]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Boolean): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Boolean]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: DataType): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[DataType]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Tensor): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Tensor]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Shape): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Shape]): Builder = {
      attributes += name -> value
      this
    }
  }
}
