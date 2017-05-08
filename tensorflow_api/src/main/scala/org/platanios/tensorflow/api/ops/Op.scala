package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.jni.{Op => NativeOp}
import org.platanios.tensorflow.api.Exception._

import java.nio.charset.Charset

import scala.collection.mutable
import scala.util.DynamicVariable

/** Represents a graph node, or as we shall call it, an operation, that performs computation on tensors.
  *
  * An `Op` is a symbolic representation of the computation it performs. It is a node in a TensorFlow [[Graph]] that
  * takes zero or more `Op.Output` objects as input, and produces zero or more `Op.Output` objects as output. `Op`
  * objects are constructed by calling op creation functions, such as [[Basic.constant]] or [[Math.matMul]].
  *
  * For example, `val c = MathOps.matMul(a, b)` creates an `Op` of type `"MatMul"` that takes `Op.Output`s `a` and
  * `b` as input, and produces `Op.Output` `c` as output.
  *
  * @note The `Op.Input` class is simply a wrapper around an `Op` meant to represent one of its inputs. Actual op inputs
  *       have type `Op.Output` since they represent outputs of other ops. Currently, `Op.Input` is only useful for
  *       representing consumers of an `Op`'s outputs.
  *
  * After the graph has been launched in a [[Session]], an `Op` can be executed by using [[Session.run]].
  *
  * TODO: Add `Op.run` use example, once that is supported.
  * @author Emmanouil Antonios Platanios
  */
final case class Op private (graph: Graph, private[api] val nativeHandle: Long) {
  graph.opsCache.update(nativeHandle, this) // Update the ops cache of the graph with the current op

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
    Option(NativeOp.getAttrStringList(nativeHandle, COLOCATION_OPS_ATTRIBUTE_NAME))
        .map(_.toSet[String]
                 .filter(_.startsWith(COLOCATION_OPS_ATTRIBUTE_PREFIX))
                 .map(opName => graph.findOp(opName.substring(COLOCATION_OPS_ATTRIBUTE_PREFIX.length)).get))
        .getOrElse(Set.empty[Op])
  }

  /** Number of inputs to this op (i.e., number of tensors fed as input to this op). */
  lazy val numInputs: Int = using(graph.reference) { _ => NativeOp.numInputs(nativeHandle) }

  /** Inputs of this op. Note that these inputs are outputs of other ops and thus have type [[Op.Output]]. */
  lazy val inputs: Array[Op.Output] = (0 until numInputs).map(index => using(graph.reference) { _ =>
    val jniOpOutput = NativeOp.input(nativeHandle, index)
    val op = graph.opsCache.getOrElseUpdate(
      jniOpOutput.opHandle,
      Op(graph, jniOpOutput.opHandle))
    op.outputs(jniOpOutput.outputIndex)
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
  lazy val outputs: Array[Op.Output] = (0 until numOutputs).map(i => Op.Output(op = this, index = i)).toArray

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

  /** Gets the data type of the specified input of this op.
    *
    * @param  index Input index.
    * @return Data type of the specified input.
    */
  private def inputDataType(index: Int): DataType = using(graph.reference) { r =>
    DataType.fromCValue(NativeOp.inputDataType(r.nativeHandle, nativeHandle, index))
  }

  /** Gets the data type of the specified output of this op.
    *
    * @param  index Output index.
    * @return Data type of the specified output.
    */
  private def outputDataType(index: Int): DataType = using(graph.reference) { r =>
    DataType.fromCValue(NativeOp.outputDataType(r.nativeHandle, nativeHandle, index))
  }

  /** Gets the (current) number of consumers of the specified output of this op. These are other ops that use the
    * specified output as one of their inputs.
    *
    * @param  index Output index.
    * @return Current consumers of the specified output.
    */
  private def outputConsumers(index: Int): Array[Op.Input] = using(graph.reference) { _ =>
    val array = NativeOp.consumers(nativeHandle, index)
    if (array == null)
      Array.empty[Op.Input]
    else
      array.map(jniOpOutput => {
        val op = graph.opsCache.getOrElseUpdate(jniOpOutput.opHandle, Op(graph, jniOpOutput.opHandle))
        Op.Input(op = op, index = index)
      })
  }

  /** Gets the value of a string-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringAttribute(name: String): String = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrString(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a string-array-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def stringArrayAttribute(name: String): Array[String] = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrStringList(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a long-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def longAttribute(name: String): Long = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrInt(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a float-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def floatAttribute(name: String): Float = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrFloat(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a boolean-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def booleanAttribute(name: String): Boolean = using(graph.reference) { _ =>
    try {
      NativeOp.getAttrBool(nativeHandle, name)
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a data type-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def dataTypeAttribute(name: String): DataType = using(graph.reference) { _ =>
    try {
      DataType.fromCValue(NativeOp.getAttrType(nativeHandle, name))
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
    }
  }

  /** Gets the value of a shape-valued attribute of this op with name `name`.
    *
    * @param  name Attribute name.
    * @return Attribute value.
    * @throws IllegalArgumentException If the no attribute with name `name` can be found for this op.
    */
  @throws[IllegalArgumentException]
  def shapeAttribute(name: String): Shape = using(graph.reference) { _ =>
    try {
      Shape.fromSeq(NativeOp.getAttrShape(nativeHandle, name).map(_.toInt))
    } catch {
      case e: Exception => throw new IllegalArgumentException(
        s"Op has no attribute named '$name'. TensorFlow native library error message: ${e.getMessage}")
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
    graph: Graph = Graph(), nameScope: String = "", device: OpSpecification => String = _ => "",
    colocationOps: Set[Op] = Set.empty, controlDependencies: Set[Op] = Set.empty,
    attributes: Map[String, Any] = Map.empty, container: String = "") // TODO: !!! Use containers.

object Op {
  /** Convenient implicit conversion function used to convert devices specified as [[String]]s for use with the
    * [[createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceImplicitConversion(device: String): OpSpecification => String = _ => device

  /** Convenient implicit conversion function used to convert op outputs to their corresponding ops for use with the
    * [[createWith]] function, when specifying control dependencies.
    *
    * @param  opOutput Op output.
    * @return Op corresponding to the provided op output.
    */
  implicit def opOutputToOpImplicitConversion(opOutput: Op.Output): Op = opOutput.op

  /** Returns the graph of the current op creation context. */
  private[api] def currentGraph(implicit context: DynamicVariable[OpCreationContext]): Graph = context.graph

  /** Returns the name scope of the current op creation context. */
  private[api] def currentNameScope(implicit context: DynamicVariable[OpCreationContext]): String = {
    if (context.nameScope == "")
      ""
    else
      s"${context.nameScope}/"
  }

  /** Creates a context that can be used for creating ops according to the provided options.
    *
    * = General Information =
    *
    * During graph creation, a context is maintained that includes:
    *   - The current graph in which new ops are placed.
    *   - The current name scope used for naming these new ops.
    *   - A device function, used to decide in which device (e.g., CPU) the new ops should be placed and executed.
    *   - A set of colocation ops for the newly constructed ops. This means that the newly created ops will be placed on
    *     the same device as these colocation ops.
    *   - A set of ops defining control dependencies for the newly constructed ops. This means that the newly
    *     constructed ops are constrained to only execute after the provided set of ops has finished executing.
    *   - A map from op attribute names to values for the newly constructed ops. These attributes will be applied to all
    *     newly constructed ops.
    *   - A container name for the newly constructed resource ops. All newly constructed resource ops will be placed in
    *     the provided container.
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
    * block will be ignored. Note that, if the [[deviceImplicitConversion]] implicit conversion function is within
    * scope, then a `String` value (or `null`) can be used directly for the `device` field. In this case, the value
    * provided will be used as the device for all newly create ops in the provided code block. For information about the
    * valid syntax of device name strings, see the documentation in
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
    *   def matMulOnGPU(opSpecification: OpSpecification): String = {
    *     if (opSpecification.opType == "MatMul")
    *       "/GPU:0"
    *     else
    *       "/CPU:0"
    *   }
    *
    *   createWith(device = matMulOnGPU) {
    *     // All ops of type "MatMul" constructed in this code block will be placed on GPU 0. All other operations will
    *     // be placed on CPU 0.
    *     val c = constant(9.0)
    *     assert(c.device == "/device:CPU:0")
    *     val m = matMul(c, constant(10.0))
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
    * the control dependencies ops have finished executing. Note that if a set of control dependencies already exist in
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
  def createWith[R](
      graph: Graph = null, nameScope: String = null, device: OpSpecification => String = _ => "",
      colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
      container: String = null)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    val newGraph: Graph = mergeGraph(graph, context)
    val newNameScope: String = mergeNameScope(nameScope, context)
    val newDevice: OpSpecification => String = mergeDevice(device, context)
    val newColocationOps: Set[Op] = mergeColocationOps(colocationOps, context)
    val newControlDependencies: Set[Op] = mergeControlDependencies(controlDependencies, context)
    val newAttributes: Map[String, Any] = mergeAttributes(attributes, context)
    val newContainer: String = mergeContainer(container, context)
    context.withValue(context.copy(
      graph = newGraph, nameScope = newNameScope, device = newDevice, colocationOps = newColocationOps,
      controlDependencies = newControlDependencies, attributes = newAttributes, container = newContainer))(block)
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
  private[ops] def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R)
      (implicit context: DynamicVariable[OpCreationContext]): R = {
    val newNameScope: String = mergeNameScope(nameScope, context)
    if (values.nonEmpty) {
      val newGraph: Graph = mergeGraph(getGraphFromInputs(values), context)
      context.withValue(context.copy(graph = newGraph, nameScope = newNameScope))(block)
    } else {
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
  def colocateWith[R](colocationOps: Set[Op], ignoreExisting: Boolean = false)
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
      if ((context.nameScope == "" && !checkName(nameScope))
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
  private[this] def getGraphFromInputs(inputs: Set[Op], graph: Graph = null): Graph = {
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

  private[ops] def stripNameScope(nameScope: String, name: String): String = {
    if (nameScope != null)
      name.replaceFirst(s"([\\^]|loc:@|^)$nameScope[\\/]+(.*)", "$1$2")
    else
      name
  }

  private[ops] def prependNameScope(nameScope: String, name: String): String = {
    if (nameScope != null)
      name.replaceFirst("([\\^]|loc:@|^)(.*)", "$1" + nameScope + "/$2")
    else
      name
  }

  //endregion ProtoBuf Helper Functions

  /** Wrapper around an `Op` meant to represent one of its inputs. Actual op inputs have type `Op.Output` since they
    * represent outputs of other ops. Currently, `Op.Input` is only useful for representing consumers of an `Op`'s
    * outputs.
    *
    * @param  op    Op whose input this class represents.
    * @param  index Input index.
    */
  final case class Input private(op: Op, index: Int) {
    /** Name of this op input. This is simply set to `"<op.name>:<index>"`. */
    lazy val name: String = s"${op.name}:$index"

    /** Data type of this op input. */
    lazy val dataType: DataType = op.inputDataType(index)

    /** Graph where the op belongs. */
    def graph: Graph = op.graph

    override def toString: String = s"Op.Input(name = $name, dataType = $dataType)"
  }

  /** Trait representing outputs of an `Op`'s computation. */
  sealed trait OutputLike {
    /** Graph where the op belongs. */
    def graph: Graph

    /** Name of this op output. */
    def name: String

    /** Data type of this op output. */
    def dataType: DataType

    /** Device on which this op output will be placed. */
    def device: String

    /** Op that generates this output. */
    def op: Op

    /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
    def consumers: Array[Op.Input]
  }

  trait OutputConvertible {
    /** Returns the [[Op.Output]] that this [[Op.OutputLike]] object represents. */
    def toOpOutput: Op.Output
  }

  sealed trait OutputIndexedSlicesConvertible {
    /** Returns an [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      *
      * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
      *                  shape of this tensor at graph creation time (instead of execution time), if known.
      * @return [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      */
    def toOpOutputIndexedSlices(optimize: Boolean = true): Op.OutputIndexedSlices
  }

  /** Representation of one of the outputs of an `Op`'s computation.
    *
    * An `Op.Output` is a symbolic handle to one of the outputs of an `Op`. It does not hold the values of that op's
    * output, but instead provides a means of computing those values in a TensorFlow [[Session]].
    *
    * This class has two primary purposes:
    *
    *   1. An `Op.Output` can be passed as input to another `Op`. This builds a dataflow connection between ops, which
    * enables TensorFlow to execute an entire [[Graph]] that represents a large, multi-step computation.
    *   2. After the graph has been launched in a [[Session]], the value of an [[Op.Output]] can be computed by passing
    * it to `Session.run`.
    *   3. `Op.Output.evaluate` can also be used to compute the value of an [[Op.Output]] If no session is provided,
    * then the default session is used.
    *
    * In the following example, `c`, `d`, and `e` are symbolic [[Op.Output]] objects, whereas `result` is a Scala array
    * that stores a concrete value:
    * {{{
    *   val c = constant(Array(Array(1.0, 2.0), Array(3.0, 4.0)))
    *   val d = constant(Array(Array(1.0, 1.0), Array(0.0, 1.0)))
    *   val e = matMul(c, d)
    *   val result = e.evaluate() // 'result' now holds the result of the matrix multiplication.
    * }}}
    *
    * @param  op    Op whose output this class represents.
    * @param  index Output index.
    */
  final case class Output private(op: Op, index: Int)
      extends OutputLike with OutputConvertible with OutputIndexedSlicesConvertible {
    /** Graph where the op belongs. */
    override def graph: Graph = op.graph

    /** Name of this op output. This is simply set to `"<op.name>:<index>"`. */
    override def name: String = s"${op.name}:$index"

    /** Data type of this op output. */
    override def dataType: DataType = op.outputDataType(index)

    /** Device on which this op output will be placed. */
    override def device: String = op.device

    /** Consumers of this op output (i.e., ops that use this op output as one of their inputs). */
    override def consumers: Array[Op.Input] = op.outputConsumers(index)

    /** Shape of the tensor that this op output represents. */
    def shape: Shape = Shape.fromSeq(using(op.graph.reference) { r =>
      NativeOp.shape(r.nativeHandle, op.nativeHandle, index).map(_.toInt)
    })

    /** Sets the shape of this op output to the provided shape.
      *
      * This method can be useful in cases when shape inference fails, but the shape of the op output is known by the
      * user of the library.
      *
      * @param  shape Shape to use.
      */
    def setShape(shape: Shape): Unit = using(op.graph.reference) { r =>
      NativeOp.setShape(r.nativeHandle, op.nativeHandle, index, shape.asArray.map(_.toLong), shape.rank)
    }

    /** Evaluates this op output.
      *
      * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
      * op output.
      *
      * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
      * the evaluation.
      *
      * @param  feeds   Tensors to feed into the session for this evaluation.
      * @param  session Optional session to use for the evaluation.
      * @return Value of this op output, for this evaluation.
      */
    def evaluate(feeds: Map[Op.Output, Tensor] = Map.empty, session: Session = null): Tensor = {
      val effectiveSession = if (session == null) graph.defaultSession else session
      effectiveSession.run(feeds, Array(this))(0)
    }

    //region Slicing

    // TODO: Maybe add support for a name argument for the constructed op?
    /** Creates an op that slices this op according to the provided indexers.
      *
      * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
      *
      * @param  indexers Sequence of indexers to use.
      * @return Created op.
      */
    def slice(indexers: Indexer*): Op.Output = Indexer.toStridedSlice(indexers: _*)(this)

    //endregion Slicing

    //region Ops

    def unary_- : Output = Math.negate(this)
    def +(other: Output): Output = Math.add(x = this, y = other) // TODO: [SPARSE]
    def -(other: Output): Output = Math.subtract(x = this, y = other) // TODO: [SPARSE]
    def *(other: Output): Output = Math.multiply(x = this, y = other) // TODO: [SPARSE]
    def /(other: Output): Output = Math.divide(x = this, y = other) // TODO: [SPARSE]
    def **(other: Output): Output = Math.pow(x = this, y = other) // TODO: [SPARSE]

    def unary_! : Output = Math.logicalNot(x = this)
    def &&(other: Output): Output = Math.logicalAnd(x = this, y = other)
    def ||(other: Output): Output = Math.logicalOr(x = this, y = other)

    def ==(other: Output): Output = Math.equal(x = this, y = other)
    def !=(other: Output): Output = Math.notEqual(x = this, y = other)
    def <(other: Output): Output = Math.less(x = this, y = other)
    def <=(other: Output): Output = Math.lessEqual(x = this, y = other)
    def >(other: Output): Output = Math.greater(x = this, y = other)
    def >=(other: Output): Output = Math.greaterEqual(x = this, y = other)

    //endregion Ops

    /** Returns the [[Op.Output]] that this [[Op.OutputLike]] object represents. */
    override def toOpOutput: Op.Output = this

    /** Returns an [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      *
      * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
      *                  shape of this tensor at graph creation time (instead of execution time), if known.
      * @return [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      */
    override def toOpOutputIndexedSlices(optimize: Boolean = true): Op.OutputIndexedSlices = {
      val denseShape = Basic.shape(this, dataType = DataType.Int32, optimize = optimize)
      val indices = Math.range(Basic.constant(0), denseShape(0))
      OutputIndexedSlices(indices = indices, values = this, denseShape = denseShape)
    }

    /** Creates an op that slices this op according to the provided indexers.
      *
      * More details into how to construct and use indexers are provided in the [[Indexer]] documentation.
      *
      * @param  indexers Sequence of indexers to use.
      * @return Created op.
      */
    def apply(indexers: Indexer*): Op.Output = slice(indexers: _*)

    override def toString: String = s"Op.Output(name = $name, shape = $shape, dataType = $dataType, device = $device)"

    override def equals(that: Any): Boolean = that match {
      case that: Output => this.op == that.op && this.index == that.index
      case _ => false
    }

    override def hashCode(): Int = {
      val prime = 31
      var result = 1
      result = prime * result + op.hashCode
      result = prime * result + index
      result
    }
  }

  /** Sparse representation of one of the outputs of an `Op`'s computation. of a set of tensor slices at given indices.
    *
    * This class if a simple wrapper for a pair (or a set of three) of [[Op.Output]] objects:
    *   - `indices`: A one-dimensional integer [[Op.Output]] with shape `[D0]`.
    *   - `values`: An [[Op.Output]] of any data type, with shape `[D0, D1, ..., Dn]`.
    *   - `denseShape`: Optionally, an integer [[Op.Output]] with shape `[LARGE0, D1, ..., Dn]`.
    *
    * An [[Op.OutputIndexedSlices]] is typically used to represent a subset of a larger [[Op.Output]], `dense`, of shape
    * `[LARGE0, D1, ..., Dn]`, where `LARGE0 >> D0`. The values in `indices` are the indices in the first dimension of
    * the slices that have been extracted from the larger tensor.
    *
    * The dense [[Op.Output]], `dense`, represented by [[Op.OutputIndexedSlices]], `slices`, has:
    * {{{
    *   dense(slices.indices(i), ::, ::, ...) = slices.values(i, ::, ::, ...)
    * }}}
    *
    * The [[Op.OutputIndexedSlices]] class is used primarily in the definition of gradients for operations that have
    * sparse gradients, such as `gather`.
    *
    * Note that this is different than [[Op.SparseOutput]] which uses multi-dimensional indices and scalar values.
    *
    * @param  indices    Indices along the first dimension of the corresponding dense [[Op.Output]].
    * @param  values     Values corresponding to the provided indices.
    * @param  denseShape Shape of the corresponding dense [[Op.Output]].
    */
  final case class OutputIndexedSlices private(indices: Op.Output, values: Op.Output, denseShape: Op.Output = null)
      extends OutputLike with OutputConvertible with OutputIndexedSlicesConvertible {
    /** Graph that contains `values`, `indices`, and `denseShape`. */
    override def graph: Graph = getGraphFromInputs(Set(values, indices, denseShape))

    /** Name of this op output indexed slices. */
    override def name: String = s"${values.name}[${indices.name}]" +
        (if (denseShape ne null) s"(shape = ${denseShape.name})" else "")

    /** Data type of this op output indexed slices. */
    override def dataType: DataType = values.dataType

    /** Device on which these op output indexed slices will be placed. */
    override def device: String = values.device

    /** Op that outputs these indexed slices. */
    override def op: Op = values.op

    /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
    override def consumers: Array[Op.Input] = values.consumers

    /** Returns the [[Op.Output]] that this [[Op.OutputLike]] object represents. */
    override def toOpOutput: Op.Output = {
      if (denseShape ne null)
        throw new IllegalStateException(
          s"Op output conversion requested the conversion of 'Op.OutputIndexedSlices', '$this', which has no dense " +
              s"shape information available.")
      // TODO: Add check for large number of elements (e.g., > 100000000).
      createWith(nameScope = "IndexedSlicesToOutput") {
        Math.unsortedSegmentSum(data = values, segmentIndices = indices, segmentsNumber = denseShape(0))
      }
    }

    /** Returns an [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      *
      * @param  optimize Boolean flag indicating whether to optimize this conversion by using a constant op with the
      *                  shape of this tensor at graph creation time (instead of execution time), if known.
      * @return [[Op.OutputIndexedSlices]] that has the same value as this [[Op.OutputLike]].
      */
    override def toOpOutputIndexedSlices(optimize: Boolean = true): Op.OutputIndexedSlices = this

    override def toString: String = {
      s"Op.OutputIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
          s"device = $device)}"
    }
  }

  /** Represents a sparse op output.
    *
    * TensorFlow represents a sparse tensor as three separate dense tensors: `indices`, `values`, and `denseShape`. In
    * Scala, the three tensors are collected into a `SparseTensor` class for ease of use.  If you have separate
    * `indices`, `values`, and `denseShape` tensors, wrap them in a `SparseTensor` object before passing to the
    * relevant sparse tensor manipulation ops.
    *
    * Concretely, the sparse tensor `SparseTensor(indices, values, denseShape)` comprises the following components,
    * where `N` and `rank` are the number of values and number of dimensions in the `SparseTensor`, respectively:
    *
    *   - `indices`: Two-dimensional `Int64` tensor with shape `[N, rank]`, which specifies the indices of the elements
    * in the sparse tensor that have nonzero values (elements are zero-indexed). For example,
    * `indices = [[1, 3], [2, 4]]` specifies that the elements with indexes `[1, 3]` and `[2, 4]` have nonzero
    * values.
    *   - `values`: One-dimensional tensor of any type, with shape `[N]`, which supplies the values for each element in
    * `indices`. For example, given `indices = [[1, 3], [2, 4]]`, the parameter `values = [18, 3.6]` specifies that
    * element `[1, 3]` of the sparse tensor has a value of `18`, and element `[2, 4]` of the tensor has a value of
    * `3.6`.
    *   - `denseShape`: One-dimensional `Int64` tensor with shape `[rank]`, which specifies the dense shape of the
    * sparse tensor.  For example, `denseShape = [3, 6]` specifies a two-dimensional 3x6 tensor,
    * `denseShape = [2, 3, 4]` specifies a three-dimensional 2x3x4 tensor, and `denseShape = [9]` specifies a
    * one-dimensional tensor with 9 elements.
    *
    * The corresponding dense tensor, `dense`, satisfies:
    * {{{
    *   dense.shape == denseShape
    *   dense(indices(i)) = values(i) // Using a somewhat loose notation with respect to indexing.
    * }}}
    *
    * IMPORTANT NOTE: By convention, `indices` should be sorted in row-major order (or equivalently lexicographic order
    * on `indices(i)`). This is not enforced when `SparseTensor` objects are constructed, but most ops assume correct
    * ordering. If the ordering of sparse tensor `st` is wrong, a fixed version can be obtained by calling
    * `sparseReorder(st)`.
    *
    * For example, the sparse tensor `SparseTensor(indices = [[0, 0], [1, 2]], values = [1, 2], denseShape = [3, 4])`,
    * represents the dense tensor `[[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]`.
    * {{{
    *   // The sparse tensor:
    *   SparseTensor(indices = Tensor(Tensor(0, 0), Tensor(1, 2)), values = Tensor(1, 2), denseShape = Shape(3, 4))
    *   // represents the dense tensor:
    *   //
    * }}}
    *
    * @param  indices    Two-dimensional `Int64` tensor with shape `[N, rank]`.
    * @param  values     One-dimensional tensor with shape `[N]`.
    * @param  denseShape One-dimensional `Int64` tensor with shape `[rank]`.
    */
  final case class SparseOutput private(indices: Op.Output, values: Op.Output, denseShape: Op.Output)
      extends OutputLike {
    // TODO: Add constructor from scala arrays?
    if (indices.dataType != DataType.Int64)
      throw InvalidDataTypeException(s"Indices cannot have '${indices.dataType}' data type. They have to be 'Int64'.")
    if (denseShape.dataType != DataType.Int64)
      throw InvalidDataTypeException(s"Dense shape cannot have '${indices.dataType}' data type. It has to be 'Int64'.")
    // TODO: Add a "subShape" method?
    Shape(indices.shape.withRank(2)(0)).assertIsCompatibleWith(Shape(values.shape.withRank(1)(0)))
    Shape(indices.shape.withRank(2)(1)).assertIsCompatibleWith(Shape(denseShape.shape.withRank(1)(0)))

    /** Graph that contains `values`, `indices`, and `denseShape`. */
    override def graph: Graph = getGraphFromInputs(Set(values, indices, denseShape))

    /** Name of this sparse op output. */
    override def name: String = s"${values.name}[${indices.name}]" +
        (if (denseShape ne null) s"(shape = ${denseShape.name})" else "")

    /** Data type of this sparse op output. */
    override def dataType: DataType = values.dataType

    /** Device on which this sparse op output will be placed. */
    override def device: String = values.device

    /** Op that outputs this sparse tensor. */
    override def op: Op = values.op

    /** Consumers of these indexed slices (i.e., ops that use this op output as one of their inputs). */
    override def consumers: Array[Op.Input] = values.consumers

    /** Gets the [[Shape]] corresponding to the shape of the dense tensor that this sparse tensor represents.
      *
      * @return Dense tensor shape.
      */
    def shape: Shape = constantValueAsShape(denseShape)

    /** Evaluates this sparse op output.
      *
      * If `feeds` is non-empty, then the provided feed values are fed into the session for computing the value of this
      * op output.
      *
      * If `session` is `null` (i.e., not provided), then the default session is used. Otherwise, `session` is used for
      * the evaluation.
      *
      * @param  feeds   Tensors to feed into the session for this evaluation.
      * @param  session Optional session to use for the evaluation.
      * @return Value of this sparse op output, for this evaluation, represented as tuple containing the indices, the
      *         values, and the dense shape.
      */
    def value(
        feeds: Map[Op.Output, Tensor] = Map.empty, session: Session = null): (Tensor, Tensor, Tensor) = {
      val effectiveSession = if (session == null) graph.defaultSession else session
      val fetches = effectiveSession.run(feeds, Array(indices, values, denseShape))
      (fetches(0), fetches(1), fetches(2))
    }

    override def toString: String = {
      s"Op.OutputIndexedSlices(values = ${values.name}, indices = ${indices.name}, denseShape = ${denseShape.name}, " +
          s"device = $device)}"
    }
  }

  /** Converts the provided sparse output value to a sparse op output.
    *
    * @param  sparseOutputValue Sparse output value represented as tuple containing the indices, the values, and the
    *                           dense shape.
    * @return Sparse op output.
    */
  private[api] def convertToSparseOutput(sparseOutputValue: (Tensor, Tensor, Tensor)): SparseOutput = {
    SparseOutput(
      Basic.constant(sparseOutputValue._1), Basic.constant(sparseOutputValue._2),
      Basic.constant(sparseOutputValue._3))
  }

  // TODO: !!!

  private[api] def constantValue(tensor: Op.Output): Tensor = {
    val value = tensor.op.opType match {
      case "Const"    => ??? // TODO: !!! Needs MakeNdArray()
      case "Shape"    =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.isFullyDefined)
          Tensor(tensor.dataType, inputShape.asArray.map(Tensor(_)): _*)
        null
      case "Size"     =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.isFullyDefined)
          Tensor(DataType.Int32, Tensor(inputShape.asArray.product))
        null
      case "Rank"     =>
        val inputShape = tensor.op.inputs(0).shape
        if (inputShape.numElements.isDefined)
          Tensor(DataType.Int32, Tensor(inputShape.numElements.get))
        null
      case "Range"    =>
        val start = constantValue(tensor.op.inputs(0))
        if (start == null) {
          null
        } else {
          val limit = constantValue(tensor.op.inputs(1))
          if (limit == null) {
            null
          } else {
            val delta = constantValue(tensor.op.inputs(2))
            if (delta == null) {
              null
            } else {
              ??? // TODO: !!! Create tensor range?
            }
          }
        }
      case "Cast"     =>
        val preCast = constantValue(tensor.op.inputs(0))
        if (preCast == null) {
          null
        } else {
          ??? // TODO: !!! Get data type attribute from op.
        }
      case "Concat"   =>
        val axis = constantValue(tensor.op.inputs(0))
        if (axis == null) {
          null
        } else {
          val values = tensor.op.inputs.tail.map(constantValue)
          if (values.contains(null)) {
            null
          } else {
            ??? // TODO: !!! Concatenate tensors.
          }
        }
      case "ConcatV2" =>
        val axis = constantValue(tensor.op.inputs(tensor.op.numInputs - 1))
        if (axis == null) {
          null
        } else {
          val values = tensor.op.inputs.dropRight(1).map(constantValue)
          if (values.contains(null)) {
            null
          } else {
            ??? // TODO: !!! Concatenate tensors.
          }
        }
      case "Pack"     =>
        val values = tensor.op.inputs.map(constantValue)
        if (values.contains(null)) {
          null
        } else {
          ??? // TODO: !!! Concatenate tensors.
        }
      case "Fill"     =>
        val fillShape = tensor.shape
        val fillValue = constantValue(tensor.op.inputs(0))
        if (fillShape.isFullyDefined && fillValue != null)
          Tensor.fill(fillValue.dataType, fillShape)(fillValue.scalar)(fillValue.dataType.supportedType)
        else
          null
      case _          => null
    }
    if (value != null) {
      // The caller may now depend on the constant value of 'tensor', so conservatively prevent it from being fed.
      tensor.graph.preventFeeding(tensor)
    }
    value
  }

  /** Version of [[constantValue]] that returns a [[Shape]].
    *
    * This version should be used when a constant tensor value is interpreted as a (possibly partial) shape (e.g., in
    * the shape function for `reshape`). By explicitly requesting a [[Shape]] as the return value, it is possible to
    * represent unknown dimensions. In contrast, [[constantValue]] is all-or-nothing.
    *
    * @param  tensor One-dimensional tensor to be evaluated.
    * @return [[Shape]] based on the constant value of `tensor`.
    */
  private[api] def constantValueAsShape(tensor: Op.Output): Shape = {
    // TODO: !!! Do we really need this function?
    val shape = tensor.shape.withRank(1)
    if (shape == Shape(0)) {
      Shape.scalar()
    } else {
      tensor.op.opType match {
        case "Shape"    => tensor.op.inputs(0).shape
        case "Pack"     =>
          var returnShape = Shape.scalar()
          tensor.op.inputs.foreach(input => {
            // 'input' must be a scalar. Attempt to evaluate it, and append it to 'returnShape'.
            returnShape = returnShape.concatenateWith(Shape(constantValue(input).scalar.asInstanceOf[Int]))
          })
          returnShape
        case "Concat"   =>
          // We assume that 'tensor.op.inputs(0)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          var returnShape = Shape.scalar()
          tensor.op.inputs.tail.foreach(input => {
            // 'input' must be a vector. Attempt to evaluate it as a shape, and concatenate it with 'returnShape'.
            returnShape = returnShape.concatenateWith(constantValueAsShape(input))
          })
          returnShape
        case "ConcatV2" =>
          // We assume that 'tensor.op.inputs(-1)' evaluates to 0, as this is the only legal value when concatenating
          // vectors, and it will have been checked by a previous shape function.
          var returnShape = Shape.scalar()
          tensor.op.inputs.dropRight(1).foreach(input => {
            // 'input' must be a vector. Attempt to evaluate it as a shape, and concatenate it with 'returnShape'.
            returnShape = returnShape.concatenateWith(constantValueAsShape(input))
          })
          returnShape
        case _          =>
          var returnShape = Shape.unknown(shape(0))
          val value = constantValue(tensor)
          if (value != null) {
            require(value.rank == 1, "Only rank-1 tensors can be converted to shapes.")
            // TODO: !!! Does this work?
            import value.dataType.supportedType
            val shape = Shape(
              (0 until value.numElements).map(value.getElementAtFlattenedIndex(_).toInt): _*)
            returnShape = returnShape.mergeWith(shape)
          }
          returnShape
      }
    }
  }

  private[ops] final case class Builder(opType: String, name: String)
      (implicit context: DynamicVariable[OpCreationContext]) {
    if (!checkName(name))
      throw IllegalNameException(s"Illegal op name '$name'.")

    private val graph : Graph  = context.graph

    private var built     : Boolean             = false
    private var inputs    : Seq[Output]         = Seq.empty
    private var inputLists: Seq[Array[Output]]  = Seq.empty
    private var device    : Option[String]      = None
    private var attributes: Map[String, Any]    = Map.empty

    /** Prunes control dependencies from the provided set, given that the op for which these control dependencies are
      * specified uses `op` as direct or indirect (through other ops) input or control input. This eliminates redundant
      * control dependencies due to transitive dependencies (e.g., if `a` depends on `b` and `c`, and `b` depends on
      * `c`, then the dependency of `a` on `c` is pruned).
      *
      * @param  controlDeps Current set of control dependencies for the op that is being built.
      * @param  op          Op that is a direct or indirect (through other ops) input or control input, for the op that
      *                     is being built.
      */
    private[this] def pruneControlDependencies(controlDeps: mutable.Set[Op], op: Op): Unit = {
      // TODO: Check if this is too expensive for large graphs.
      // Prune op that is already used as input to the dependant op
      controlDeps -= op
      // Prune transitive control dependencies
      op.inputs.foreach(input => pruneControlDependencies(controlDeps, input.op))
      op.controlInputs.foreach(pruneControlDependencies(controlDeps, _))
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
        inputs.foreach(input => NativeOp.addInput(nativeHandle, input.op.nativeHandle, input.index))
        inputLists.foreach(inputList => NativeOp.addInputList(
          nativeHandle, inputList.map(_.nativeHandle), inputList.map(_.index)))
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
          case value: Tensor.NativeView =>
            NativeOp.setAttrTensor(nativeHandle, attribute._1, value.nativeHandle)
          case value: Array[Tensor.NativeView] =>
            NativeOp.setAttrTensorList(nativeHandle, attribute._1, value.map(_.nativeHandle))
          case value: Shape =>
            NativeOp.setAttrShape(nativeHandle, attribute._1, value.asArray.map(_.toLong), value.rank)
          case value: Array[Shape] => ??? // TODO: !!!
          case _ =>
            throw new IllegalArgumentException(s"Unsupported attribute type for attribute named '${attribute._1}.'")
        }
      })
    }

    private def encodeString(value: String): Array[Byte] = value.getBytes(Charset.forName("UTF-8"))

    def addInput(input: Output): Builder = {
      inputs :+= input
      this
    }

    def addInputs(inputs: Seq[Output]): Builder = {
      this.inputs ++= inputs
      this
    }

    def addInputList(inputs: Seq[Output]): Builder = {
      this.inputLists :+= inputs.toArray
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

    def setAttribute(name: String, value: Tensor.NativeView): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Tensor.NativeView]): Builder = {
      attributes += name -> value
      this
    }

    def setAttribute(name: String, value: Shape): Builder = {
      attributes += name -> value
      this
    }
  }
}
