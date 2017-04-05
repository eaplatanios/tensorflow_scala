package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.jni.{Operation => NativeOperation}
import org.platanios.tensorflow.api.Exception.{IllegalNameException, OpBuilderUsedException}

import java.nio.charset.Charset
import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Op(graph: Graph, unsafeNativeHandle: Long) {
  /** Name of the op. */
  def name: String = using(graph.reference) { _ => NativeOperation.name(unsafeNativeHandle) }

  /** Type of the op (i.e., the name of the computation performed by the operation). */
  def opType: String = using(graph.reference) { _ => NativeOperation.opType(unsafeNativeHandle) }

  /** Device in which the op tensors are stored and where computations are performed for this op. */
  def device: String = using(graph.reference) { _ => NativeOperation.device(unsafeNativeHandle) }

  /** Returns a symbolic handle to one of the tensors produced by this operation. */
  def output(index: Int): Op.Output = Op.Output(op = this, index = index)

  /** Number of tensors produced by this operation. */
  def numOutputs: Int = using(graph.reference) { _ => NativeOperation.numOutputs(unsafeNativeHandle) }

  def outputDataType(outputIndex: Int): DataType[_] =
    using(graph.reference) { r =>
      DataType.fromCValue(NativeOperation.outputDataType(r.nativeHandle, unsafeNativeHandle, outputIndex))
    }

  /** Returns the number of tensors fed as input to this operation. */
  def numInputs: Int = using(graph.reference) { _ => NativeOperation.numInputs(unsafeNativeHandle) }

  def inputDataType(inputIndex: Int): DataType[_] =
    using(graph.reference) { r =>
      DataType.fromCValue(NativeOperation.inputDataType(r.nativeHandle, unsafeNativeHandle, inputIndex))
    }
}

// TODO: Add control input options.
private[ops] final case class OpSpecification(name: String, opType: String)
private[api] final case class OpCreationContext(
    graph: Graph = Graph(), nameScope: String = "", device: OpSpecification => String = _ => "")

object Op {
  /** Convenient implicit conversion function used to convert devices specified as [[String]]s for use with the
    * [[createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceConversion(device: String): OpSpecification => String = _ => device

  /** Creates a context that can be used for creating ops according to the provided options.
    *
    * During graph creation, a context is maintained that includes: (i) the current graph in which new ops are placed,
    * (ii) the current name scope used for naming these new ops, and (iii) a device function, used to decide in which
    * device (e.g., CPU vs. GPU) the new ops should be placed and executed.
    *
    * when `createWith(...)` is used with a name scope, the provided name scope is appended to the context name scope,
    * generating a new op creation context. This new context is used for all ops created within the code block provided
    * in the `createWith(...)` function. The `nameScope` argument will be interpreted as follows:
    *   - A string will create a new name scope, in which `nameScope` is appended to the prefix of all operations
    *     created in the provided code block. If `nameScope` has been used before, it will be made unique by calling
    *     `uniqueName(graph = context.graph, name = nameScope)`.
    *   - A value of `""` will reset the current name scope to the top-level (i.e., empty) name scope.
    *
    * TODO: Support re-entering existing name scopes.
    *
    * This function checks the provided `nameScope` for validity by checking whether it matches: (i) the regular
    * expression `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*` if the current context name scope is empty (i.e., at the root), or
    * (ii) the regular expression `[A-Za-z0-9_.\\-/]*`, otherwise.
    *
    * When `createWith(...)` is used with a device, the `device` argument needs to be a function taking an
    * [[OpSpecification]] as input and returning a string representation of the device where the corresponding op should
    * be placed. This function is invoked every time a new op is created within the provided code block. If the function
    * returns `null` for some op, then all subsequent invocations of `createWith(device = ...)` in the provided code
    * block will be ignored. Note that, if the [[deviceConversion]] implicit conversion function is within scope, then
    * a `String` value (or `null`) can be used directly for the `device` field. In this case, the value provided will be
    * used as the device for all newly create ops in the provided code block. For information about the valid syntax of
    * device name strings, see the documentation in
    * [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).
    *
    * Note that the device scope may be overridden by op wrappers or other library code. For example, a variable
    * assignment op must be colocated with the corresponding variable. Incompatible device scopes will be ignored.
    *
    * Note that all arguments of this function are optional. If they are not provided, then the corresponding option in
    * current op creation context is left unchanged.
    *
    * Care must be taken if concurrency is used while creating the graph because the op creation context is wrapped
    * inside a [[scala.util.DynamicVariable]]. More information on this general issue can be found at
    * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
    *
    * @example {{{
    *   // Placing the new ops in the provided graph
    *
    *   val g = Graph()
    *   createWith(graph = g) {
    *     val c = constant(5.0)
    *     assert(c.graph == g)
    *   }
    *
    *   // Changing the name scope for the new ops
    *
    *   // No name scope used
    *   val c = constant(1.0, name = "C")
    *   assert(c.op.name == "C")
    *   val c1 = constant(2.0, name = "C_1")
    *   assert(c_1.op.name == "C_1")
    *
    *   // Create a name scope called "Nested"
    *   createWith(nameScope = "Nested") {
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
    *       // Reset the name scope using ""
    *       createWith(nameScope = "") {
    *         val c2 = constant(6.0, name = "C_2")
    *         assert(c2.op.name == "C_2")
    *       }
    *     }
    *   }
    *
    *   // Specifying which device to use
    *
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
    *
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
    *
    *   // Changing graph, name scope, and device to use for new ops.
    *
    *   createWith(graph = g, nameScope = "Nested", device = "/GPU:0") {
    *     val c = constant(11.0, name = "C")
    *     assert(c.graph == g)
    *     assert(c.op.name == "Nested/C")
    *     assert(c.device == "/device:GPU:0")
    *   }
    * }}}
    *
    * @param  graph     Graph to use as default for new ops.
    * @param  nameScope Name scope to use.
    * @param  device    Device function to use.
    * @param  block     Code block to run using the provided options.
    * @param  context   Current op creation context.
    * @tparam R         Return type of the code block.
    * @return Return value of the code block.
    * @throws IllegalNameException If the provided name scope does not pass the regular expression validity checks.
    */
  @throws[IllegalNameException]
  def createWith[R](graph: Graph = null, nameScope: String = null, device: OpSpecification => String = _ => "")
      (block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    val newGraph: Graph = if (graph == null) context.graph else graph
    val newNameScope: String = {
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
        else if (context.nameScope == "")
          uniqueName(graph = context.graph, name = s"${convertNameScopeToName(nameScope)}")
        else
          uniqueName(graph = context.graph, name = s"${context.nameScope}/${convertNameScopeToName(nameScope)}")
      }
    }
    val newDevice: OpSpecification => String = {
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
    context.withValue(context.copy(graph = newGraph, nameScope = newNameScope, device = newDevice))(block)
  }

  private[this] val validOpNameRegex: Regex = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r

  /** Checks whether the provided string is a valid op name.
    *
    * @param  name String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[this] def checkName(name: String): Boolean =
    validOpNameRegex.pattern.matcher(name).matches

  /** Returns a unique operation name in a graph, based on the provided `name`.
    *
    * `uniqueName` first checks if an op named `name` exists in `graph`. If it doesn't, then `name` is returned.
    * Otherwise, `{name}_{i}` is returned, where `i` is the first non-zero integer for which no op with that name exists
    * in the `graph`.
    *
    * @note If this function is called while creating a new op, the graph needs to be locked while generating a unique
    *       name and adding the new op to the graph, so that no other op with the same name is added to the graph in the
    *       meantime. You rarely need to call `uniqueName` directly. Most of the time you just need to create
    *       `usingNameScope(...)` (which is also thread-safe) blocks to generate structured names.
    *
    * @note Operation names are displayed in error messages reported by the TensorFlow runtime, and in various
    *       visualization tools such as TensorBoard.
    *
    * @param  graph   Graph for which the unique name is generated.
    * @param  name    Name in which to base the generated unique name.
    * @param  counter Current counter value `i`.
    * @return Unique name.
    */
  private[this] def uniqueName(graph: Graph, name: String, counter: Int = 1): String = {
    if (graph.op(name).isEmpty)
      name
    else if (graph.op(s"${name}_$counter").isEmpty)
      s"${name}_$counter"
    else
      uniqueName(graph = graph, name = name, counter = counter + 1)
  }

  private[this] val validNameScopeRegex: Regex = "^[A-Za-z0-9_.\\-/]*$".r

  /** Checks whether the provided string is a valid name scope for creating ops.
    *
    * @param  nameScope String to check.
    * @return Boolean value indicating whether the check was successful.
    */
  private[this] def checkNameScope(nameScope: String): Boolean =
    validNameScopeRegex.pattern.matcher(nameScope).matches

  /** Converts the provided name scope to a valid op name, by removing a trailing `"/"` if there exists one.
    *
    * @param  nameScope Name scope to convert.
    * @return Name obtained from the provided name scope.
    */
  private[this] def convertNameScopeToName(nameScope: String): String = {
    if (nameScope.endsWith("/"))
      nameScope.substring(0, nameScope.length - 1)
    else
      nameScope
  }

  final case class Output(op: Op, index: Int) {
    def graph: Graph = op.graph
    def name: String = s"${op.name}:$index"
    def device: String = op.device
    def dataType: DataType[_] = op.outputDataType(index)
    def shape: Shape = Shape(
      using(op.graph.reference) { r => NativeOperation.shape(r.nativeHandle, op.unsafeNativeHandle, index) })

    //region Ops

    def +(other: Output): Output = MathOps.add(x = this, y = other)
    def -(other: Output): Output = MathOps.subtract(x = this, y = other)
    def *(other: Output): Output = MathOps.multiply(x = this, y = other)
    def /(other: Output): Output = MathOps.divide(x = this, y = other)

    //endregion Ops

    override def toString: String = name
  }

  private[ops] final case class Builder(context: OpCreationContext, opType: String, name: String) {
    private val graph: Graph = context.graph
    private val opName: String = if (context.nameScope == "") name else s"${context.nameScope}/$name"
    if (!checkName(name = opName))
      throw IllegalNameException(s"Illegal op name '$opName'.")

    private var built: Boolean = false
    private var inputs: Seq[Output] = Seq[Output]()
    private var device: Option[String] = None
    private var byteArrayAttributes: Map[String, Array[Byte]] = Map[String, Array[Byte]]()
    private var longAttributes: Map[String, Long] = Map[String, Long]()
    private var longArrayAttributes: Map[String, Array[Long]] = Map[String, Array[Long]]()
    private var floatAttributes: Map[String, Float] = Map[String, Float]()
    private var floatArrayAttributes: Map[String, Array[Float]] = Map[String, Array[Float]]()
    private var booleanAttributes: Map[String, Boolean] = Map[String, Boolean]()
    private var booleanArrayAttributes: Map[String, Array[Boolean]] = Map[String, Array[Boolean]]()
    private var dataTypeAttributes: Map[String, Int] = Map[String, Int]()
    private var dataTypeArrayAttributes: Map[String, Array[Int]] = Map[String, Array[Int]]()
    private var tensorAttributes: Map[String, Long] = Map[String, Long]()
    private var tensorArrayAttributes: Map[String, Array[Long]] = Map[String, Array[Long]]()
    private var shapeAttributes: Map[String, Shape] = Map[String, Shape]()

    def build(): Op = using(graph.reference) { _ =>
      if (built)
        throw OpBuilderUsedException("This op builder has already been used to built an op and cannot be re-used.")
      device = Option(context.device(OpSpecification(name = name, opType = opType)))
      graph.synchronized {
        using(graph.reference) { r =>
          val nativeHandle: Long = NativeOperation.allocate(
            r.nativeHandle, opType, uniqueName(graph = graph, name = opName))
          inputs.foreach(input => NativeOperation.addInput(nativeHandle, input.op.unsafeNativeHandle, input.index))
          device.foreach(NativeOperation.setDevice(nativeHandle, _))
          byteArrayAttributes.foreach(a => NativeOperation.setAttrString(nativeHandle, a._1, a._2))
          longAttributes.foreach(a => NativeOperation.setAttrInt(nativeHandle, a._1, a._2))
          longArrayAttributes.foreach(a => NativeOperation.setAttrIntList(nativeHandle, a._1, a._2))
          floatAttributes.foreach(a => NativeOperation.setAttrFloat(nativeHandle, a._1, a._2))
          floatArrayAttributes.foreach(a => NativeOperation.setAttrFloatList(nativeHandle, a._1, a._2))
          booleanAttributes.foreach(a => NativeOperation.setAttrBool(nativeHandle, a._1, a._2))
          booleanArrayAttributes.foreach(a => NativeOperation.setAttrBoolList(nativeHandle, a._1, a._2))
          dataTypeAttributes.foreach(a => NativeOperation.setAttrType(nativeHandle, a._1, a._2))
          dataTypeArrayAttributes.foreach(a => NativeOperation.setAttrTypeList(nativeHandle, a._1, a._2))
          tensorAttributes.foreach(a => NativeOperation.setAttrTensor(nativeHandle, a._1, a._2))
          tensorArrayAttributes.foreach(a => NativeOperation.setAttrTensorList(nativeHandle, a._1, a._2))
          shapeAttributes.foreach(a => NativeOperation.setAttrShape(nativeHandle, a._1, a._2.shape, a._2.rank))
          val operation = Op(graph, NativeOperation.finish(nativeHandle))
          built = true
          operation
        }
      }
    }

    def addInput(input: Output): Builder = {
      inputs :+= input
      this
    }

    def addInputs(inputs: Seq[Output]): Builder = {
      this.inputs ++= inputs
      this
    }

    def setDevice(device: String): Builder = {
      this.device = Some(device)
      this
    }

    def setAttribute(name: String, value: String): Builder = {
      byteArrayAttributes += name -> value.getBytes(Charset.forName("UTF-8"))
      this
    }

    def setAttribute(name: String, value: Array[Byte]): Builder = {
      byteArrayAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Long): Builder = {
      longAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Long]): Builder = {
      longArrayAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Float): Builder = {
      floatAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Float]): Builder = {
      floatArrayAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Boolean): Builder = {
      booleanAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: Array[Boolean]): Builder = {
      booleanArrayAttributes += name -> value
      this
    }

    def setAttribute(name: String, value: DataType[_]): Builder = {
      dataTypeAttributes += name -> value.cValue
      this
    }

    def setAttribute(name: String, value: Array[DataType[_]]): Builder = {
      dataTypeArrayAttributes += name -> value.map(_.cValue)
      this
    }

    def setAttribute(name: String, value: Tensor[_]): Builder = {
      tensorAttributes += name -> value.nativeHandle
      this
    }

    def setAttribute(name: String, value: Array[Tensor[_]]): Builder = {
      tensorArrayAttributes += name -> value.map(_.nativeHandle)
      this
    }

    def setAttribute(name: String, value: Shape): Builder = {
      shapeAttributes += name -> value
      this
    }
  }
}
