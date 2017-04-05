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
  /** Returns the full name of the Operation. */
  def name: String = using(graph.reference) { _ => NativeOperation.name(unsafeNativeHandle) }

  /** Returns the type of the operation, i.e., the name of the computation performed by the
    * operation. */
  def opType: String = using(graph.reference) { _ => NativeOperation.opType(unsafeNativeHandle) }

  /** Returns a symbolic handle to one of the tensors produced by this operation. */
  def output(index: Int): Op.Output = Op.Output(op = this, index = index)

  /** Returns the number of tensors produced by this operation. */
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

// TODO: Add device and control inputs options.
private[api] final case class OpCreationContext(graph: Graph = Graph(), nameScope: String = "")

object Op {
  /** Creates a context that can be used for creating operations within the specified graph.
    *
    * During graph creation, a context is maintained that includes the current graph in which new ops are placed.
    * Whenever `usingGraph(...)` is used, all ops created within the provided code block will be placed in the provided
    * graph.
    *
    * This method should be used if you want to create multiple graphs in the same process. For convenience, a global
    * default graph is provided and used by default. All ops will be added to this graph if you do not create a new
    * graph explicitly.
    *
    * Care must be taken if concurrency is used while creating the graph because the graph creation context is wrapped
    * inside a [[scala.util.DynamicVariable]]. More information on this general issue can be found at
    * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
    *
    * @example  {{{
    *   val graph = Graph()
    *   usingGraph(graph) {
    *     val c = constant(5.0)
    *     assert(c.graph == graph)
    *   }
    * }}}
    *
    * @param  graph   Graph to use as default for new ops.
    * @param  block   Code block to run using the provided graph as the default graph in which new ops are placed.
    * @param  context Current graph creation context.
    * @tparam R       Return type of the code block.
    *
    * @return Return value of the code block.
    */
  def usingGraph[R](graph: Graph)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R =
    context.withValue(context.copy(graph = graph))(block)

  /** Creates a context that can be used for generating hierarchical names for operations.
    *
    * During graph creation, a context is maintained that includes the current name scope. Whenever
    * `usingNameScope(...)` is used, the provided name scope is appended to the context name scope, generating a new
    * graph creation context. This new context is used for all ops created within the code block provided in the
    * `usingNameScope(...)` function.
    *
    * The `nameScope` argument will be interpreted as follows:
    *   - A string will create a new name scope, in which `nameScope` is appended to the prefix of all operations
    *     created in the provided code block. If `nameScope` has been used before, it will be made unique by calling
    *     `uniqueName(graph = context.graph, name = nameScope)`.
    *   - A value of `""` or `null` will reset the current name scope to the top-level (i.e., empty) name scope.
    *
    * Care must be taken if concurrency is used while creating the graph because the graph creation context is wrapped
    * inside a [[scala.util.DynamicVariable]]. More information on this general issue can be found at
    * [[http://stevenskelton.ca/threadlocal-variables-scala-futures/]].
    *
    * TODO: Support re-entering existing name scopes.
    *
    * @example  {{{
    *   // No name scope used
    *   val c = constant(1.0, name = "c")
    *   assert(c.op.name == "c")
    *   val c1 = constant(2.0, name = "c_1")
    *   assert(c_1.op.name == "c_1")
    *
    *   // Create a name scope called "nested"
    *   usingNameScope("nested") {
    *     val nestedC = constant(3.0, name = "c")
    *     assert(nestedC.op.name == "nested/c")
    *
    *     // Create a nested name scope called "inner"
    *     usingNameScope("inner") {
    *       val nestedInnerC = constant(4.0, name = "c")
    *       assert(nestedInnerC.op.name == "nested/inner/c")
    *     }
    *
    *     // Create a nested name scope called "inner_1"
    *     usingNameScope("inner_1") {
    *       val nestedInner1C = constant(5.0, name = "c")
    *       assert(nestedInner1C.op.name == "nested/inner_1/c")
    *
    *       // Reset the name scope using ""
    *       usingNameScope("") {
    *         val c2 = constant(6.0, name = "c_2")
    *         assert(c2.op.name == "c_2")
    *       }
    *
    *       // Reset the name scope using null
    *       usingNameScope(null) {
    *         val c3 = constant(7.0, name = "c_3")
    *         assert(c3.op.name == "c_3")
    *       }
    *     }
    *   }
    * }}}
    *
    * @note This function checks the provided `nameScope` for validity by checking whether it matches: (i) the regular
    *       expression `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*` if the current context name scope is empty (i.e., at the root),
    *       or (ii) the regular expression `[A-Za-z0-9_.\\-/]*`, otherwise.
    *
    * @param  nameScope Name scope to use.
    * @param  block     Code block to run using the provided name scope.
    * @param  context   Current graph creation context.
    * @tparam R         Return type of the code block.
    *
    * @return Return value of the code block.
    *
    * @throws IllegalNameException If the provided name scope does not pass the validity regular expression checks.
    */
  @throws[IllegalNameException]
  def usingNameScope[R](nameScope: String)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    // Check whether the provided name scope is valid.
    // If the root name scope is being set, then stricter checks are performed on it (i.e., op naming checks). This
    // makes sure the name scope does not start with any illegal characters (e.g., '_', '-', '\', and '/').
    if (nameScope != null &&
        ((context.nameScope == "" && !checkName(nameScope)) || (context.nameScope != "" && !checkNameScope(nameScope))))
      throw IllegalNameException(s"Illegal name scope '$nameScope'.")
    val newNameScope: String = {
      if (nameScope == "" || nameScope == null)
        ""
      else if (context.nameScope == "")
        uniqueName(graph = context.graph, name = s"${convertNameScopeToName(nameScope)}")
      else
        uniqueName(graph = context.graph, name = s"${context.nameScope}/${convertNameScopeToName(nameScope)}")
    }
    context.withValue(context.copy(nameScope = newNameScope))(block)
  }

  private[this] val validOpNameRegex: Regex = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r

  /** Checks whether the provided string is a valid op name.
    *
    * @param  name  String to check.
    *
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
    *
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
    *
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

  private[ops] def opBuildHelper(
      context: OpCreationContext, opType: String, name: String, inputs: Output*): Op.Builder = {
    val opName: String = {
      if (context.nameScope == "")
        name
      else
        s"${context.nameScope}/$name"
    }
    Op.Builder(graph = context.graph, opType = opType, name = opName).addInputs(inputs)
  }

  private[ops] final case class Builder(graph: Graph, opType: String, name: String) {
    if (!checkName(name = name))
      throw IllegalNameException(s"Illegal op name '$name'.")

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
      graph.synchronized {
        val nativeHandle: Long = using(graph.reference) { r =>
          NativeOperation.allocate(r.nativeHandle, opType, uniqueName(graph = graph, name = name))
        }
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
