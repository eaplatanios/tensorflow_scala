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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.data.{Dataset, SupportedData}
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.{DataType, SupportedType, VARIANT}
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{Function => NativeFunction, Graph => NativeGraph}

import org.tensorflow.framework.FunctionDef
import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.mutable
import scala.collection.JavaConverters._

// TODO: [FUNCTIONS] Add support for function descriptions.

/**
  * @author Emmanouil Antonios Platanios
  */
case class Function[I, O](
    name: String,
    function: I => O
)(implicit
    evInput: Function.ArgType[I],
    evOutput: Function.ArgType[O]
) {
  def apply(
      arg: I,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false
  ): O = {
    val dataTypes = evInput.dataTypes(arg)
    val key = dataTypes.map(_.toString).mkString(":")
    InstantiatedFunction(
      name = s"${name}_$key",
      function = function,
      inputDataTypes = dataTypes,
      input = Some(arg),
      captureByValue = captureByValue, appendHashToName = appendHashToName
    )(evInput, evOutput)(arg)
  }

  def instantiate(
      inputDataTypes: Seq[DataType[Any]],
      inputShapes: Seq[Shape] = null,
      input: Option[I] = None,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false
  ): InstantiatedFunction[I, O] = {
    val key = (inputDataTypes.map(_.toString) ++
        Option(inputShapes).getOrElse(Seq.empty).map(_.toString)).mkString(":")
    InstantiatedFunction(
      name = s"${name}_$key",
      function = function,
      inputDataTypes = inputDataTypes,
      inputShapes = Option(inputShapes),
      input = input,
      captureByValue = captureByValue,
      appendHashToName = appendHashToName
    )(evInput, evOutput)
  }
}

object Function {
  trait ArgType[O] {
    def numOutputs: Int
    def outputs(arg: O): Seq[Output[Any]]
    def dataTypes(arg: O): Seq[DataType[Any]]
    def outputsDecoder(outputs: Seq[Output[Any]]): (O, Seq[Output[Any]])
    def outputsDecoderWithKnownArg(arg: O, outputs: Seq[Output[Any]]): (O, Seq[Output[Any]])
  }

  object ArgType {
    implicit def outputArgType[T]: ArgType[Output[T]] = {
      new ArgType[Output[T]] {
        override def numOutputs: Int = {
          1
        }

        override def outputs(arg: Output[T]): Seq[Output[Any]] = {
          Seq(arg)
        }

        override def dataTypes(arg: Output[T]): Seq[DataType[Any]] = {
          Seq(arg.dataType)
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (Output[T], Seq[Output[Any]]) = {
          (outputs.head.asInstanceOf[Output[T]], outputs.tail)
        }

        override def outputsDecoderWithKnownArg(
            arg: Output[T],
            outputs: Seq[Output[Any]]
        ): (Output[T], Seq[Output[Any]]) = {
          (outputs.head.asInstanceOf[Output[T]], outputs.tail)
        }
      }
    }

    implicit def outputIndexedSlicesArgType[T]: ArgType[OutputIndexedSlices[T]] = {
      new ArgType[OutputIndexedSlices[T]] {
        override def numOutputs: Int = {
          3
        }

        override def outputs(
            arg: OutputIndexedSlices[T]
        ): Seq[Output[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def dataTypes(
            arg: OutputIndexedSlices[T]
        ): Seq[DataType[Any]] = {
          Seq(arg.indices.dataType, arg.values.dataType, arg.denseShape.dataType)
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
          (OutputIndexedSlices(
            indices = outputs(0).asInstanceOf[Output[Long]],
            values = outputs(1).asInstanceOf[Output[T]],
            denseShape = outputs(2).asInstanceOf[Output[Long]]),
              outputs.drop(3))
        }

        override def outputsDecoderWithKnownArg(
            arg: OutputIndexedSlices[T],
            outputs: Seq[Output[Any]]
        ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
          (OutputIndexedSlices(
            indices = outputs(0).asInstanceOf[Output[Long]],
            values = outputs(1).asInstanceOf[Output[T]],
            denseShape = outputs(2).asInstanceOf[Output[Long]]),
              outputs.drop(3))
        }
      }
    }

    implicit def sparseOutputArgType[T]: ArgType[SparseOutput[T]] = {
      new ArgType[SparseOutput[T]] {
        override def numOutputs: Int = {
          3
        }

        override def outputs(
            arg: SparseOutput[T]
        ): Seq[Output[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def dataTypes(
            arg: SparseOutput[T]
        ): Seq[DataType[Any]] = {
          Seq(arg.indices.dataType, arg.values.dataType, arg.denseShape.dataType)
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (SparseOutput[T], Seq[Output[Any]]) = {
          (SparseOutput(
            indices = outputs(0).asInstanceOf[Output[Long]],
            values = outputs(1).asInstanceOf[Output[T]],
            denseShape = outputs(2).asInstanceOf[Output[Long]]),
              outputs.drop(3))
        }

        override def outputsDecoderWithKnownArg(
            arg: SparseOutput[T],
            outputs: Seq[Output[Any]]
        ): (SparseOutput[T], Seq[Output[Any]]) = {
          (SparseOutput(
            indices = outputs(0).asInstanceOf[Output[Long]],
            values = outputs(1).asInstanceOf[Output[T]],
            denseShape = outputs(2).asInstanceOf[Output[Long]]),
              outputs.drop(3))
        }
      }
    }

    // TODO: [FUNCTIONS] !!! Find a better way to deal with this for use in the reduce function of the "GroupByWindowDataset".

    case class VariantDataset[T] protected(
        handle: Output[Long],
        override val evData: SupportedData[T],
        private val dataType: Any = null,
        private val shape: Any = null
    ) extends Dataset[T] {
      override val name: String = "VariantDataset"

      override def createHandle(): Output[Long] = handle
      override def outputDataTypes: evData.D = dataType.asInstanceOf[evData.D]
      override def outputShapes: evData.S = shape.asInstanceOf[evData.S]
    }

    implicit def datasetArgType[T](implicit
        evData: SupportedData[T]
    ): ArgType[Dataset[T]] = {
      new ArgType[Dataset[T]] {
        override def numOutputs: Int = {
          1
        }

        override def outputs(arg: Dataset[T]): Seq[Output[Any]] = {
          Seq(arg.createHandle())
        }

        override def dataTypes(arg: Dataset[T]): Seq[DataType[Any]] = {
          Seq(VARIANT)
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (Dataset[T], Seq[Output[Any]]) = {
          (VariantDataset[T](
            handle = outputs.head.asInstanceOf[Output[Long]],
            evData = evData),
              outputs.drop(1))
        }

        override def outputsDecoderWithKnownArg(
            arg: Dataset[T],
            outputs: Seq[Output[Any]]
        ): (Dataset[T], Seq[Output[Any]]) = {
          (VariantDataset[T](
            handle = outputs.head.asInstanceOf[Output[Long]],
            evData = evData,
            dataType = arg.outputDataTypes.asInstanceOf[evData.D],
            shape = arg.outputShapes.asInstanceOf[evData.S]
          ), outputs.drop(1))
        }
      }
    }

    implicit val hnil: ArgType[HNil] = {
      new ArgType[HNil] {
        override def numOutputs: Int = {
          0
        }

        override def outputs(arg: HNil): Seq[Output[Any]] = {
          Seq.empty
        }

        override def dataTypes(arg: HNil): Seq[DataType[Any]] = {
          Seq.empty
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (HNil, Seq[Output[Any]]) = {
          (HNil, outputs)
        }

        override def outputsDecoderWithKnownArg(
            arg: HNil,
            outputs: Seq[Output[Any]]
        ): (HNil, Seq[Output[Any]]) = {
          (HNil, outputs)
        }
      }
    }

    implicit def recursiveConstructor[H, T <: HList](implicit
        argTypeHead: Lazy[ArgType[H]],
        argTypeTail: ArgType[T]
    ): ArgType[H :: T] = {
      new ArgType[H :: T] {
        override def numOutputs: Int = {
          argTypeHead.value.numOutputs + argTypeTail.numOutputs
        }

        override def outputs(arg: H :: T): Seq[Output[Any]] = {
          argTypeHead.value.outputs(arg.head) ++
              argTypeTail.outputs(arg.tail)
        }

        override def dataTypes(arg: H :: T): Seq[DataType[Any]] = {
          argTypeHead.value.dataTypes(arg.head) ++
              argTypeTail.dataTypes(arg.tail)
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (H :: T, Seq[Output[Any]]) = {
          val (decodedHead, outputsTail) = argTypeHead.value.outputsDecoder(outputs)
          val (decodedTail, tail) = argTypeTail.outputsDecoder(outputsTail)
          (decodedHead :: decodedTail, tail)
        }

        override def outputsDecoderWithKnownArg(
            arg: H :: T,
            outputs: Seq[Output[Any]]
        ): (H :: T, Seq[Output[Any]]) = {
          val (decodedHead, outputsTail) = argTypeHead.value.outputsDecoderWithKnownArg(arg.head, outputs)
          val (decodedTail, tail) = argTypeTail.outputsDecoderWithKnownArg(arg.tail, outputsTail)
          (decodedHead :: decodedTail, tail)
        }
      }
    }

    // This also covers `OutputIndexedSlices` and `SparseOutput` as they are case classes (i.e., products).
    implicit def productConstructor[P, L <: HList](implicit
        gen: Generic.Aux[P, L],
        argTypeL: ArgType[L],
        tupler: Tupler.Aux[L, P]
    ): ArgType[P] = {
      new ArgType[P] {
        override def numOutputs: Int = {
          argTypeL.numOutputs
        }

        override def outputs(arg: P): Seq[Output[Any]] = {
          argTypeL.outputs(gen.to(arg))
        }

        override def dataTypes(arg: P): Seq[DataType[Any]] = {
          argTypeL.dataTypes(gen.to(arg))
        }

        override def outputsDecoder(
            outputs: Seq[Output[Any]]
        ): (P, Seq[Output[Any]]) = {
          val (decoded, tail) = argTypeL.outputsDecoder(outputs)
          (tupler(decoded), tail)
        }

        override def outputsDecoderWithKnownArg(
            arg: P,
            outputs: Seq[Output[Any]]
        ): (P, Seq[Output[Any]]) = {
          val (decoded, tail) = argTypeL.outputsDecoderWithKnownArg(gen.to(arg), outputs)
          (tupler(decoded), tail)
        }
      }
    }
  }
}

// TODO: [TYPES] !!! What about type variance here?

private[api] class InstantiatedFunction[I, O] protected (
    val hashedName: String,
    val inputNames: Seq[String],
    val outputNames: Seq[String],
    private[ops] val dummyOutputs: O,
    val outputDataTypes: Seq[DataType[Any]],
    val outputShapes: Seq[Shape],
    val subFunctions: Set[InstantiatedFunction[_, _]],
    val extraInputs: Seq[Output[Any]],
    val functionDef: FunctionDef,
    val name: String,
    private[this] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
)(implicit
    evInput: Function.ArgType[I],
    evOutput: Function.ArgType[O]
) extends Closeable {
  /** Lock for the native handle. */
  private[InstantiatedFunction] def NativeHandleLock = {
    nativeHandleWrapper.Lock
  }

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = {
    nativeHandleWrapper.handle
  }

  /** Adds this function to the provided graph. */
  def addToGraph(graph: Graph): Unit = {
    graph.getFunction(hashedName) match {
      case Some(_) => ()
      case None =>
        // Add this function into the graph
        graph.addFunction(this)
        // Ensure that all sub-functions are also added to the graph
        subFunctions.foreach(graph.addFunction)
      // TODO: [FUNCTIONS] Add the gradient function too.
    }
  }

  /** Creates an op in the current graph that calls this function.
    *
    * The op that calls this function, passing the tensors in `inputs` as arguments. It returns the outputs of the call,
    * which are one or more tensors, structured according to the original Scala function that this function represents.
    *
    * @param  input                     Input to the function.
    * @param  inline                    Boolean parameter instructing the runtime whether to inline the function body
    *                                   into the call site.
    * @param  compiled                  Boolean parameter specifying whether to use XLA to compile the function.
    * @param  separateCompiledGradients Boolean parameter specifying whether to put each gradient sub-graph into a
    *                                   separate compilation scope. This gives fine-grained control over which portions
    *                                   of the graph will be compiled as a single unit. Compiling gradients separately
    *                                   may yield better performance for some graphs. The scope is named based on the
    *                                   scope of the forward computation as well as the name of the gradients. As a
    *                                   result, the gradients will be compiled in a scope that is separate from both the
    *                                   forward computation, and from other gradients.
    * @param  name                      Name for the created op.
    * @return Function output.
    */
  def apply(
      input: I,
      inline: Boolean = true,
      compiled: Boolean = false,
      separateCompiledGradients: Boolean = false,
      name: String = name
  ): O = {
    val outputs = Op.nameScope(name) {
      val outputs = evInput.outputs(input)
      addToGraph(outputs.head.graph)
      val builder = Op.Builder[Seq[Output[Any]], Seq[Output[Any]]](
        opType = hashedName,
        name = "Call",
        input = outputs ++ extraInputs,
        addAsIndividualInputs = true)
      builder.setAttribute("_noinline", inline)
      if (compiled) {
        val xlaScope = graphConstructionScope.value.attributes
            .getOrElse("_XlaScope", s"function_$name").toString
        builder
            .setAttribute("_XlaCompile", compiled)
            .setAttribute("_XlaSeparateCompiledGradients", separateCompiledGradients)
            .setAttribute("_XlaScope", xlaScope)
      }
      builder.build().output
    }
    evOutput.outputsDecoderWithKnownArg(dummyOutputs, outputs)._1
  }

  /** Constructs and returns a [[FunctionDef]] object, which is a serialized version of this function. */
  def toFunctionDef: FunctionDef = {
    functionDef
  }
}

object InstantiatedFunction {
  private[api] def apply[I, O](
      name: String,
      function: I => O,
      inputDataTypes: Seq[DataType[Any]],
      inputShapes: Option[Seq[Shape]] = None,
      input: Option[I] = None,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false,
      _inputNames: Seq[String] = null,
      _outputNames: Seq[String] = null
  )(implicit
      evInput: Function.ArgType[I],
      evOutput: Function.ArgType[O]
  ): InstantiatedFunction[I, O] = {
    require(inputDataTypes.lengthCompare(evInput.numOutputs) == 0,
      s"The number of 'inputDataTypes' provided (${inputDataTypes.length}) " +
          s"does not match the number of inputs (${evInput.numOutputs}).")

    // List of placeholders for the function definition
    val inputs = mutable.ListBuffer.empty[Output[Any]]
    val functionGraph = FunctionGraph(captureByValue)
    val (inputNames, outputNames, outputs, flattenedOutputs) = Op.createWith(functionGraph) {
      // Determine names for the function inputs
      val inputNames = {
        if (_inputNames != null) {
          require(_inputNames.lengthCompare(inputDataTypes.length) == 0,
            s"The number of 'inputNames' provided (${_inputNames.length}) " +
                s"does not match the number of inputs (${inputDataTypes.length}).")
          _inputNames
        } else {
          inputDataTypes.indices.map(i => s"input_$i")
        }
      }

      inputShapes match {
        case None =>
          inputs.appendAll((inputNames, inputDataTypes).zipped
              .map((name, dataType) => {
                Basic.placeholder(dataType = dataType, name = name)
              }))
        case Some(shapes) =>
          inputs.appendAll((inputNames, inputDataTypes, shapes).zipped
              .map((name, dataType, shape) => {
                Basic.placeholder(dataType = dataType, shape = shape, name = name)
              }))
      }

      // Call the Scala function and gather the output tensors
      val (outputs, flattenedOutputs) = {
        VariableScope.scope(
          name = "",
          underlyingGetter = functionGraph.customVariableGetter
        ) {
          // Unflatten the inputs, pass them to the function, and then flatten the returned outputs
          val outputs = function(
            input.map(evInput.outputsDecoderWithKnownArg(_, inputs)._1)
                .getOrElse(evInput.outputsDecoder(inputs)._1))
          val flattenedOutputs = evOutput.outputs(outputs)
              .map(functionGraph.capture)
          (evOutput.outputsDecoderWithKnownArg(outputs, flattenedOutputs)._1, flattenedOutputs)
        }
      }

      // Determine names for the function outputs
      val outputNames = {
        if (_outputNames != null) {
          require(_outputNames.lengthCompare(evOutput.numOutputs) == 0,
            s"The number of 'outputNames' provided (${_outputNames.length}) " +
                s"does not match the number of outputs (${evOutput.numOutputs}).")
          _outputNames
        } else {
          flattenedOutputs.indices.map(i => s"output_$i")
        }
      }
      (inputNames, outputNames, outputs, flattenedOutputs)
    }

    val extraInputs = functionGraph.extraInputs
    val subFunctions = functionGraph.functions
    inputs.appendAll(functionGraph.extraArgs)

    // Create the native function
    val nativeHandle = NativeFunction.graphToFunction(
      functionGraph.nativeHandle, name, appendHashToFnName = appendHashToName, null,
      inputs.map(_.op.nativeHandle).toArray, inputs.map(_.index).toArray,
      flattenedOutputs.map(_.op.nativeHandle).toArray, flattenedOutputs.map(_.index).toArray,
      outputNames.toArray)
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val functionDef = FunctionDef.parseFrom(nativeHandleWrapper.Lock.synchronized {
      NativeFunction.toFunctionDef(nativeHandle)
    })
    val functionName = functionDef.getSignature.getName
    val closeFn = () => {
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandleWrapper.handle != 0) {
          NativeFunction.delete(nativeHandleWrapper.handle)
          nativeHandleWrapper.handle = 0
        }
      }
    }
    val instantiatedFunction = new InstantiatedFunction(
      functionName, inputNames, outputNames, outputs, flattenedOutputs.map(_.dataType), flattenedOutputs.map(_.shape),
      subFunctions, extraInputs, functionDef, name, nativeHandleWrapper, closeFn)(evInput, evOutput)
    // Keep track of references in the Scala side and notify the native library when the function is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(instantiatedFunction, closeFn)
    instantiatedFunction.asInstanceOf[InstantiatedFunction[I, O]]
  }
}

/** Graph extension helper for constructing a function.
  *
  * A [[FunctionGraph]] keeps track of all inputs into every op created inside it. If any input is from another graph,
  * we keep track of it and substitute the input with a placeholder.
  *
  * Each captured input's corresponding placeholder is converted into a function argument and the caller passes in the
  * captured tensor.
  *
  * @param  captureByValue  TODO: !!!
  *
  * @author Emmanouil Antonios Platanios
  */
class FunctionGraph(
    override private[api] val nativeHandleWrapper: NativeHandleWrapper,
    protected val captureByValue: Boolean,
    override protected val closeFn: () => Unit
) extends Graph(nativeHandleWrapper, closeFn) {
  /** Graph used during construction of this graph. */
  val outerGraph: Graph = {
    Op.currentGraph
  }

  /** Variable scope used during construction of this graph. */
  private[ops] val outerVariableScope: VariableScope = {
    VariableScope.current
  }

  /** Captured op outputs that belong to other graphs and are used within this graph. */
  private[ops] val capturedOutputs: mutable.HashMap[Output[Any], Output[Any]] = {
    mutable.HashMap.empty[Output[Any], Output[Any]]
  }

  /** Extra placeholder arguments to feed to the function being created, that represent captured op output values. */
  private[ops] val extraArgs: mutable.ListBuffer[Output[Any]] = {
    mutable.ListBuffer.empty[Output[Any]]
  }

  /** Extra inputs to use when calling the function being created, that correspond to the extra arguments. */
  private[ops] val extraInputs: mutable.ListBuffer[Output[Any]] = {
    mutable.ListBuffer.empty[Output[Any]]
  }

  /** Extra variables that have been created on the outer graph and correspond to those created within this graph. */
  private[ops] val extraVars: mutable.ListBuffer[Output[Any]] = {
    mutable.ListBuffer.empty[Output[Any]]
  }

  /** Helper function for processing tensors before using them as inputs for ops placed in this graph. Useful for
    * creating function graphs. */
  override private[api] def processOpInput[T](value: Output[T]): Output[T] = {
    capture(value)
  }

  /** Adds the provided tensor to this graph and returns the captured tensor. */
  private[ops] def capture[T](output: Output[T]): Output[T] = {
    if (output.graph == this) {
      output
    } else if (captureByValue) {
      addOutputAndParents(outerGraph match {
        case g: FunctionGraph => g.capture(output)
        case _ => output
      })
    } else {
      // Referring to a tensor from other graph
      capturedOutputs.getOrElseUpdate(output, {
        // Substitute with a placeholder and hoist the new input placeholder out of any control flow context we might
        // currently be in.
        val placeholder = Op.createWith(controlDependencies = Set.empty) {
          Basic.placeholder(output.dataType, output.shape)
        }
        extraArgs.append(placeholder)
        extraInputs.append(outerGraph match {
          case g: FunctionGraph => g.capture(output)
          case _ => output
        })
        placeholder
      }).asInstanceOf[Output[T]]
    }
  }

  protected def addOutputAndParents[T](output: Output[T]): Output[T] = {
    addOpAndParents(output.op)
        .outputsSeq(output.index)
        .asInstanceOf[Output[T]]
  }

  protected def addOpAndParents[I, O](op: Op[I, O]): Op[I, O] = {
    op.graph.functions
        .filter(_.hashedName == op.opType)
        .foreach(_.addToGraph(Op.currentGraph))

    if (op.toOpDef.getIsStateful)
      throw InvalidArgumentException(s"Cannot capture a stateful op (name: ${op.name}, type: ${op.opType}) by value.")
    if (op.opType == "Placeholder" || op.opType == "PlaceholderV2")
      throw InvalidArgumentException(s"Cannot capture a placeholder (name: ${op.name}, type: ${op.opType}) by value.")

    val capturedOp = Op.createWith(controlDependencies = op.controlInputs.map(addOpAndParents)) {
      val opBuilder = Op.Builder[Seq[Output[Any]], Seq[Output[Any]]](
        opType = op.opType,
        name = op.name,
        input = op.inputsSeq.map(addOutputAndParents),
        addAsIndividualInputs = true)
      op.toNodeDef.getAttrMap.asScala.foreach(attribute => {
        opBuilder.setAttribute(attribute._1, attribute._2)
      })
      opBuilder.build()
    }
    op.outputsSeq.zip(capturedOp.outputsSeq)
        .foreach(o => capturedOutputs.update(o._1, o._2))
    capturedOp.asInstanceOf[Op[I, O]]
  }

  /** Custom variable getter for variables created within this function graph. */
  private[ops] val customVariableGetter: VariableGetter = new VariableGetter {
    override def apply[T: SupportedType](
        name: String,
        dataType: DataType[T],
        shape: Shape = null,
        initializer: Initializer = null,
        regularizer: Regularizer = null,
        trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
        cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null
    ): Variable[T] = {
      // TODO: [FUNCTIONS] !!! Deal with nested function graphs.
      // TODO: [FUNCTIONS] !!! Not sure if this works as it should. Especially the '.value' method of resource variables.
      // Here, we switch the default graph to the outer graph and ask the variable scope in which the function is defined
      // to give us the variable. The variable is stashed in extra_vars and returned to the caller. We capture these
      // variables so that the variable definition is hoisted upward to the outer-most graph.
      Op.createWith(outerGraph) {
        val variable = outerVariableScope.getVariable[T](
          store = VariableStore.current,
          name = name,
          shape = shape,
          initializer = initializer,
          regularizer = regularizer,
          trainable = trainable,
          reuse = reuse,
          collections = collections,
          cachingDevice = cachingDevice)
        extraVars.append(variable)
        variable
      }
    }
  }
}

/** Contains helper functions for dealing with function graphs. */
object FunctionGraph {
  /** Constructs and returns an empty new function graph. */
  def apply(captureByValue: Boolean): FunctionGraph = {
    val nativeHandle = NativeGraph.allocate()
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      var done = false
      nativeHandleWrapper.preCleanupFunctions.foreach(_ ())
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandle != 0) {
          while (!done && nativeHandleWrapper.referenceCount > 0) {
            try {
              nativeHandleWrapper.Lock.wait()
            } catch {
              case _: InterruptedException =>
                Thread.currentThread().interrupt()
                // TODO: Possible leak of the graph in this case?
                done = true
            }
          }
          if (!done) {
            nativeHandleWrapper.cleanupFunctions.foreach(_ ())
            NativeGraph.delete(nativeHandleWrapper.handle)
            nativeHandleWrapper.handle = 0
          }
        }
      }
    }
    val graph = new FunctionGraph(nativeHandleWrapper, captureByValue, closeFn)
    // Keep track of references in the Scala side and notify the native library when the graph is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(graph, closeFn)
    graph
  }
}
