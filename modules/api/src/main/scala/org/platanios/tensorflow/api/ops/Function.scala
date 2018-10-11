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
import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{Function => NativeFunction, Graph => NativeGraph}

import org.tensorflow.framework.FunctionDef

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
    evInput: NestedStructure[I],
    evOutput: NestedStructure[O]
) {
  def apply(
      arg: I,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false
  ): O = {
    val dataTypes = evInput.outputs(arg).map(_.dataType)
    val key = dataTypes.map(_.toString).mkString(":")
    InstantiatedFunction(
      name = s"${name}_$key",
      function = function,
      inputDataType = evInput.dataType(arg),
      input = Some(arg),
      captureByValue = captureByValue, appendHashToName = appendHashToName
    )(evInput.asInstanceOf[NestedStructure.Aux[I, evInput.D, evInput.S]], evOutput).apply(arg)
  }

  def instantiate[ID, IS](
      inputDataType: ID,
      inputShape: Option[IS] = None,
      input: Option[I] = None,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false
  )(implicit
      evInputSpecific: NestedStructure.Aux[I, ID, IS]
  ): InstantiatedFunction[I, O] = {
    val inputDataTypes = evInputSpecific.dataTypes(inputDataType)
    val inputShapes = inputShape.map(evInputSpecific.shapes)
    val key = (inputDataTypes.map(_.toString) ++
        inputShapes.getOrElse(Seq.empty).map(_.toString)).mkString(":")
    InstantiatedFunction(
      name = s"${name}_$key",
      function = function,
      inputDataType = inputDataType,
      inputShape = inputShape,
      input = input,
      captureByValue = captureByValue,
      appendHashToName = appendHashToName
    )(evInputSpecific, evOutput)
  }
}

// TODO: [TYPES] !!! What about type variance here?

private[api] class InstantiatedFunction[I, O] protected(
    val hashedName: String,
    val inputNames: Seq[String],
    val outputNames: Seq[String],
    private[ops] val _dummyOutput: O,
    val subFunctions: Set[InstantiatedFunction[_, _]],
    val extraInputs: Seq[Output[Any]],
    val functionDef: FunctionDef,
    val name: String,
    private[this] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
)(implicit
    evInput: NestedStructure[I],
    evOutput: NestedStructure[O]
) extends Closeable {
  def outputDataTypes[D](implicit evOSpecific: NestedStructure.Aux[O, D, _]): D = {
    evOSpecific.dataType(_dummyOutput)
  }

  def outputShapes[S](implicit evOSpecific: NestedStructure.Aux[O, _, S]): S = {
    evOSpecific.shape(_dummyOutput)
  }

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
    evOutput.decodeOutputFromOutput(_dummyOutput, outputs)._1
  }

  /** Constructs and returns a [[FunctionDef]] object, which is a serialized version of this function. */
  def toFunctionDef: FunctionDef = {
    functionDef
  }
}

object InstantiatedFunction {
  private[api] def apply[I, ID, IS, O](
      name: String,
      function: I => O,
      inputDataType: ID,
      inputShape: Option[IS] = None,
      input: Option[I] = None,
      captureByValue: Boolean = false,
      appendHashToName: Boolean = false
  )(implicit
      evInput: NestedStructure.Aux[I, ID, IS],
      evOutput: NestedStructure[O]
  ): InstantiatedFunction[I, O] = {
    // List of placeholders for the function definition.
    val inputDataTypes = evInput.dataTypes(inputDataType)
    val inputs = mutable.ListBuffer.empty[Output[Any]]
    val functionGraph = FunctionGraph(captureByValue)
    val (inputNamesWithDefault, outputNamesWithDefault, outputs, flattenedOutputs) = Op.createWith(functionGraph) {
      // Determine names for the function inputs.
      val inputNamesWithDefault = inputDataTypes.indices.map(i => s"input_$i")
      inputShape match {
        case None =>
          inputs.appendAll((inputNamesWithDefault, inputDataTypes).zipped
              .map((name, dataType) => {
                Basic.placeholder(name = name)(TF.fromDataType(dataType))
              }))
        case Some(shape) =>
          inputs.appendAll((inputNamesWithDefault, inputDataTypes, evInput.shapes(shape)).zipped
              .map((name, dataType, shape) => {
                Basic.placeholder(shape, name = name)(TF.fromDataType(dataType))
              }))
      }

      // Call the Scala function and gather the output tensors.
      val (outputs, flattenedOutputs) = {
        VariableScope.scope(
          name = "",
          underlyingGetter = functionGraph.customVariableGetter
        ) {
          // Unflatten the inputs, pass them to the function, and then flatten the returned outputs.
          val outputs = function(input.map(evInput.decodeOutputFromOutput(_, inputs)._1)
              .getOrElse(evInput.decodeOutputFromDataType(inputDataType, inputs)._1))
          val flattenedOutputs = evOutput.outputs(outputs).map(functionGraph.capture)
          (evOutput.decodeOutputFromOutput(outputs, flattenedOutputs)._1, flattenedOutputs)
        }
      }

      // Determine names for the function outputs.
      val outputNamesWithDefault = flattenedOutputs.indices.map(i => s"output_$i")

      (inputNamesWithDefault, outputNamesWithDefault, outputs, flattenedOutputs)
    }

    val extraInputs = functionGraph.extraInputs
    val subFunctions = functionGraph.functions
    inputs.appendAll(functionGraph.extraArgs)

    // Create the native function
    val nativeHandle = NativeFunction.graphToFunction(
      functionGraph.nativeHandle, name, appendHashToFnName = appendHashToName, null,
      inputs.map(_.op.nativeHandle).toArray, inputs.map(_.index).toArray,
      flattenedOutputs.map(_.op.nativeHandle).toArray, flattenedOutputs.map(_.index).toArray,
      outputNamesWithDefault.toArray)
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
    val instantiatedFunction = new InstantiatedFunction[I, O](
      functionName, inputNamesWithDefault, outputNamesWithDefault, outputs,
      subFunctions, extraInputs, functionDef, name, nativeHandleWrapper, closeFn)
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
          Basic.placeholder(output.shape)(TF.fromDataType(output.dataType))
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
    override def apply[T: TF](
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
