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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.io.data.{Data, Dataset}
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, VARIANT}
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Function => NativeFunction, Graph => NativeGraph}

import org.tensorflow.framework.FunctionDef
import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.mutable
import scala.util.DynamicVariable

// TODO: [FUNCTIONS] !!! Unique name using the hashing functionality.
// TODO: [FUNCTIONS] !!! Add support for function descriptions.

/**
  * @author Emmanouil Antonios Platanios
  */
case class Function[IO, IR, R](name: String, function: (IO) => R)(implicit
    evInput: Function.ArgType[IO],
    evOutput: Function.ArgType[R]
) {
  private[this] val instantiatedFunctions = mutable.HashMap.empty[String, InstantiatedFunction[IO, R]]

  def apply(arg: IO): R = {
    val dataTypes = evInput.dataTypes(arg)
    val key = dataTypes.map(_.toString).mkString(":")
    instantiatedFunctions.getOrElseUpdate(key, {
      InstantiatedFunction(s"${name}_$key", function, dataTypes)(evInput, evOutput)
    })(arg)
  }

  private[ops] def instantiate(
      inputDataTypes: Seq[DataType], inputShapes: Seq[Shape] = null): InstantiatedFunction[IO, R] = {
    val key = (inputDataTypes.map(_.toString) ++ Option(inputShapes).map(_.map(_.toString))).mkString(":")
    instantiatedFunctions.getOrElseUpdate(key, {
      InstantiatedFunction(s"${name}_$key", function, inputDataTypes, Option(inputShapes))(evInput, evOutput)
    })
  }
}

object Function {
  trait ArgType[O] {
    def numOutputs: Int
    def outputs(arg: O): Seq[Output]
    def dataTypes(arg: O): Seq[DataType]
    def outputsDecoder(outputs: Seq[Output]): (O, Seq[Output])
  }

  object ArgType {
    def apply[O](implicit ev: ArgType[O]): ArgType[O] = ev

    implicit val outputArgType: ArgType[Output] = new ArgType[Output] {
      override def numOutputs: Int = 1
      override def outputs(arg: Output): Seq[Output] = Seq(arg)
      override def dataTypes(arg: Output): Seq[DataType] = Seq(arg.dataType)
      override def outputsDecoder(outputs: Seq[Output]): (Output, Seq[Output]) = (outputs.head, outputs.tail)
    }

    implicit def datasetArgType[T, O, D, S](implicit
        ev: Data.Aux[T, O, D, S]
    ): ArgType[Dataset[T, O, D, S]] = new ArgType[Dataset[T, O, D, S]] {
      override def numOutputs: Int = 1
      override def outputs(arg: Dataset[T, O, D, S]): Seq[Output] = Seq(arg.createHandle())
      override def dataTypes(arg: Dataset[T, O, D, S]): Seq[DataType] = Seq(VARIANT)
      // TODO: [FUNCTIONS] !!! Find a better way to deal with this -- it's only used for Dataset.flatMap().
      override def outputsDecoder(outputs: Seq[Output]): (Dataset[T, O, D, S], Seq[Output]) = (null, outputs.tail)
    }

    implicit val hnil: ArgType[HNil] = new ArgType[HNil] {
      override def numOutputs: Int = 0
      override def outputs(arg: HNil): Seq[Output] = Seq.empty[Output]
      override def dataTypes(arg: HNil): Seq[DataType] = Seq.empty[DataType]
      override def outputsDecoder(outputs: Seq[Output]): (HNil, Seq[Output]) = (HNil, outputs.tail)
    }

    implicit def recursiveConstructor[H, T <: HList](implicit
        argTypeHead: Lazy[ArgType[H]],
        argTypeTail: ArgType[T]
    ): ArgType[H :: T] = new ArgType[H :: T] {
      override def numOutputs: Int = argTypeHead.value.numOutputs + argTypeTail.numOutputs

      override def outputs(arg: H :: T): Seq[Output] = {
        argTypeHead.value.outputs(arg.head) ++ argTypeTail.outputs(arg.tail)
      }

      override def dataTypes(arg: H :: T): Seq[DataType] = {
        argTypeHead.value.dataTypes(arg.head) ++ argTypeTail.dataTypes(arg.tail)
      }

      override def outputsDecoder(outputs: Seq[Output]): (H :: T, Seq[Output]) = {
        val (decodedHead, outputsTail) = argTypeHead.value.outputsDecoder(outputs)
        val (decodedTail, tail) = argTypeTail.outputsDecoder(outputsTail)
        (decodedHead :: decodedTail, tail)
      }
    }

    // This also covers `OutputIndexedSlices` and `SparseOutput` as they are case classes (i.e., products).
    implicit def productConstructor[P <: Product, L <: HList](implicit
        gen: Generic.Aux[P, L],
        argTypeL: ArgType[L],
        tupler: Tupler.Aux[L, P]
    ): ArgType[P] = new ArgType[P] {
      override def numOutputs: Int = argTypeL.numOutputs
      override def outputs(arg: P): Seq[Output] = argTypeL.outputs(gen.to(arg))
      override def dataTypes(arg: P): Seq[DataType] = argTypeL.dataTypes(gen.to(arg))
      override def outputsDecoder(outputs: Seq[Output]): (P, Seq[Output]) = {
        val (decoded, tail) = argTypeL.outputsDecoder(outputs)
        (tupler(decoded), tail)
      }
    }
  }
}

private[api] case class InstantiatedFunction[O, R] private[ops] (
    name: String, function: (O) => R,
    inputDataTypes: Seq[DataType],
    inputShapes: Option[Seq[Shape]] = None,
    private val _inputNames: Seq[String] = null,
    private val _outputNames: Seq[String] = null)(implicit
    evInput: Function.ArgType[O],
    evOutput: Function.ArgType[R]
) extends Closeable {
  require(inputDataTypes.length == evInput.numOutputs,
          s"The number of 'inputDataTypes' provided (${inputDataTypes.length}) " +
              s"does not match the number of inputs (${evInput.numOutputs}).")

  // The following code block initializes this function
  private[this] val initializationOutput = {
    // List of placeholders for the function definition
    val inputs = mutable.ListBuffer.empty[Output]
    val functionGraph = FunctionGraph()
    val (inputNames, outputNames, outputs, flattenedOutputs) = Op.createWith(functionGraph) {
      // Determine names for the function inputs
      val inputNames = {
        if (_inputNames != null) {
          require(_inputNames.length == inputDataTypes.length,
                  s"The number of 'inputNames' provided (${_inputNames.length}) " +
                      s"does not match the number of inputs (${inputDataTypes.length}).")
          _inputNames
        } else {
          inputDataTypes.indices.map(i => s"input_$i")
        }
      }

      inputShapes match {
        case None =>
          inputs.appendAll((inputNames, inputDataTypes).zipped.map((name, dataType) => {
            Basic.placeholder(dataType = dataType, name = name)
          }))
        case Some(shapes) =>
          inputs.appendAll((inputNames, inputDataTypes, shapes).zipped.map((name, dataType, shape) => {
            Basic.placeholder(dataType = dataType, shape = shape, name = name)
          }))
      }

      // Call the Scala function and gather the output tensors
      val (outputs, flattenedOutputs) = {
        VariableScope.createWithVariableScope("", customGetter = functionGraph.customVariableGetter) {
          // Unflatten the inputs, pass them to the function, and then flatten the returned outputs
          val outputs = function(evInput.outputsDecoder(inputs)._1)
          val flattenedOutputs = evOutput.outputs(outputs)
          (outputs, flattenedOutputs)
        }
      }

      // Determine names for the function outputs
      val outputNames = {
        if (_outputNames != null) {
          require(_outputNames.length == evOutput.numOutputs,
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
    // TODO: [FUNCTIONS] !!! Use the name hashing functionality.
    val nativeHandle = NativeFunction.graphToFunction(
      functionGraph.nativeHandle, name, appendHashToFnName = false, null,
      inputs.map(_.op.nativeHandle).toArray, inputs.map(_.index).toArray,
      flattenedOutputs.map(_.op.nativeHandle).toArray, flattenedOutputs.map(_.index).toArray,
      outputNames.toArray)
    (inputNames, outputNames, outputs, flattenedOutputs, subFunctions, extraInputs, nativeHandle)
  }

  /** Names of the function inputs. */
  val inputNames: Seq[String] = initializationOutput._1

  /** Names of the function outputs. */
  val outputNames: Seq[String] = initializationOutput._2

  /** Data types of the function outputs. */
  val outputDataTypes: Seq[DataType] = initializationOutput._4.map(_.dataType)

  /** Shapes of the function outputs. */
  val outputShapes: Seq[Shape] = initializationOutput._4.map(_.shape)

  /** Dummy outputs used to store the structure information of the output type. */
  private[ops] val dummyOutputs: R = initializationOutput._3

  /** Functions defined in the graph used while creating this function. These functions will be added to all graphs
    * where this function is added to. */
  private[this] val subFunctions = initializationOutput._5

  /** Extra inputs to feed to the function as arguments when calling it, which correspond to the values of op outputs
    * that are used in the function, btu which belong to a different graph, than the function graph. */
  private[ops] val extraInputs = initializationOutput._6

  /** Lock for the native handle. */
  private[this] object NativeHandleLock

  /** Handle for the underlying native function object. */
  private[this] var _nativeHandle = initializationOutput._7

  private[api] def nativeHandle: Long = _nativeHandle

  // Keep track of references in the Scala side and notify the native library when the function is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Adds this function to the provided graph. */
  def addToGraph(graph: Graph): Unit = graph.getFunction(name) match {
    case Some(_) => ()
    case None =>
      // Add this function into the graph
      graph.addFunction(this)
      // Ensure that all sub-functions are also added to the graph
      subFunctions.foreach(graph.addFunction)
    // TODO: [FUNCTIONS] Add the gradient function too.
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
      input: O, inline: Boolean = true, compiled: Boolean = false, separateCompiledGradients: Boolean = false,
      name: String = this.name)(implicit context: DynamicVariable[OpCreationContext]): R = {
    val outputs = Op.createWithNameScope(name) {
      val outputs = evInput.outputs(input)
      addToGraph(outputs.head.graph)
      val builder = Op.Builder(opType = this.name, name = "Call")(context)
      (outputs ++ extraInputs).foreach(builder.addInput)
      builder.setAttribute("_noinline", inline)
      if (compiled) {
        builder
            .setAttribute("_XlaCompile", compiled)
            .setAttribute("_XlaSeparateCompiledGradients", separateCompiledGradients)
            .setAttribute("_XlaScope", context.value.attributes.getOrElse("_XlaScope", s"function_$name").toString)
      }
      builder.build().outputs
    }
    evOutput.outputsDecoder(outputs)._1
  }

  /** Constructs and returns a [[FunctionDef]] object, which is a serialized version of this function. */
  def toFunctionDef: FunctionDef = {
    FunctionDef.parseFrom(NativeHandleLock.synchronized(NativeFunction.toFunctionDef(_nativeHandle)))
  }

  /** Releases the native resources associated with this function instance. */
  override def close(): Unit = NativeHandleLock.synchronized {
    if (_nativeHandle != 0) {
      NativeFunction.delete(_nativeHandle)
      _nativeHandle = 0
    }
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
  * @author Emmanouil Antonios Platanios
  */
class FunctionGraph(private[this] val _nativeHandle: Long) extends Graph(_nativeHandle) {
  /** Graph used during construction of this graph. */
  private[ops] val outerGraph = Op.currentGraph

  /** Variable scope used during construction of this graph. */
  private[ops] val outerVariableScope = Op.currentVariableScope

  /** Captured op outputs that belong to other graphs and are used within this graph. */
  private[ops] val capturedOutputs = mutable.HashMap.empty[Output, Output]

  /** Extra placeholder arguments to feed to the function being created, that represent captured op output values. */
  private[ops] val extraArgs = mutable.ListBuffer.empty[Output]

  /** Extra inputs to use when calling the function being created, that correspond to the extra arguments. */
  private[ops] val extraInputs = mutable.ListBuffer.empty[Output]

  /** Extra variables that have been created on the outer graph and correspond to those created within this graph. */
  private[ops] val extraVars = mutable.ListBuffer.empty[Output]

  /** Helper function for processing tensors before using them as inputs for ops placed in this graph. Useful for
    * creating function graphs. */
  override private[api] def processOpInput(output: Output): Output = {
    if (output.graph == this) {
      output
    } else {
      // Referring to a tensor from other graph
      capturedOutputs.getOrElseUpdate(output, {
        // Substitute with a placeholder
        val placeholder = Basic.placeholder(output.dataType, output.shape)
        extraArgs.append(placeholder)
        extraInputs.append(output)
        placeholder
      })
    }
  }

  /** Custom variable getter for variables created within this function graph. */
  private[ops] val customVariableGetter = new VariableGetter {
    override def apply(name: String, dataType: DataType = FLOAT32, shape: Shape = null, initializer: Initializer = null,
        regularizer: Regularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null): Variable = {
      // TODO: [FUNCTIONS] !!! Not sure if this works as it should. Especially the '.value' method of resource variables.
      // Here, we switch the default graph to the outer graph and ask the variable scope in which the function is defined
      // to give us the variable. The variable is stashed in extra_vars and returned to the caller. We capture these
      // variables so that the variable definition is hoisted upward to the outer-most graph.
      Op.createWith(outerGraph) {
        val variable = outerVariableScope.getVariable(
          store = Op.currentVariableStore,
          name = name,
          dataType = dataType,
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
  def apply(): FunctionGraph = new FunctionGraph(NativeGraph.allocate())
}
