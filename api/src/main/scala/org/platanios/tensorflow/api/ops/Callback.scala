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

import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.types.{DataType, INT64}
import org.platanios.tensorflow.jni.{ScalaCallbacksRegistry => NativeCallbacksRegistry, TensorFlow => NativeLibrary}

import scala.collection.SeqLike
import scala.collection.generic.CanBuildFrom

/** Contains functions for constructing ops related to Scala callback functions.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Callback {
  /** $OpDocCallbackCallback
    *
    * @group CallbackOps
    * @param  function        Scala function to use for the callback op.
    * @param  input           Input for the created op.
    * @param  outputDataType  Data types of the Scala function outputs.
    * @param  stateful        If `true`, the function should be considered stateful. If a function is stateless, when
    *                         given the same input it will return the same output and have no observable side effects.
    *                         Optimizations such as common subexpression elimination are only performed on stateless
    *                         operations.
    * @param  name            Name for the created op.
    * @tparam T               Scala function input type (e.g., `Tensor`).
    * @tparam TS              Op input type, which is the symbolic type corresponding to `T` (e.g., `Output`).
    * @tparam R               Scala function output type (e.g., `Tensor`).
    * @tparam RS              Op output type, which is the symbolic type corresponding to `R` (e.g., `Output`).
    * @tparam RD              Structure of data types corresponding to `R` (e.g., `DataType`).
    * @return Created op output.
    */
  def callback[T, TS, TD, R, RS, RD](
      function: (T) => R, input: TS, outputDataType: RD, stateful: Boolean = true,
      name: String = "Callback")(implicit
      evInput: Callback.ArgType.Aux[T, TS, TD],
      evOutput: Callback.ArgType.Aux[R, RS, RD]
  ): RS = {
    val id = NativeCallbacksRegistry.register(inputs => {
      val inputTensors = inputs.map(Tensor.fromNativeHandle).toSeq
      val outputs = function(evInput.decode(inputTensors))
      val outputTensors = evOutput.tensors(outputs)
      outputTensors.map(_.nativeHandle).toArray
    })
    // We tie the registered function's lifetime with the current graph. That is, when the current graph is destroyed,
    // we should also deregister its callback functions.
    var graph = Op.currentGraph
    // If the callback function was declared inside a function graph, then its lifetime should be bound to that of the
    // outer graph instead.
    while (graph.isInstanceOf[FunctionGraph])
      graph = graph.asInstanceOf[FunctionGraph].outerGraph
    // When the graph is destroyed, all callback functions used in the graph are de-registered from the native callbacks
    // registry.
    graph.addCleanupFunction(() => NativeCallbacksRegistry.deregister(id))
    val builder = {
      if (stateful)
        Op.Builder(opType = "JVMCallback", name = name)
      else
        Op.Builder(opType = "JVMCallbackStateless", name = name)
    }
    builder.setAttribute("id", id)
    builder.setAttribute("jvm_pointer", NativeLibrary.jvmPointer)
    builder.setAttribute("registry_pointer", NativeLibrary.callbackRegistryPointer)
    builder.setAttribute("registry_call_pointer", NativeLibrary.callbackRegistryCallMethodPointer)
    builder.setAttribute("Tout", evOutput.dataTypes(outputDataType).toArray)
    builder.addInputList(evInput.outputs(input))
    evOutput.decodeSymbolic(builder.build().outputs.toSeq)
  }
}

/** Contains helpers for dealing with callbacks. */
private[ops] object Callback extends Callback {
  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("JVMCallback")
    GradientsRegistry.registerNonDifferentiable("JVMCallbackStateless")
  }

  /** Type trait representing valid callback function argument/output types. */
  trait ArgType[T] {
    /** Represents the corresponding symbolic type of `T` where tensors are replaced with their symbolic equivalent. */
    type TS

    /** Represents the corresponding data type structure type of `T` where tensors are replaced with their data types. */
    type TD

    def tensors(arg: T): Seq[Tensor]
    def outputs(arg: TS): Seq[Output]
    def dataTypes(types: TD): Seq[DataType]
    def decode(tensors: Seq[Tensor]): T
    def decodeSymbolic(outputs: Seq[Output]): TS
  }

  /** Contains implicits for the [[ArgType]] type trait. */
  object ArgType {
    // TODO: [CALLBACKS] Find a way to make the implicits here more elegant and maybe generalizable.
    // TODO: [CALLBACKS] Add support for tuples of outputs, etc., as arg types.

    type Aux[T, S, D] = ArgType[T] {
      type TS = S
      type TD = D
    }

    def apply[T, S, D](implicit ev: Aux[T, S, D]): Aux[T, S, D] = ev

    implicit val unitArgType: ArgType.Aux[Unit, Unit, Unit] = new ArgType[Unit] {
      override type TS = Unit
      override type TD = Unit

      override def tensors(arg: Unit): Seq[Tensor] = Seq.empty
      override def outputs(arg: Unit): Seq[Output] = Seq.empty
      override def dataTypes(types: Unit): Seq[DataType] = Seq.empty
      override def decode(tensors: Seq[Tensor]): Unit = ()
      override def decodeSymbolic(outputs: Seq[Output]): Unit = ()
    }

    implicit def tensorArgType[D <: DataType]: ArgType.Aux[Tensor, Output, D] = new ArgType[Tensor] {
      override type TS = Output
      override type TD = D

      override def tensors(arg: Tensor): Seq[Tensor] = Seq(arg)
      override def outputs(arg: Output): Seq[Output] = Seq(arg)
      override def dataTypes(types: D): Seq[DataType] = Seq(types)
      override def decode(tensors: Seq[Tensor]): Tensor = tensors.head
      override def decodeSymbolic(outputs: Seq[Output]): Output = outputs.head
    }

    implicit def tensorIndexedSlicesArgType[D <: DataType]: ArgType.Aux[TensorIndexedSlices, OutputIndexedSlices, D] = {
      new ArgType[TensorIndexedSlices] {
        override type TS = OutputIndexedSlices
        override type TD = D

        override def tensors(arg: TensorIndexedSlices): Seq[Tensor] = Seq(arg.indices, arg.values, arg.denseShape)
        override def outputs(arg: OutputIndexedSlices): Seq[Output] = Seq(arg.indices, arg.values, arg.denseShape)
        // TODO: Is INT64 safe here?
        override def dataTypes(types: D): Seq[DataType] = Seq(INT64, types, INT64)

        override def decode(tensors: Seq[Tensor]): TensorIndexedSlices = {
          TensorIndexedSlices(tensors(0), tensors(1), tensors(2))
        }

        override def decodeSymbolic(outputs: Seq[Output]): OutputIndexedSlices = {
          OutputIndexedSlices(outputs(0), outputs(1), outputs(2))
        }
      }
    }

    implicit def sparseTensorArgType[D <: DataType]: ArgType.Aux[SparseTensor, SparseOutput, D] = {
      new ArgType[SparseTensor] {
        override type TS = SparseOutput
        override type TD = D

        override def tensors(arg: SparseTensor): Seq[Tensor] = Seq(arg.indices, arg.values, arg.denseShape)
        override def outputs(arg: SparseOutput): Seq[Output] = Seq(arg.indices, arg.values, arg.denseShape)
        // TODO: Is INT64 safe here?
        override def dataTypes(types: D): Seq[DataType] = Seq(INT64, types, INT64)

        override def decode(tensors: Seq[Tensor]): SparseTensor = {
          SparseTensor(tensors(0), tensors(1), tensors(2))
        }

        override def decodeSymbolic(outputs: Seq[Output]): SparseOutput = {
          SparseOutput(outputs(0), outputs(1), outputs(2))
        }
      }
    }

    implicit def tensorArrayArgType[D <: DataType]: Aux[Array[Tensor], Array[Output], Array[D]] = {
      new ArgType[Array[Tensor]] {
        override type TS = Array[Output]
        override type TD = Array[D]

        override def tensors(arg: Array[Tensor]): Seq[Tensor] = arg.toSeq
        override def outputs(arg: Array[Output]): Seq[Output] = arg.toSeq
        override def dataTypes(types: Array[D]): Seq[DataType] = types.toSeq
        override def decode(tensors: Seq[Tensor]): Array[Tensor] = tensors.toArray
        override def decodeSymbolic(outputs: Seq[Output]): Array[Output] = outputs.toArray
      }
    }

    implicit def tensorIndexedSlicesArrayArgType[D <: DataType]
    : Aux[Array[TensorIndexedSlices], Array[OutputIndexedSlices], Array[D]] = {
      new ArgType[Array[TensorIndexedSlices]] {
        override type TS = Array[OutputIndexedSlices]
        override type TD = Array[D]

        override def tensors(arg: Array[TensorIndexedSlices]): Seq[Tensor] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[OutputIndexedSlices]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: Array[D]): Seq[DataType] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor]): Array[TensorIndexedSlices] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(t(0), t(1), t(2))).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output]): Array[OutputIndexedSlices] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(o(0), o(1), o(2))).toArray
        }
      }
    }

    implicit def sparseTensorArrayArgType[D <: DataType]: Aux[Array[SparseTensor], Array[SparseOutput], Array[D]] = {
      new ArgType[Array[SparseTensor]] {
        override type TS = Array[SparseOutput]
        override type TD = Array[D]

        override def tensors(arg: Array[SparseTensor]): Seq[Tensor] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[SparseOutput]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: Array[D]): Seq[DataType] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor]): Array[SparseTensor] = {
          tensors.grouped(3).map(t => SparseTensor(t(0), t(1), t(2))).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output]): Array[SparseOutput] = {
          outputs.grouped(3).map(o => SparseOutput(o(0), o(1), o(2))).toArray
        }
      }
    }

    implicit def tensorSeqArgType[D <: DataType, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[Tensor], Tensor, CC[Tensor]],
        cbfOutput: CanBuildFrom[Seq[Output], Output, CC[Output]]
    ): Aux[CC[Tensor], CC[Output], CC[D]] = {
      new ArgType[CC[Tensor]] {
        override type TS = CC[Output]
        override type TD = CC[D]

        override def tensors(arg: CC[Tensor]): Seq[Tensor] = arg.toSeq
        override def outputs(arg: CC[Output]): Seq[Output] = arg.toSeq
        override def dataTypes(types: CC[D]): Seq[DataType] = types.toSeq
        override def decode(tensors: Seq[Tensor]): CC[Tensor] = tensors.to[CC](cbfTensor)
        override def decodeSymbolic(outputs: Seq[Output]): CC[Output] = outputs.to[CC](cbfOutput)
      }
    }

    implicit def tensorIndexedSlicesSeqArgType[D <: DataType, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[TensorIndexedSlices], TensorIndexedSlices, CC[TensorIndexedSlices]],
        cbfOutput: CanBuildFrom[Seq[OutputIndexedSlices], OutputIndexedSlices, CC[OutputIndexedSlices]]
    ): Aux[CC[TensorIndexedSlices], CC[OutputIndexedSlices], CC[D]] = {
      new ArgType[CC[TensorIndexedSlices]] {
        override type TS = CC[OutputIndexedSlices]
        override type TD = CC[D]

        override def tensors(arg: CC[TensorIndexedSlices]): Seq[Tensor] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[OutputIndexedSlices]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: CC[D]): Seq[DataType] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor]): CC[TensorIndexedSlices] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(t(0), t(1), t(2))).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output]): CC[OutputIndexedSlices] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(o(0), o(1), o(2))).to[CC](cbfOutput)
        }
      }
    }

    implicit def sparseTensorSeqArgType[D <: DataType, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[SparseTensor], SparseTensor, CC[SparseTensor]],
        cbfOutput: CanBuildFrom[Seq[SparseOutput], SparseOutput, CC[SparseOutput]]
    ): Aux[CC[SparseTensor], CC[SparseOutput], CC[D]] = {
      new ArgType[CC[SparseTensor]] {
        override type TS = CC[SparseOutput]
        override type TD = CC[D]

        override def tensors(arg: CC[SparseTensor]): Seq[Tensor] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[SparseOutput]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: CC[D]): Seq[DataType] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor]): CC[SparseTensor] = {
          tensors.grouped(3).map(t => SparseTensor(t(0), t(1), t(2))).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output]): CC[SparseOutput] = {
          outputs.grouped(3).map(o => SparseOutput(o(0), o(1), o(2))).to[CC](cbfOutput)
        }
      }
    }
  }

  /** @define OpDocCallbackCallback
    *  The `callback` op wraps a Scala function and uses it as a TensorFlow op.
    *
    *  Given a Scala function `function`, which takes an arbitrary structure of tensors as its input and returns another
    *  arbitrary structure of tensors as its output, the op wraps this function as an op in a TensorFlow graph. The
    *  following snippet constructs a simple TensorFlow graph that invokes the imperative `Tensor.sinh` function as an
    *  op in the graph:
    *
    *  {{{
    *    def customFunction(x: Tensor): Tensor = x.sinh
    *
    *    val input = tf.placeholder(FLOAT32)
    *    val output = tf.callback(customFunction, input, FLOAT32)
    *  }}}
    *
    *  '''NOTE:''' The `callback` op has the following known limitations:
    *  - The body of the Scala function (i.e. `function`) will not be serialized in a `GraphDef`. Therefore, you should
    *    not use this op if you need to serialize your model and restore it in a different environment.
    *  - The op must be able to access the JVM instance that the Scala program that constructed it was running on. This
    *    can be important if you are using distributed TensorFlow.
    */
  private[ops] trait Documentation
}
