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
      function: T => R,
      input: TS,
      outputDataType: RD,
      stateful: Boolean = true,
      name: String = "Callback"
  )(implicit
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
    builder.setAttribute("jvm_pointer", NativeLibrary.currentJvmPointer)
    builder.setAttribute("registry_pointer", NativeLibrary.currentCallbackRegistryPointer)
    builder.setAttribute("Tout", evOutput.dataTypes(outputDataType).toArray)
    builder.addInputList(evInput.outputs(input))
    evOutput.decodeSymbolic(builder.build().outputs.toSeq)
  }
}

/** Contains helpers for dealing with callbacks. */
private[ops] object Callback extends Callback {
  /** Type trait representing valid callback function argument/output types. */
  trait ArgType[T] {
    /** Represents the corresponding symbolic type of `T` where tensors are replaced with their symbolic equivalent. */
    type TS

    /** Represents the corresponding data type structure type of `T` where tensors are replaced with their data types. */
    type TD

    def tensors(arg: T): Seq[Tensor[_]]
    def outputs(arg: TS): Seq[Output]
    def dataTypes(types: TD): Seq[DataType[_]]
    def decode(tensors: Seq[Tensor[_]]): T
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

      override def tensors(arg: Unit): Seq[Tensor[_]] = Seq.empty
      override def outputs(arg: Unit): Seq[Output] = Seq.empty
      override def dataTypes(types: Unit): Seq[DataType[_]] = Seq.empty
      override def decode(tensors: Seq[Tensor[_]]): Unit = ()
      override def decodeSymbolic(outputs: Seq[Output]): Unit = ()
    }

    implicit def tensorArgType[T]: ArgType.Aux[Tensor[T], Output, DataType[T]] = new ArgType[Tensor[T]] {
      override type TS = Output
      override type TD = DataType[T]

      override def tensors(arg: Tensor[T]): Seq[Tensor[_]] = Seq(arg)
      override def outputs(arg: Output): Seq[Output] = Seq(arg)
      override def dataTypes(types: DataType[T]): Seq[DataType[_]] = Seq(types)
      override def decode(tensors: Seq[Tensor[_]]): Tensor[T] = tensors.head.asInstanceOf[Tensor[T]]
      override def decodeSymbolic(outputs: Seq[Output]): Output = outputs.head
    }

    implicit def tensorIndexedSlicesArgType[T]: ArgType.Aux[TensorIndexedSlices[T], OutputIndexedSlices, DataType[T]] = {
      new ArgType[TensorIndexedSlices[T]] {
        override type TS = OutputIndexedSlices
        override type TD = DataType[T]

        override def tensors(arg: TensorIndexedSlices[T]): Seq[Tensor[_]
            ] = Seq(arg.indices, arg.values, arg.denseShape)
        override def outputs(arg: OutputIndexedSlices): Seq[Output] = Seq(arg.indices, arg.values, arg.denseShape)
        override def dataTypes(types: DataType[T]): Seq[DataType[_]] = Seq(INT64, types, INT64)

        override def decode(tensors: Seq[Tensor[_]]): TensorIndexedSlices[T] = {
          TensorIndexedSlices(
            tensors(0).asInstanceOf[Tensor[Long]],
            tensors(1).asInstanceOf[Tensor[T]],
            tensors(2).asInstanceOf[Tensor[Long]])
        }

        override def decodeSymbolic(outputs: Seq[Output]): OutputIndexedSlices = {
          OutputIndexedSlices(outputs(0), outputs(1), outputs(2))
        }
      }
    }

    implicit def sparseTensorArgType[T]: ArgType.Aux[SparseTensor[T], SparseOutput, DataType[T]] = {
      new ArgType[SparseTensor[T]] {
        override type TS = SparseOutput
        override type TD = DataType[T]

        override def tensors(arg: SparseTensor[T]): Seq[Tensor[_]] = Seq(arg.indices, arg.values, arg.denseShape)
        override def outputs(arg: SparseOutput): Seq[Output] = Seq(arg.indices, arg.values, arg.denseShape)
        override def dataTypes(types: DataType[T]): Seq[DataType[_]] = Seq(INT64, types, INT64)

        override def decode(tensors: Seq[Tensor[_]]): SparseTensor[T] = {
          SparseTensor(
            tensors(0).asInstanceOf[Tensor[Long]],
            tensors(1).asInstanceOf[Tensor[T]],
            tensors(2).asInstanceOf[Tensor[Long]])
        }

        override def decodeSymbolic(outputs: Seq[Output]): SparseOutput = {
          SparseOutput(outputs(0), outputs(1), outputs(2))
        }
      }
    }

    implicit def tensorArrayArgType[T]: Aux[Array[Tensor[T]], Array[Output], Array[DataType[T]]] = {
      new ArgType[Array[Tensor[T]]] {
        override type TS = Array[Output]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[Tensor[T]]): Seq[Tensor[_]] = arg.toSeq
        override def outputs(arg: Array[Output]): Seq[Output] = arg.toSeq
        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[_]] = types.toSeq
        override def decode(tensors: Seq[Tensor[_]]): Array[Tensor[T]] = tensors.map(_.asInstanceOf[Tensor[T]]).toArray
        override def decodeSymbolic(outputs: Seq[Output]): Array[Output] = outputs.toArray
      }
    }

    implicit def tensorIndexedSlicesArrayArgType[T]: Aux[Array[TensorIndexedSlices[T]], Array[OutputIndexedSlices], Array[DataType[T]]] = {
      new ArgType[Array[TensorIndexedSlices[T]]] {
        override type TS = Array[OutputIndexedSlices]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[TensorIndexedSlices[T]]): Seq[Tensor[_]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[OutputIndexedSlices]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[_]] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor[_]]): Array[TensorIndexedSlices[T]] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output]): Array[OutputIndexedSlices] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(o(0), o(1), o(2))).toArray
        }
      }
    }

    implicit def sparseTensorArrayArgType[T]: Aux[Array[SparseTensor[T]], Array[SparseOutput], Array[DataType[T]]] = {
      new ArgType[Array[SparseTensor[T]]] {
        override type TS = Array[SparseOutput]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[SparseTensor[T]]): Seq[Tensor[_]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[SparseOutput]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        // TODO: Is INT64 safe here?
        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[_]] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor[_]]): Array[SparseTensor[T]] = {
          tensors.grouped(3).map(t => SparseTensor(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output]): Array[SparseOutput] = {
          outputs.grouped(3).map(o => SparseOutput(o(0), o(1), o(2))).toArray
        }
      }
    }

    implicit def tensorSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[Tensor[T]], Tensor[T], CC[Tensor[T]]],
        cbfOutput: CanBuildFrom[Seq[Output], Output, CC[Output]]
    ): Aux[CC[Tensor[T]], CC[Output], CC[DataType[T]]] = {
      new ArgType[CC[Tensor[T]]] {
        override type TS = CC[Output]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[Tensor[T]]): Seq[Tensor[_]] = arg.toSeq
        override def outputs(arg: CC[Output]): Seq[Output] = arg.toSeq
        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[_]] = types.toSeq
        override def decode(tensors: Seq[Tensor[_]]): CC[Tensor[T]] = tensors.map(_.asInstanceOf[Tensor[T]]).to[CC](cbfTensor)
        override def decodeSymbolic(outputs: Seq[Output]): CC[Output] = outputs.to[CC](cbfOutput)
      }
    }

    implicit def tensorIndexedSlicesSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[TensorIndexedSlices[T]], TensorIndexedSlices[T], CC[TensorIndexedSlices[T]]],
        cbfOutput: CanBuildFrom[Seq[OutputIndexedSlices], OutputIndexedSlices, CC[OutputIndexedSlices]]
    ): Aux[CC[TensorIndexedSlices[T]], CC[OutputIndexedSlices], CC[DataType[T]]] = {
      new ArgType[CC[TensorIndexedSlices[T]]] {
        override type TS = CC[OutputIndexedSlices]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[TensorIndexedSlices[T]]): Seq[Tensor[_]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[OutputIndexedSlices]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[_]] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor[_]]): CC[TensorIndexedSlices[T]] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output]): CC[OutputIndexedSlices] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(o(0), o(1), o(2))).to[CC](cbfOutput)
        }
      }
    }

    implicit def sparseTensorSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[SparseTensor[T]], SparseTensor[T], CC[SparseTensor[T]]],
        cbfOutput: CanBuildFrom[Seq[SparseOutput], SparseOutput, CC[SparseOutput]]
    ): Aux[CC[SparseTensor[T]], CC[SparseOutput], CC[DataType[T]]] = {
      new ArgType[CC[SparseTensor[T]]] {
        override type TS = CC[SparseOutput]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[SparseTensor[T]]): Seq[Tensor[_]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[SparseOutput]): Seq[Output] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[_]] = types.flatMap(Seq(INT64, _, INT64)).toSeq

        override def decode(tensors: Seq[Tensor[_]]): CC[SparseTensor[T]] = {
          tensors.grouped(3).map(t => SparseTensor(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).to[CC](cbfTensor)
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
    *
    *  '''NOTE:''' The input and output tensors of the callback functions are not guaranteed to be copies. In some cases
    *  their underlying memory will be shared with the corresponding TensorFlow session tensors. In-place modification
    *  or storing of the inputs or return values in Scala data structures without explicit copies can have
    *  non-deterministic consequences.
    */
  private[ops] trait Documentation
}
