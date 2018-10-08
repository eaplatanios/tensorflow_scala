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
import org.platanios.tensorflow.api.core.types.{DataType, INT64}
import org.platanios.tensorflow.jni.{ScalaCallbacksRegistry => NativeCallbacksRegistry, TensorFlow => NativeLibrary}

import scala.collection.SeqLike
import scala.collection.generic.CanBuildFrom

/** Contains functions for constructing ops related to Scala callback functions.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Callback {
  /** $OpDocCallbackCallback
    *
    * @group CallbackOps
    * @param  function       Scala function to use for the callback op.
    * @param  input          Input for the created op.
    * @param  outputDataType Data types of the Scala function outputs.
    * @param  stateful       If `true`, the function should be considered stateful. If a function is stateless, when
    *                        given the same input it will return the same output and have no observable side effects.
    *                        Optimizations such as common subexpression elimination are only performed on stateless
    *                        operations.
    * @param  name           Name for the created op.
    * @tparam IT             Scala function input type (e.g., `Tensor`).
    * @tparam IO             Op input type, which is the symbolic type corresponding to `T` (e.g., `Output`).
    * @tparam OT             Scala function output type (e.g., `Tensor`).
    * @tparam OO             Op output type, which is the symbolic type corresponding to `R` (e.g., `Output`).
    * @tparam OD             Structure of data types corresponding to `R` (e.g., `DataType`).
    * @return Created op output.
    */
  def callback[IT, IO, ID, OT, OO, OD](
      function: IT => OT,
      input: IO,
      outputDataType: OD,
      stateful: Boolean = true,
      name: String = "Callback"
  )(implicit
      evInput: Callback.ArgType.Aux[IT, IO, ID],
      evOutput: Callback.ArgType.Aux[OT, OO, OD]
  ): OO = {
    val id = NativeCallbacksRegistry.register(inputs => {
      val inputTensors = inputs.map(Tensor.fromNativeHandle[Any]).toSeq
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
      if (stateful) {
        Op.Builder[Seq[Output[Any]], Seq[Output[Any]]](
          opType = "JVMCallback",
          name = name,
          input = evInput.outputs(input))
      } else {
        Op.Builder[Seq[Output[Any]], Seq[Output[Any]]](
          opType = "JVMCallbackStateless",
          name = name,
          input = evInput.outputs(input))
      }
    }
    builder.setAttribute("id", id)
    builder.setAttribute("jvm_pointer", NativeLibrary.currentJvmPointer)
    builder.setAttribute("registry_pointer", NativeLibrary.currentCallbackRegistryPointer)
    builder.setAttribute("Tout", evOutput.dataTypes(outputDataType).toArray)
    evOutput.decodeSymbolic(builder.build().output)
  }
}

/** Contains helpers for dealing with callbacks. */
object Callback extends Callback {
  /** Type trait representing valid callback function argument/output types. */
  trait ArgType[T] {
    /** Represents the corresponding symbolic type of `T` where tensors are replaced with their symbolic equivalent. */
    type TO

    /** Represents the corresponding data type structure type of `T` where tensors are replaced with their data types. */
    type TD

    def tensors(arg: T): Seq[Tensor[Any]]
    def outputs(arg: TO): Seq[Output[Any]]
    def dataTypes(types: TD): Seq[DataType[Any]]
    def decode(tensors: Seq[Tensor[Any]]): T
    def decodeSymbolic(outputs: Seq[Output[Any]]): TO
  }

  /** Contains implicits for the [[ArgType]] type trait. */
  object ArgType {
    // TODO: [CALLBACKS] Find a way to make the implicits here more elegant and maybe generalizable.
    // TODO: [CALLBACKS] Add support for tuples of outputs, etc., as arg types.

    type Aux[T, O, D] = ArgType[T] {
      type TO = O
      type TD = D
    }

    def apply[T, O, D](implicit ev: Aux[T, O, D]): Aux[T, O, D] = ev

    implicit val unitArgType: ArgType.Aux[Unit, Unit, Unit] = new ArgType[Unit] {
      override type TO = Unit
      override type TD = Unit

      override def tensors(arg: Unit): Seq[Tensor[Any]] = {
        Seq.empty
      }

      override def outputs(arg: Unit): Seq[Output[Any]] = {
        Seq.empty
      }

      override def dataTypes(types: Unit): Seq[DataType[Any]] = {
        Seq.empty
      }

      override def decode(tensors: Seq[Tensor[Any]]): Unit = {
        ()
      }

      override def decodeSymbolic(outputs: Seq[Output[Any]]): Unit = {
        ()
      }
    }

    implicit def tensorArgType[T]: ArgType.Aux[Tensor[T], Output[T], DataType[T]] = {
      new ArgType[Tensor[T]] {
        override type TO = Output[T]
        override type TD = DataType[T]

        override def tensors(arg: Tensor[T]): Seq[Tensor[Any]] = {
          Seq(arg)
        }

        override def outputs(arg: Output[T]): Seq[Output[Any]] = {
          Seq(arg)
        }

        override def dataTypes(types: DataType[T]): Seq[DataType[Any]] = {
          Seq(types)
        }

        override def decode(tensors: Seq[Tensor[Any]]): Tensor[T] = {
          tensors.head.asInstanceOf[Tensor[T]]
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): Output[T] = {
          outputs.head.asInstanceOf[Output[T]]
        }
      }
    }

    implicit def tensorIndexedSlicesArgType[T]: ArgType.Aux[TensorIndexedSlices[T], OutputIndexedSlices[T], DataType[T]] = {
      new ArgType[TensorIndexedSlices[T]] {
        override type TO = OutputIndexedSlices[T]
        override type TD = DataType[T]

        override def tensors(arg: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def outputs(arg: OutputIndexedSlices[T]): Seq[Output[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def dataTypes(types: DataType[T]): Seq[DataType[Any]] = {
          Seq(INT64, types, INT64)
        }

        override def decode(tensors: Seq[Tensor[Any]]): TensorIndexedSlices[T] = {
          TensorIndexedSlices(
            tensors(0).asInstanceOf[Tensor[Long]],
            tensors(1).asInstanceOf[Tensor[T]],
            tensors(2).asInstanceOf[Tensor[Long]])
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): OutputIndexedSlices[T] = {
          OutputIndexedSlices(
            outputs(0).asInstanceOf[Tensor[Long]],
            outputs(1).asInstanceOf[Tensor[T]].toOutput,
            outputs(2).asInstanceOf[Tensor[Long]])
        }
      }
    }

    implicit def sparseTensorArgType[T]: ArgType.Aux[SparseTensor[T], SparseOutput[T], DataType[T]] = {
      new ArgType[SparseTensor[T]] {
        override type TO = SparseOutput[T]
        override type TD = DataType[T]

        override def tensors(arg: SparseTensor[T]): Seq[Tensor[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def outputs(arg: SparseOutput[T]): Seq[Output[Any]] = {
          Seq(arg.indices, arg.values, arg.denseShape)
        }

        override def dataTypes(types: DataType[T]): Seq[DataType[Any]] = {
          Seq(INT64, types, INT64)
        }

        override def decode(tensors: Seq[Tensor[Any]]): SparseTensor[T] = {
          SparseTensor(
            tensors(0).asInstanceOf[Tensor[Long]],
            tensors(1).asInstanceOf[Tensor[T]],
            tensors(2).asInstanceOf[Tensor[Long]])
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): SparseOutput[T] = {
          SparseOutput(
            outputs(0).asInstanceOf[Tensor[Long]],
            outputs(1).asInstanceOf[Tensor[T]].toOutput,
            outputs(2).asInstanceOf[Tensor[Long]])
        }
      }
    }

    implicit def tensorArrayArgType[T]: Aux[Array[Tensor[T]], Array[Output[T]], Array[DataType[T]]] = {
      new ArgType[Array[Tensor[T]]] {
        override type TO = Array[Output[T]]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[Tensor[T]]): Seq[Tensor[Any]] = {
          arg.toSeq
        }

        override def outputs(arg: Array[Output[T]]): Seq[Output[Any]] = {
          arg.toSeq
        }

        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[Any]] = {
          types.toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): Array[Tensor[T]] = {
          tensors.map(_.asInstanceOf[Tensor[T]]).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): Array[Output[T]] = {
          outputs.map(_.asInstanceOf[Output[T]]).toArray
        }
      }
    }

    implicit def tensorIndexedSlicesArrayArgType[T]: Aux[Array[TensorIndexedSlices[T]], Array[OutputIndexedSlices[T]], Array[DataType[T]]] = {
      new ArgType[Array[TensorIndexedSlices[T]]] {
        override type TO = Array[OutputIndexedSlices[T]]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[TensorIndexedSlices[T]]): Seq[Tensor[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[OutputIndexedSlices[T]]): Seq[Output[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[Any]] = {
          types.flatMap(Seq(INT64, _, INT64)).toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): Array[TensorIndexedSlices[T]] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): Array[OutputIndexedSlices[T]] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(
            o(0).asInstanceOf[Tensor[Long]],
            o(1).asInstanceOf[Tensor[T]].toOutput,
            o(2).asInstanceOf[Tensor[Long]])).toArray
        }
      }
    }

    implicit def sparseTensorArrayArgType[T]: Aux[Array[SparseTensor[T]], Array[SparseOutput[T]], Array[DataType[T]]] = {
      new ArgType[Array[SparseTensor[T]]] {
        override type TO = Array[SparseOutput[T]]
        override type TD = Array[DataType[T]]

        override def tensors(arg: Array[SparseTensor[T]]): Seq[Tensor[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: Array[SparseOutput[T]]): Seq[Output[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: Array[DataType[T]]): Seq[DataType[Any]] = {
          types.flatMap(Seq(INT64, _, INT64)).toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): Array[SparseTensor[T]] = {
          tensors.grouped(3).map(t => SparseTensor(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).toArray
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): Array[SparseOutput[T]] = {
          outputs.grouped(3).map(o => SparseOutput(
            o(0).asInstanceOf[Tensor[Long]],
            o(1).asInstanceOf[Tensor[T]].toOutput,
            o(2).asInstanceOf[Tensor[Long]])).toArray
        }
      }
    }

    implicit def tensorSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[Tensor[T]], Tensor[T], CC[Tensor[T]]],
        cbfOutput: CanBuildFrom[Seq[Output[T]], Output[T], CC[Output[T]]]
    ): Aux[CC[Tensor[T]], CC[Output[T]], CC[DataType[T]]] = {
      new ArgType[CC[Tensor[T]]] {
        override type TO = CC[Output[T]]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[Tensor[T]]): Seq[Tensor[Any]] = {
          arg.toSeq
        }

        override def outputs(arg: CC[Output[T]]): Seq[Output[Any]] = {
          arg.toSeq
        }

        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[Any]] = {
          types.toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): CC[Tensor[T]] = {
          tensors.map(_.asInstanceOf[Tensor[T]]).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): CC[Output[T]] = {
          outputs.map(_.asInstanceOf[Output[T]]).to[CC](cbfOutput)
        }
      }
    }

    implicit def tensorIndexedSlicesSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[TensorIndexedSlices[T]], TensorIndexedSlices[T], CC[TensorIndexedSlices[T]]],
        cbfOutput: CanBuildFrom[Seq[OutputIndexedSlices[T]], OutputIndexedSlices[T], CC[OutputIndexedSlices[T]]]
    ): Aux[CC[TensorIndexedSlices[T]], CC[OutputIndexedSlices[T]], CC[DataType[T]]] = {
      new ArgType[CC[TensorIndexedSlices[T]]] {
        override type TO = CC[OutputIndexedSlices[T]]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[TensorIndexedSlices[T]]): Seq[Tensor[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[OutputIndexedSlices[T]]): Seq[Output[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[Any]] = {
          types.flatMap(Seq(INT64, _, INT64)).toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): CC[TensorIndexedSlices[T]] = {
          tensors.grouped(3).map(t => TensorIndexedSlices(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): CC[OutputIndexedSlices[T]] = {
          outputs.grouped(3).map(o => OutputIndexedSlices(
            o(0).asInstanceOf[Tensor[Long]],
            o(1).asInstanceOf[Tensor[T]].toOutput,
            o(2).asInstanceOf[Tensor[Long]])).to[CC](cbfOutput)
        }
      }
    }

    implicit def sparseTensorSeqArgType[T, CC[A] <: SeqLike[A, CC[A]]](implicit
        cbfTensor: CanBuildFrom[Seq[SparseTensor[T]], SparseTensor[T], CC[SparseTensor[T]]],
        cbfOutput: CanBuildFrom[Seq[SparseOutput[T]], SparseOutput[T], CC[SparseOutput[T]]]
    ): Aux[CC[SparseTensor[T]], CC[SparseOutput[T]], CC[DataType[T]]] = {
      new ArgType[CC[SparseTensor[T]]] {
        override type TO = CC[SparseOutput[T]]
        override type TD = CC[DataType[T]]

        override def tensors(arg: CC[SparseTensor[T]]): Seq[Tensor[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def outputs(arg: CC[SparseOutput[T]]): Seq[Output[Any]] = {
          arg.flatMap(a => Seq(a.indices, a.values, a.denseShape)).toSeq
        }

        override def dataTypes(types: CC[DataType[T]]): Seq[DataType[Any]] = {
          types.flatMap(Seq(INT64, _, INT64)).toSeq
        }

        override def decode(tensors: Seq[Tensor[Any]]): CC[SparseTensor[T]] = {
          tensors.grouped(3).map(t => SparseTensor(
            t(0).asInstanceOf[Tensor[Long]],
            t(1).asInstanceOf[Tensor[T]],
            t(2).asInstanceOf[Tensor[Long]])).to[CC](cbfTensor)
        }

        override def decodeSymbolic(outputs: Seq[Output[Any]]): CC[SparseOutput[T]] = {
          outputs.grouped(3).map(o => SparseOutput(
            o(0).asInstanceOf[Tensor[Long]],
            o(1).asInstanceOf[Tensor[T]].toOutput,
            o(2).asInstanceOf[Tensor[Long]])).to[CC](cbfOutput)
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
