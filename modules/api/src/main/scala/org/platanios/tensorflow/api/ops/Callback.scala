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

import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, TensorToOutput}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.jni.{ScalaCallbacksRegistry => NativeCallbacksRegistry, TensorFlow => NativeLibrary}

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
    * @return Created op output.
    */
  def callback[IT, IV, ID, IS, OT, OV, OD, OS](
      function: IV => OV,
      input: IT,
      outputDataType: OD,
      stateful: Boolean = true,
      name: String = "Callback"
  )(implicit
      evTensorToOutputInput: TensorToOutput.Aux[IV, IT],
      evTensorToOutputOutput: TensorToOutput.Aux[OV, OT],
      evInput: NestedStructure.Aux[IT, IV, ID, IS],
      evOutput: NestedStructure.Aux[OT, OV, OD, OS]
  ): OT = {
    val id = NativeCallbacksRegistry.register(inputs => {
      val inputTensors = inputs.map(Tensor.fromNativeHandle[Any]).toSeq
      val outputs = function(evInput.decodeTensorFromOutput(input, inputTensors)._1)
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
    evOutput.decodeOutputFromDataType(outputDataType, builder.build().output)._1
  }
}

/** Contains helpers for dealing with callbacks. */
object Callback extends Callback {
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
