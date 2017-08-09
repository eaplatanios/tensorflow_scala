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

package org.platanios.tensorflow.api.ops.io

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.DataType

/** Represents the state of iterating through a [[Dataset]].
  *
  * @param  handle          Handle of the iterator.
  * @param  initializer     Iterator initializer op.
  * @param  outputDataTypes Output data types of the created iterator.
  * @param  outputShapes    Output shapes of the created iterator.
  *
  * @author Emmanouil Antonios Platanios
  */
class Iterator private[io] (
    val handle: Output,
    val initializer: Op,
    val outputDataTypes: Seq[DataType],
    val outputShapes: Seq[Shape]) {

}

object Iterator {
  trait API {
    def fromDataset(dataset: Dataset, sharedName: String = "", name: String = "Iterator"): Iterator = ???
  }

  object API extends API

  /** Creates an op that is a container for an `Iterator` resource.
    *
    * @param  container       If non-empty, then the constructed iterator is placed in the provided container.
    *                         Otherwise, a default container is used.
    * @param  sharedName      If non-empty, then the constructed iterator will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  outputDataTypes Output data types of the created iterator.
    * @param  outputShapes    Output shapes of the created iterator.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the constructed iterator.
    */
  private[io] def createIterator(
      container: String = "", sharedName: String = "", outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "Iterator"): Output = {
    Op.Builder(opType = "Iterator", name = name)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op that makes a new iterator for the provided dataset and stores it in the container pointed to by the
    * provided iterator handle.
    *
    * **Note:** The created op may be executed multiple times. Each execution will reset the iterator in `iterator` to
    * the first element of `dataset`.
    *
    * @param  dataset  Handle of the dataset.
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[io] def makeIterator(dataset: Output, iterator: Output, name: String = "MakeIterator"): Op = {
    Op.Builder(opType = "Iterator", name = name)
        .addInput(dataset)
        .addInput(iterator)
        .build()
  }

  // TODO: [DATASETS] [FUNCTIONS] "oneShotIterator".

  /** Creates an op that gets the next output from the provided iterator.
    *
    * @param  iterator        Handle of the iterator.
    * @param  outputDataTypes Output data types of the iterator.
    * @param  outputShapes    Output shapes of the iterator.
    * @param  name            Name for the created op.
    * @return Created op outputs, which correspond to the iterator outputs.
    */
  private[io] def iteratorGetNext(
      iterator: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "IteratorGetNext"): Seq[Output] = {
    Op.Builder(opType = "IteratorGetNext", name = name)
        .addInput(iterator)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs.toSeq
  }

  /** Creates an op that releases any resources used by the provided iterator.
    *
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[io] def iteratorDispose(iterator: Output, name: String = "IteratorDispose"): Op = {
    Op.Builder(opType = "IteratorDispose", name = name)
        .addInput(iterator)
        .build()
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the string handle.
    */
  private[io] def iteratorToStringHandle(iterator: Output, name: String = "IteratorToStringHandle"): Output = {
    Op.Builder(opType = "IteratorToStringHandle", name = name)
        .addInput(iterator)
        .build().outputs(0)
  }

  /** Creates an op that converts the provided string representing a handle to an iterator to the corresponding iterator
    * handle.
    *
    * @param  stringHandle `STRING` scalar tensor containing the string representation of a handle of an iterator.
    * @param  name         Name for the created op.
    * @return Created op output, which is a `RESOURCE` scalar tensor containing the iterator handle.
    */
  private[io] def iteratorFromStringHandle(
      stringHandle: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "IteratorFromStringHandle"): Output = {
    Op.Builder(opType = "IteratorFromStringHandle", name = name)
        .addInput(stringHandle)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  private[io] object Gradients {
    GradientsRegistry.registerNonDifferentiable("Iterator")
    GradientsRegistry.registerNonDifferentiable("MakeIterator")
    GradientsRegistry.registerNonDifferentiable("OneShotIterator")
    GradientsRegistry.registerNonDifferentiable("IteratorGetNext")
    GradientsRegistry.registerNonDifferentiable("IteratorDispose")
    GradientsRegistry.registerNonDifferentiable("IteratorToStringHandle")
    GradientsRegistry.registerNonDifferentiable("IteratorFromStringHandle")
  }
}
