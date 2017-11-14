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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.ops.{Basic, Function, Op, Output}

/** Dataset that wraps the application of the `map` op.
  *
  * $OpDocDatasetMap
  *
  * @param  inputDataset Input dataset.
  * @param  function     Mapping function.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class MapDataset[T, O, D, S, RT, RO, RD, RS](
    inputDataset: Dataset[T, O, D, S],
    function: (O) => RO,
    override val name: String = "MapDataset"
)(implicit
    ev: Data.Aux[T, O, D, S],
    evR: Data.Aux[RT, RO, RD, RS],
    evFunctionInput: Function.ArgType[O],
    evFunctionOutput: Function.ArgType[RO]
) extends Dataset[RT, RO, RD, RS](name) {
  private[this] lazy val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes)
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "MapDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedFunction.extraInputs)
        .setAttribute("f", instantiatedFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  private[this] lazy val (_outputDataTypes, _outputShapes): (RD, RS) = {
    val dataTypes = evR.dataTypesFromO(instantiatedFunction.dummyOutputs)
    (evR.unflattenDataTypes(dataTypes, instantiatedFunction.outputDataTypes),
        evR.unflattenShapes(dataTypes, instantiatedFunction.outputShapes))
  }

  override def outputDataTypes: RD = _outputDataTypes
  override def outputShapes: RS = _outputShapes
}

/** Dataset that wraps the application of the `parallelMap` op.
  *
  * $OpDocDatasetMap
  *
  * @param  inputDataset     Input dataset.
  * @param  function         Mapping function.
  * @param  numParallelCalls Number of concurrent invocations of `function` that process elements from `inputDataset` in
  *                          parallel.
  * @param  name             Name for this dataset.
  * @tparam T                Tensor type (i.e., nested structure of tensors).
  * @tparam O                Output type (i.e., nested structure of symbolic tensors).
  * @tparam D                Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S                Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class ParallelMapDataset[T, O, D, S, RT, RO, RD, RS](
    inputDataset: Dataset[T, O, D, S],
    function: (O) => RO,
    numParallelCalls: Int,
    override val name: String = "ParallelMapDataset"
)(implicit
    ev: Data.Aux[T, O, D, S],
    evR: Data.Aux[RT, RO, RD, RS],
    evFunctionInput: Function.ArgType[O],
    evFunctionOutput: Function.ArgType[RO]
) extends Dataset[RT, RO, RD, RS](name) {
  private[this] lazy val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes)
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "ParallelMapDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedFunction.extraInputs)
        .addInput(Op.createWithNameScope(name)(Basic.constant(numParallelCalls, name = "NumParallelCalls")))
        .setAttribute("f", instantiatedFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  private[this] lazy val (_outputDataTypes, _outputShapes): (RD, RS) = {
    val dataTypes = evR.dataTypesFromO(instantiatedFunction.dummyOutputs)
    (evR.unflattenDataTypes(dataTypes, instantiatedFunction.outputDataTypes),
        evR.unflattenShapes(dataTypes, instantiatedFunction.outputShapes))
  }

  override def outputDataTypes: RD = _outputDataTypes
  override def outputShapes: RS = _outputShapes
}

object MapDataset {
  private[data] trait Implicits {
    implicit def datasetToMapDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S])(implicit
        ev: Data.Aux[T, O, D, S]
    ): MapDatasetOps[T, O, D, S] = {
      MapDatasetOps(dataset)
    }
  }

  case class MapDatasetOps[T, O, D, S] private[MapDataset] (dataset: Dataset[T, O, D, S])(implicit
      ev: Data.Aux[T, O, D, S]
  ) {
    /** $OpDocDatasetMap
      *
      * @param  function         Mapping function.
      * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
      *                          sequentially.
      * @param  bufferSize       Maximum number of processed elements that will be buffered.
      * @param  name        Name for the created dataset.
      * @return Created dataset.
      */
    def map[RT, RO, RD, RS](
        function: (O) => RO,
        numParallelCalls: Int = 1,
        bufferSize: Long = 1,
        name: String = "Map"
    )(implicit
        evR: Data.Aux[RT, RO, RD, RS],
        evFunctionInput: Function.ArgType[O],
        evFunctionOutput: Function.ArgType[RO]
    ): Dataset[RT, RO, RD, RS] = {
      Op.createWithNameScope(dataset.name) {
        val mappedDataset: Dataset[RT, RO, RD, RS] = {
          if (numParallelCalls > 1)
            ParallelMapDataset(dataset, function, numParallelCalls, name)
          else
            MapDataset(dataset, function, name)
        }
        if (bufferSize > 1)
          mappedDataset.prefetch(bufferSize)
        else
          mappedDataset
      }
    }
  }

  /** @define OpDocDatasetMap
    *   The dataset `map` op creates a new dataset by a function across all elements of another dataset.
    *
    *   The op has similar semantics to the built-in Scala collections `map` function.
    */
  private[data] trait Documentation
}
