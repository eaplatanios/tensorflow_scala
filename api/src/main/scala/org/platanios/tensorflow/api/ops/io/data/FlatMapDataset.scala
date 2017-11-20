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

import org.platanios.tensorflow.api.implicits.helpers.OutputToTensor
import org.platanios.tensorflow.api.ops.{Function, Op, Output}

/** Dataset that wraps the application of the `flatMap` op.
  *
  * $OpDocDatasetFlatMap
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
case class FlatMapDataset[T, O, D, S, RT, RO, RD, RS](
    inputDataset: Dataset[T, O, D, S],
    function: (O) => Dataset[RT, RO, RD, RS],
    override val name: String = "FlatMapDataset"
)(implicit
    evOToT: OutputToTensor.Aux[O, T] = inputDataset.evOToT,
    evData: Data.Aux[T, O, D, S] = inputDataset.evData,
    evFunctionInput: Function.ArgType[O] = inputDataset.evFunctionInput,
    evROToRT: OutputToTensor.Aux[RO, RT],
    evRData: Data.Aux[RT, RO, RD, RS],
    evFunctionOutput: Function.ArgType[RO]
) extends Dataset[RT, RO, RD, RS](name)(evROToRT, evRData, evFunctionOutput) {
  private[this] lazy val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes, appendHashToName = true)
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "FlatMapDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedFunction.extraInputs)
        .setAttribute("f", instantiatedFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: RD = instantiatedFunction.dummyOutputs.outputDataTypes
  override def outputShapes: RS = instantiatedFunction.dummyOutputs.outputShapes
}

object FlatMapDataset {
  case class FlatMapDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetFlatMap
      *
      * @param  function Mapping function.
      * @param  name     Name for the created dataset.
      * @return Created dataset.
      */
    def flatMap[RT, RO, RD, RS](function: (O) => Dataset[RT, RO, RD, RS], name: String = "FlatMap")(implicit
        evROToRT: OutputToTensor.Aux[RO, RT],
        evRData: Data.Aux[RT, RO, RD, RS],
        evFunctionOutput: Function.ArgType[RO]
    ): Dataset[RT, RO, RD, RS] = {
      Op.createWithNameScope(dataset.name) {
        FlatMapDataset(dataset, function, name)
      }
    }
  }

  /** @define OpDocDatasetFlatMap
    *   The dataset `flatMap` op creates a new dataset by a mapping function across all elements of another dataset and
    *   then flattening the results.
    *
    *   The op has similar semantics to the built-in Scala collections `flatMap` function.
    */
  private[data] trait Documentation
}
