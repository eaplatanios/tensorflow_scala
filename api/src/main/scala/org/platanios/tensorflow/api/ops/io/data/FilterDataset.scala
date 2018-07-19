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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.ops.{Function, Op, Output}

/** Dataset that wraps the application of the `filter` op.
  *
  * $OpDocDatasetFilter
  *
  * @param  inputDataset Input dataset.
  * @param  predicateFn  Filter predicate function.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class FilterDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    predicateFn: O => Output,
    override val name: String = "FilterDataset"
) extends Dataset[T, O, D, S](name)(
  inputDataset.evStructure, inputDataset.evData, inputDataset.evFunctionInput
) {
  private[this] lazy val instantiatedPredicateFunction = {
    Function(s"$name/Predicate", predicateFn).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes,
      appendHashToName = true)
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "FilterDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedPredicateFunction.extraInputs)
        .setAttribute("predicate", instantiatedPredicateFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object FilterDataset {
  case class FilterDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetFilter
      *
      * @param  predicateFn Filter predicate function.
      * @param  name        Name for the created dataset.
      * @return Created dataset.
      */
    def filter(predicateFn: O => Output, name: String = "Filter"): Dataset[T, O, D, S] = {
      FilterDataset(dataset, predicateFn, name)
    }
  }

  /** @define OpDocDatasetFilter
    *   The dataset `filter` op creates a new dataset by filtering the elements of another dataset using the provided
    *   predicate function. The predicate function must return a scalar boolean tensor.
    *
    *   The op has similar semantics to the built-in Scala collections filter` function.
    */
  private[data] trait Documentation
}
