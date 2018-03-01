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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Function, Op, Output}
import org.platanios.tensorflow.api.types.{INT64, VARIANT}

/** $OpDocDatasetGroupByWindow
  *
  * @param  inputDataset Input dataset.
  * @param  keyFn        Function used to compute the grouping key.
  * @param  reduceFn     Function used to reduce each group.
  * @param  windowSizeFn Function used to compute the maximum window size per key.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class GroupByWindowDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    keyFn: (O) => Output,
    reduceFn: ((Output, Dataset[T, O, D, S])) => Dataset[T, O, D, S],
    windowSizeFn: (Output) => Output,
    override val name: String = "GroupByWindowDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  private[this] lazy val instantiatedKeyFunction = {
    Function(s"$name/KeyFunction", keyFn).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes)
  }

  private[this] lazy val instantiatedReduceFunction = {
    Function(s"$name/ReduceFunction", reduceFn).instantiate(
      Seq(INT64, VARIANT), Seq(Shape.scalar(), Shape.scalar()), Some((null, inputDataset)))
  }

  private[this] lazy val instantiatedWindowSizeFunction = {
    Function(s"$name/WindowSizeFunction", (o: Output) => windowSizeFn(o).cast(INT64)).instantiate(
      Seq(INT64), Seq(Shape.scalar()))
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "GroupByWindowDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedKeyFunction.extraInputs)
        .addInputList(instantiatedReduceFunction.extraInputs)
        .addInputList(instantiatedWindowSizeFunction.extraInputs)
        .setAttribute("key_func", instantiatedKeyFunction)
        .setAttribute("reduce_func", instantiatedReduceFunction)
        .setAttribute("window_size_func", instantiatedWindowSizeFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = instantiatedReduceFunction.dummyOutputs.outputDataTypes
  override def outputShapes: S = instantiatedReduceFunction.dummyOutputs.outputShapes
}

object GroupByWindowDataset {
  case class GroupByWindowDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    def groupByWindow(
        keyFn: (O) => Output,
        reduceFn: ((Output, Dataset[T, O, D, S])) => Dataset[T, O, D, S],
        windowSizeFn: (Output) => Output,
        name: String = "GroupByWindowDataset"
    ): Dataset[T, O, D, S] = {
      GroupByWindowDataset(dataset, keyFn, reduceFn, windowSizeFn, name)
    }
  }

  /** @define OpDocDatasetGroupByWindow
    *   The dataset `groupByWindow` op creates a new dataset by applying transformation that groups windows of elements
    *   by a key and then reduces them.
    *
    *   This transformation maps each consecutive element in a dataset to a key using `keyFn` and groups the elements by
    *   key. It then applies `reduceFn` to at most `windowSizeFn(key)` elements matching the same key. All except the
    *   final window for each key will contain `windowSizeFn(key)` elements; the final window may be smaller.
    */
  private[data] trait Documentation
}
