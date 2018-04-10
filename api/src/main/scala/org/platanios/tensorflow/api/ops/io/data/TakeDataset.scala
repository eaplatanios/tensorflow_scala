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

import org.platanios.tensorflow.api.ops.{Basic, Op, Output}

/** Dataset that wraps the application of the `take` op.
  *
  * $OpDocDatasetTake
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to take.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class TakeDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    count: Long,
    override val name: String = "TakeDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "TakeDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(count)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

/** Dataset that wraps the application of the `take` op.
  *
  * $OpDocDatasetTake
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to take.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class DynamicTakeDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    count: Output,
    override val name: String = "TakeDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "TakeDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(count)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object TakeDataset {
  case class TakeDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetTake
      *
      * @param  count Number of elements to take.
      * @return Created dataset.
      */
    def take(count: Long): Dataset[T, O, D, S] = take(count, "Take")

    /** $OpDocDatasetTake
      *
      * @param  count Number of elements to take.
      * @param  name  Name for the created dataset.
      * @return Created dataset.
      */
    def take(count: Long, name: String): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        TakeDataset(dataset, count, name)
      }
    }

    /** $OpDocDatasetTake
      *
      * @param  count Number of elements to take.
      * @return Created dataset.
      */
    def take(count: Output): Dataset[T, O, D, S] = take(count, "Take")

    /** $OpDocDatasetTake
      *
      * @param  count Number of elements to take.
      * @param  name  Name for the created dataset.
      * @return Created dataset.
      */
    def take(count: Output, name: String): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        DynamicTakeDataset(dataset, count, name)
      }
    }
  }

  /** @define OpDocDatasetTake
    *   The dataset `take` op takes at most the provided number of elements from a dataset, forming a new dataset. If
    *   the provided number is `-1`, then all of the elements are taken.
    *
    *   The op has similar semantics to the built-in Scala collections `take` function.
    */
  private[data] trait Documentation
}
