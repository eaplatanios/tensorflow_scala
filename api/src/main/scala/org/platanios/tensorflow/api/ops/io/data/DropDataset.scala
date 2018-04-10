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

/** Dataset that wraps the application of the `drop` op.
  *
  * $OpDocDatasetDrop
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to drop.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class DropDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    count: Long,
    override val name: String = "DropDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "SkipDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(count)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

/** Dataset that wraps the application of the `drop` op.
  *
  * $OpDocDatasetDrop
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to drop.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class DynamicDropDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    count: Output,
    override val name: String = "DropDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "SkipDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(count)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object DropDataset {
  case class DropDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetDrop
      *
      * @param  count Number of elements to drop.
      * @return Created dataset.
      */
    def drop(count: Long): Dataset[T, O, D, S] = drop(count, "Drop")

    /** $OpDocDatasetDrop
      *
      * @param  count Number of elements to drop.
      * @param  name  Name for the created dataset.
      * @return Created dataset.
      */
    def drop(count: Long, name: String): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        DropDataset(dataset, count, name)
      }
    }

    /** $OpDocDatasetDrop
      *
      * @param  count Number of elements to drop.
      * @return Created dataset.
      */
    def drop(count: Output): Dataset[T, O, D, S] = drop(count, "Drop")

    /** $OpDocDatasetDrop
      *
      * @param  count Number of elements to drop.
      * @param  name  Name for the created dataset.
      * @return Created dataset.
      */
    def drop(count: Output, name: String): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        DynamicDropDataset(dataset, count, name)
      }
    }
  }

  /** @define OpDocDatasetDrop
    *   The dataset `drop` op drops at most the provided number of elements from a dataset, forming a new dataset. If
    *   the provided number is `-1`, then all of the elements are dropped.
    *
    *   The op has similar semantics to the built-in Scala collections `drop` function.
    */
  private[data] trait Documentation
}
