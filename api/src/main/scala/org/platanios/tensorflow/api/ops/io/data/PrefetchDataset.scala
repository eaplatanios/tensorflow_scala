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

import org.platanios.tensorflow.api.ops.{Basic, Op, Output}

/** Dataset that wraps the application of the `prefetch` op.
  *
  * $OpDocDatasetPrefetch
  *
  * @param  inputDataset Input dataset.
  * @param  bufferSize   Number of elements to prefetch.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class PrefetchDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    bufferSize: Long,
    override val name: String = "PrefetchDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.evData, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "PrefetchDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(bufferSize)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object PrefetchDataset {
  case class PrefetchDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetPrefetch
      *
      * @param  bufferSize Number of elements to prefetch.
      * @param  name       Name for the created dataset.
      * @return Created dataset.
      */
    def prefetch(bufferSize: Long, name: String = "Prefetch"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        PrefetchDataset(dataset, bufferSize, name)
      }
    }
  }

  /** @define OpDocDatasetPrefetch
    *   The dataset `prefetch` op creates a new dataset by asynchronously prefetching elements from the provided
    *   dataset.
    */
  private[data] trait Documentation
}
