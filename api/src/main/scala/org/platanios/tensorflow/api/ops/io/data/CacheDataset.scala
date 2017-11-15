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

/** Dataset that wraps the application of the `cache` op.
  *
  * $OpDocDatasetCache
  *
  * @param  inputDataset Input dataset.
  * @param  directory    Directory to use for caching. If empty, then the provided dataset will be cached in memory.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class CacheDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    directory: String,
    override val name: String = "CacheDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.ev, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    Op.Builder(opType = "CacheDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(directory)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object CacheDataset {
  private[data] trait Implicits {
    implicit def datasetToCacheDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): CacheDatasetOps[T, O, D, S] = {
      CacheDatasetOps(dataset)
    }
  }

  case class CacheDatasetOps[T, O, D, S] private[CacheDataset] (dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetCache
      *
      * @param  directory Directory to use for caching. If empty, then the provided dataset will be cached in memory.
      * @param  name      Name for the created dataset.
      * @return Created dataset.
      */
    def cache(directory: String, name: String = "Cache"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        CacheDataset(dataset, directory, name)
      }
    }
  }

  /** @define OpDocDatasetCache
    *   The dataset `cache` op caches the elements in a dataset in the provided directory. If the provided directory is
    *   empty, then the elements are cached in memory.
    */
  private[data] trait Documentation
}
