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
import org.platanios.tensorflow.api.types.INT64

/** Dataset that wraps the application of the `shuffle` op.
  *
  * $OpDocDatasetShuffle
  *
  * @param  inputDataset Input dataset.
  * @param  bufferSize   Buffer size, meaning the number of output elements to buffer in an iterator over this dataset.
  * @param  seed         Seed value for the random number generator. If not provided, a random seed is used.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class ShuffleDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    bufferSize: Long,
    seed: Option[Int],
    override val name: String = "ShuffleDataset"
) extends Dataset[T, O, D, S](name)(inputDataset.evOToT, inputDataset.ev, inputDataset.evFunctionInput) {
  override def createHandle(): Output = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val seed1 = graphSeed.getOrElse(0)
    val seed2 = opSeed.getOrElse(0)
    Op.Builder(opType = "ShuffleDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(bufferSize)))
        .addInput(Op.createWithNameScope(name)(Basic.constant(seed1, INT64)))
        .addInput(Op.createWithNameScope(name)(Basic.constant(seed2, INT64)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object ShuffleDataset {
  case class ShuffleDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetShuffle
      *
      * @param  bufferSize Buffer size, meaning the number of output elements to buffer in an iterator over this dataset.
      * @param  seed       Seed value for the random number generator. If not provided, a random seed is used.
      * @param  name       Name for the created dataset.
      * @return Created dataset.
      */
    def shuffle(bufferSize: Long, seed: Option[Int] = None, name: String = "Shuffle"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        ShuffleDataset(dataset, bufferSize, seed, name)
      }
    }
  }

  /** @define OpDocDatasetShuffle
    *   The dataset `shuffle` op randomly shuffles the elements of a dataset.
    */
  private[data] trait Documentation
}
