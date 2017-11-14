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
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}

/** Dataset that wraps the application of the `batch` op.
  *
  * $OpDocDatasetBatch
  *
  * @param  inputDataset Input dataset.
  * @param  batchSize    Batch size to use.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class BatchDataset[T, O, D, S](
    inputDataset: Dataset[T, O, D, S],
    batchSize: Long,
    override val name: String = "BatchDataset"
)(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Op.Builder(opType = "BatchDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInput(Op.createWithNameScope(name)(Basic.constant(batchSize)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = {
    ev.unflattenShapes(outputDataTypes, inputDataset.flattenedOutputShapes.map(Shape(-1) ++ _))
  }
}

// TODO: !!! PaddedBatchDataset

object BatchDataset {
  private[data] trait Implicits {
    implicit def datasetToBatchDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S])(implicit
        ev: Data.Aux[T, O, D, S]
    ): BatchDatasetOps[T, O, D, S] = {
      BatchDatasetOps(dataset)
    }
  }

  case class BatchDatasetOps[T, O, D, S] private[BatchDataset] (dataset: Dataset[T, O, D, S])(implicit
      ev: Data.Aux[T, O, D, S]
  ) {
    /** $OpDocDatasetBatch
      *
      * @param  batchSize Batch size.
      * @return Created dataset.
      */
    def batch(batchSize: Long, name: String = "Batch"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        BatchDataset(dataset, batchSize, name)
      }
    }
  }

  /** @define OpDocDatasetBatch
    *   The dataset `batch` op combines consecutive elements of a dataset into batches.
    */
  private[data] trait Documentation
}
