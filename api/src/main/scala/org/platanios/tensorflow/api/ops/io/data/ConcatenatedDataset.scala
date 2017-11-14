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

import org.platanios.tensorflow.api.ops.{Op, Output}

/** Dataset that wraps the application of the `concatenate` op.
  *
  * $OpDocDatasetConcatenate
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  * @throws IllegalArgumentException If the data types of the input datasets are not identical of if their shapes are
  *                                  not compatible.
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[IllegalArgumentException]
case class ConcatenatedDataset[T, O, D, S](
    inputDataset1: Dataset[T, O, D, S],
    inputDataset2: Dataset[T, O, D, S],
    override val name: String = "ConcatenatedDataset"
)(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  if (inputDataset1.flattenedOutputDataTypes != inputDataset2.flattenedOutputDataTypes)
    throw new IllegalArgumentException("The data types of the datasets being concatenated are not the identical.")
  private[this] lazy val mostSpecificFlattenedShapes = {
    inputDataset1.flattenedOutputShapes.zip(inputDataset2.flattenedOutputShapes).map(p => {
      if (!p._1.isCompatibleWith(p._2))
        throw new IllegalArgumentException("The shapes of the datasets being concatenated are not compatible.")
      p._1.mergeWith(p._2)
    })
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "ConcatenateDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset1.createHandle()))
        .addInput(Op.createWithNameScope(name)(inputDataset2.createHandle()))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset1.outputDataTypes
  override def outputShapes: S = ev.unflattenShapes(outputDataTypes, mostSpecificFlattenedShapes)
}

object ConcatenatedDataset {
  private[data] trait Implicits {
    implicit def datasetToConcatenatedDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S])(implicit
        ev: Data.Aux[T, O, D, S]
    ): ConcatenatedDatasetOps[T, O, D, S] = {
      ConcatenatedDatasetOps(dataset)
    }
  }

  case class ConcatenatedDatasetOps[T, O, D, S] private[ConcatenatedDataset] (dataset: Dataset[T, O, D, S])(implicit
      ev: Data.Aux[T, O, D, S]
  ) {
    /** $OpDocDatasetConcatenate
      *
      * @param  other Dataset to concatenate with the current dataset.
      * @param  name  Name for the created dataset.
      * @return Created dataset.
      */
    def concatenate(other: Dataset[T, O, D, S], name: String = "Concatenated"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(s"${dataset.name}_${other.name}") {
        ConcatenatedDataset(dataset, other, name)
      }
    }
  }

  /** @define OpDocDatasetConcatenate
    *   The dataset `concatenate` op creates a new dataset by concatenating the provided datasets.
    *
    *   For example:
    *   {{{
    *     // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *     a = { 1, 2, 3 }
    *     b = { 4, 5, 6, 7 }
    *     a.concatenate(b) ==> { 1, 2, 3, 4, 5, 6, 7 }
    *
    *     // The datasets to be concatenated should have the same nested structures and output types.
    *     c = { (8, 9), (10, 11), (12, 13) }
    *     d = { 14.0, 15.0, 16.0 }
    *     // a.concatenate(c) and a.concatenate(d) would result in exceptions being thrown.
    *   }}}
    */
  private[data] trait Documentation
}
