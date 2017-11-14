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

import org.platanios.tensorflow.api.ops.{Function, Op, Output}

/** Dataset that wraps the application of the `ignoreErrors` op.
  *
  * $OpDocDatasetIgnoreErrors
  *
  * @param  inputDataset Input dataset.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class IgnoreErrorsDataset[T, O, D, S] private[io] (
    inputDataset: Dataset[T, O, D, S],
    override val name: String = "IgnoreErrorsDataset"
)(implicit
    ev: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Op.Builder(opType = "IgnoreErrorsDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = inputDataset.outputDataTypes
  override def outputShapes: S = inputDataset.outputShapes
}

object IgnoreErrorsDataset {
  private[data] trait Implicits {
    implicit def datasetToIgnoreErrorsDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S])(implicit
        ev: Data.Aux[T, O, D, S],
        evFunctionInput: Function.ArgType[O]
    ): IgnoreErrorsDatasetOps[T, O, D, S] = {
      IgnoreErrorsDatasetOps(dataset)
    }
  }

  case class IgnoreErrorsDatasetOps[T, O, D, S] private[IgnoreErrorsDataset] (dataset: Dataset[T, O, D, S])(implicit
      ev: Data.Aux[T, O, D, S],
      evFunctionInput: Function.ArgType[O]
  ) {
    /** $OpDocDatasetIgnoreErrors
      *
      * @param  name Name for the created dataset.
      * @return Created dataset.
      */
    def ignoreErrors(name: String = "IgnoreErrors"): Dataset[T, O, D, S] = {
      Op.createWithNameScope(dataset.name) {
        IgnoreErrorsDataset(dataset, name)
      }
    }
  }

  /** @define OpDocDatasetIgnoreErrors
    *   The dataset `ignoreErrors` creates a new dataset from the provided one and silently ignores any errors.
    *
    *   Use this transformation to produce a dataset that contains the same elements as the input, but silently drops
    *   any elements that caused an error. For example:
    *   {{{
    *     dataset = datasetFromSlices(Tensor(1.0, 2.0, 0.0, 4.0))
    *
    *     // Computing `checkNumerics(1.0 / 0.0)` will raise an [[IllegalArgumentException]].
    *     dataset = dataset.map(x => checkNumerics(1.0 / x, "error"))
    *
    *     // Using `ignoreErrors` will drop the elements that cause errors.
    *     dataset = dataset.ignoreErrors()  // ==> { 1.0, 0.5, 0.2 }
    *   }}}
    */
  private[data] trait Documentation
}
