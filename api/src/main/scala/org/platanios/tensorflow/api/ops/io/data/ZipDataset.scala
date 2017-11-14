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
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.types.DataType

/** Dataset that wraps the application of the `zip` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  name          Name for this dataset.
  * @tparam T1            First tensor type (i.e., nested structure of tensors).
  * @tparam O1            First output type (i.e., nested structure of symbolic tensors).
  * @tparam D1            First data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S1            First shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  * @tparam T2            Second tensor type (i.e., nested structure of tensors).
  * @tparam O2            Second output type (i.e., nested structure of symbolic tensors).
  * @tparam D2            Second data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S2            Second shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2](
    inputDataset1: Dataset[T1, O1, D1, S1],
    inputDataset2: Dataset[T2, O2, D2, S2],
    override val name: String = "ZipDataset"
)(implicit
    ev1: Data.Aux[T1, O1, D1, S1],
    ev2: Data.Aux[T2, O2, D2, S2]
) extends Dataset[(T1, T2), (O1, O2), (D1, D2), (S1, S2)](name) {
  override def createHandle(): Output = {
    ZipDataset.datasetZip(
      Op.createWithNameScope(name)(Seq(inputDataset1.createHandle(), inputDataset2.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override def outputDataTypes: (D1, D2) = (inputDataset1.outputDataTypes, inputDataset2.outputDataTypes)
  override def outputShapes: (S1, S2) = (inputDataset1.outputShapes, inputDataset2.outputShapes)
}

/** Dataset that wraps the application of the `zip3` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  inputDataset3 Third input dataset.
  * @param  name          Name for this dataset.
  * @tparam T1            First tensor type (i.e., nested structure of tensors).
  * @tparam O1            First output type (i.e., nested structure of symbolic tensors).
  * @tparam D1            First data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S1            First shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  * @tparam T2            Second tensor type (i.e., nested structure of tensors).
  * @tparam O2            Second output type (i.e., nested structure of symbolic tensors).
  * @tparam D2            Second data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S2            Second shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  * @tparam T3            Third tensor type (i.e., nested structure of tensors).
  * @tparam O3            Third output type (i.e., nested structure of symbolic tensors).
  * @tparam D3            Third data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S3            Third shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3](
    inputDataset1: Dataset[T1, O1, D1, S1],
    inputDataset2: Dataset[T2, O2, D2, S2],
    inputDataset3: Dataset[T3, O3, D3, S3],
    override val name: String = "Zip3Dataset"
)(implicit
    ev1: Data.Aux[T1, O1, D1, S1],
    ev2: Data.Aux[T2, O2, D2, S2],
    ev3: Data.Aux[T3, O3, D3, S3]
) extends Dataset[(T1, T2, T3), (O1, O2, O3), (D1, D2, D3), (S1, S2, S3)](name) {
  override def createHandle(): Output = {
    ZipDataset.datasetZip(
      Op.createWithNameScope(name)(Seq(
        inputDataset1.createHandle(), inputDataset2.createHandle(), inputDataset3.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override def outputDataTypes: (D1, D2, D3) = {
    (inputDataset1.outputDataTypes, inputDataset2.outputDataTypes, inputDataset3.outputDataTypes)
  }

  override def outputShapes: (S1, S2, S3) = {
    (inputDataset1.outputShapes, inputDataset2.outputShapes, inputDataset3.outputShapes)
  }
}

/** Dataset that wraps the application of the `zipMultiple` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDatasets Input datasets.
  * @param  name          Name for this dataset.
  * @tparam T             Tensor type (i.e., nested structure of tensors).
  * @tparam O             Output type (i.e., nested structure of symbolic tensors).
  * @tparam D             Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S             Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class ZipMultipleDataset[T, O, D, S](
    inputDatasets: Seq[Dataset[T, O, D, S]],
    override val name: String = "ZipMultipleDataset"
)(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[Seq[T], Seq[O], Seq[D], Seq[S]](name) {
  override def createHandle(): Output = {
    ZipDataset.datasetZip(
      Op.createWithNameScope(name)(inputDatasets.map(_.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override def outputDataTypes: Seq[D] = inputDatasets.map(_.outputDataTypes)
  override def outputShapes: Seq[S] = inputDatasets.map(_.outputShapes)
}

object ZipDataset {
  private[data] trait Implicits {
    implicit def datasetToZipDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S])(implicit
        ev: Data.Aux[T, O, D, S]
    ): ZipDatasetOps[T, O, D, S] = {
      ZipDatasetOps(dataset)
    }
  }

  case class ZipDatasetOps[T, O, D, S] private[ZipDataset] (dataset: Dataset[T, O, D, S])(implicit
      ev: Data.Aux[T, O, D, S]
  ) {
    /** $OpDocDatasetZip
      *
      * @param  other Dataset to zip with the current dataset.
      * @return Created dataset.
      */
    def zip[T2, O2, D2, S2](other: Dataset[T2, O2, D2, S2], name: String = "Zip")(implicit
        ev2: Data.Aux[T2, O2, D2, S2]
    ): Dataset[(T, T2), (O, O2), (D, D2), (S, S2)] = {
      Op.createWithNameScope(s"${dataset.name}_${other.name}") {
        ZipDataset(dataset, other, name)
      }
    }

    /** $OpDocDatasetZip
      *
      * @param  other1 First dataset to zip with the current dataset.
      * @param  other2 Second dataset to zip with the current dataset.
      * @return Created dataset.
      */
    def zip3[T2, O2, D2, S2, T3, O3, D3, S3](
        other1: Dataset[T2, O2, D2, S2],
        other2: Dataset[T3, O3, D3, S3],
        name: String = "Zip3"
    )(implicit
        ev2: Data.Aux[T2, O2, D2, S2],
        ev3: Data.Aux[T3, O3, D3, S3]
    ): Dataset[(T, T2, T3), (O, O2, O3), (D, D2, D3), (S, S2, S3)] = {
      Op.createWithNameScope(s"${dataset.name}_${other1.name}_${other2.name}") {
        Zip3Dataset(dataset, other1, other2, name)
      }
    }
  }

  /** Creates an op that zips multiple datasets together.
    *
    * A zip dataset is a dataset that zips together multiple datasets.
    *
    * @param  datasets        Tensors containing the handles of the datasets to zip together.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[data] def datasetZip(
      datasets: Seq[Output], outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetZip"): Output = {
    Op.Builder(opType = "ZipDataset", name = name)
        .addInputList(datasets)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** @define OpDocDatasetZip
    *   The dataset `zip`, `zip3`, and `zipMultiple` ops create a new dataset by zipping together multiple datasets.
    *
    *   The ops have similar semantics to the built-in Scala collections `zip` function.
    *
    *   The main difference between the three ops is that `zip` is limited to two datasets of potentially
    *   differently-typed elements, `zip3` is similarly limited to three datasets of potentially differently-typed
    *   elements, and `zipMultiple` can zip together an arbitrary number of datasets containing elements of the same
    *   type.
    *
    *   For example:
    *   {{{
    *     // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *     a = { 1, 2, 3 }
    *     b = { 4, 5, 6 }
    *     c = { (7, 8), (9, 10), (11, 12) }
    *     d = { 13, 14 }
    *
    *     // The nested structure of the `datasets` argument determines the structure of elements in the resulting
    *     // dataset.
    *     a.zip(b) ==> { (1, 4), (2, 5), (3, 6) }
    *     b.zip(a) ==> { (4, 1), (5, 2), (6, 3) }
    *
    *     // The `datasets` argument may contain an arbitrary number of datasets.
    *     a.zip3(b, c) ==> { (1, 4, (7, 8)), (2, 5, (9, 10)), (3, 6, (11, 12)) }
    *
    *     // The number of elements in the resulting dataset is the same as the size of the smallest provided dataset.
    *     a.zip(d) ==> { (1, 13), (2, 14) }
    *
    *     // The `zipMultiple` op returns datasets with sequence-valued elements.
    *     a.zipMultiple(b) ==> { Seq(1, 4), Seq(2, 5), Seq(3, 6) }
    *   }}}
    */
  private[data] trait Documentation
}
