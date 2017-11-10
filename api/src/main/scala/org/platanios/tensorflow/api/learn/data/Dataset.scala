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

package org.platanios.tensorflow.api.learn.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Function, SparseOutput, io}
import org.platanios.tensorflow.api.ops.io.Data
import org.platanios.tensorflow.api.tensors.SparseTensor
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
trait Dataset {
  def DatasetFrom[T, O, D, S](
      data: T, name: String = "TensorDataset")(implicit ev: Data.Aux[T, O, D, S]): io.Dataset[T, O, D, S] = {
    io.Dataset.from(data, name)(ev)
  }

  def DatasetFromSlices[T, O, D, S](
      data: T, name: String = "TensorSlicesDataset")(implicit ev: Data.Aux[T, O, D, S]): io.Dataset[T, O, D, S] = {
    io.Dataset.fromSlices(data, name)(ev)
  }

  private[api] def DatasetFromSparseSlices(tensor: SparseTensor, name: String = "SparseTensorSliceDataset"):
  io.Dataset[SparseTensor, SparseOutput, (DataType, DataType, DataType), (Shape, Shape, Shape)] = {
    io.Dataset.fromSparseSlices(tensor, name)
  }

  def DatasetFromGenerator[T, O, D, S](
      generator: () => Iterable[T],
      outputDataType: D,
      outputShape: S = null
  )(implicit
      ev: Data.Aux[T, O, D, S],
      evFunctionOutput: Function.ArgType[O]
  ): io.Dataset[T, O, D, S] = {
    io.Dataset.fromGenerator[T, O, D, S](generator, outputDataType, outputShape)(ev, evFunctionOutput)
  }
}
