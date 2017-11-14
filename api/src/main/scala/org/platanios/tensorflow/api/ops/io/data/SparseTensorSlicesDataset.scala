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
import org.platanios.tensorflow.api.ops.{Op, Output, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor}
import org.platanios.tensorflow.api.types.{DataType, INT64}

/** Dataset that splits a sparse tensor into its rows.
  *
  * @param  tensor Sparse tensor.
  * @param  name   Name for this dataset.
  *
  * @author Emmanouil Antonios Platanios
  */
case class SparseTensorSlicesDataset(
    tensor: SparseTensor,
    override val name: String = "SparseTensorSliceDataset"
) extends Dataset[
    (Tensor, Tensor, Tensor), (Output, Output, Output), (DataType, DataType, DataType), (Shape, Shape, Shape)](name) {
  /** Creates a `RESOURCE` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  override def createHandle(): Output = {
    Op.Builder(opType = "SparseTensorSliceDataset", name = name)
        .addInput(tensor.indices)
        .addInput(tensor.values)
        .addInput(tensor.denseShape)
        .build().outputs(0)
  }

  override def outputDataTypes: (DataType, DataType, DataType) = (INT64, tensor.dataType, INT64)

  override def outputShapes: (Shape, Shape, Shape) = {
    val indicesShape = tensor.indices.shape
    val denseShapeShape = tensor.denseShape.shape
    val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
    (Shape(-1, rank), Shape(-1), Shape(rank))
  }
}

/** Dataset that splits a sparse tensor into its rows.
  *
  * @param  tensor Sparse tensor.
  * @param  name   Name for this dataset.
  *
  * @author Emmanouil Antonios Platanios
  */
case class SparseOutputSlicesDataset(
    tensor: SparseOutput,
    override val name: String = "SparseOutputSliceDataset"
) extends Dataset[
    (Tensor, Tensor, Tensor), (Output, Output, Output), (DataType, DataType, DataType), (Shape, Shape, Shape)](name) {
  /** Creates a `RESOURCE` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  override def createHandle(): Output = {
    Op.Builder(opType = "SparseTensorSliceDataset", name = name)
        .addInput(tensor.indices)
        .addInput(tensor.values)
        .addInput(tensor.denseShape)
        .build().outputs(0)
  }

  override def outputDataTypes: (DataType, DataType, DataType) = (INT64, tensor.dataType, INT64)

  override def outputShapes: (Shape, Shape, Shape) = {
    val indicesShape = tensor.indices.shape
    val denseShapeShape = tensor.denseShape.shape
    val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
    (Shape(-1, rank), Shape(-1), Shape(rank))
  }
}
