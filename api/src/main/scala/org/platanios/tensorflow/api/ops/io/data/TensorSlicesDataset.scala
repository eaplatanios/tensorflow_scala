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

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Function, Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import scala.language.postfixOps

/** Dataset with slices from the nested structure of [[Tensor]]s (i.e., a [[Data]]-supported type). The slices are
  * taken along the first axis of each [[Tensor]] in the nested structure.
  *
  * @param  data Data representing the elements of this dataset.
  * @param  name Name for this dataset.
  * @tparam T    Tensor type (i.e., nested structure of tensors).
  * @tparam O    Output type (i.e., nested structure of symbolic tensors).
  * @tparam D    Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S    Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorSlicesDataset[T, O, D, S](
    data: T,
    override val name: String = "TensorSlicesDataset"
)(implicit
    ev: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = ev.flattenedOutputsFromT(data)
    Op.Builder(opType = "TensorSliceDataset", name = name)
        .addInputList(flattenedOutputs)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = ev.dataTypesFromT(data)

  override def outputShapes: S = {
    val flattenedShapes = ev.flattenedShapes(ev.shapesFromT(data))
    ev.unflattenShapes(outputDataTypes, flattenedShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))
  }
}

/** Dataset with slices from the nested structure of [[Output]]s (i.e., a [[Data]]-supported type). The slices are
  * taken along the first axis of each [[Output]] in the nested structure.
  *
  * @param  data Data representing the elements of this dataset.
  * @param  name Name for this dataset.
  * @tparam T    Tensor type (i.e., nested structure of tensors).
  * @tparam O    Output type (i.e., nested structure of symbolic tensors).
  * @tparam D    Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S    Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class OutputSlicesDataset[T, O, D, S](
    data: O,
    override val name: String = "OutputSlicesDataset"
)(implicit
    ev: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = ev.flattenedOutputsFromO(data)
    Op.Builder(opType = "TensorSliceDataset", name = name)
        .addInputList(flattenedOutputs)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = ev.dataTypesFromO(data)

  override def outputShapes: S = {
    val flattenedShapes = ev.flattenedShapes(ev.shapesFromO(data))
    ev.unflattenShapes(outputDataTypes, flattenedShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))
  }
}
