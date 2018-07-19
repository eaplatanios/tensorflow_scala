/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.implicits.helpers.{StructureFromOutput, StructureFromTensor}
import org.platanios.tensorflow.api.ops.{Function, Op, Output}

/** Dataset with a single element.
  *
  * @param  data Data representing the single element of this dataset.
  * @param  name Name for this dataset.
  * @tparam T    Tensor type (i.e., nested structure of tensors).
  * @tparam O    Output type (i.e., nested structure of symbolic tensors).
  * @tparam D    Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S    Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorDataset[T, O, D, S](
    data: T,
    override val name: String = "TensorDataset"
)(implicit
    evStructure: StructureFromTensor.Aux[T, O, D, S],
    evData: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = evData.flattenedOutputsFromT(data)
    Op.Builder(opType = "TensorDataset", name = name)
        .addInputList(flattenedOutputs)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = evData.dataTypesFromT(data)
  override def outputShapes: S = evData.shapesFromT(data)
}

// TODO: [DATA] Tensor and output datasets should be one class (the outputs one). However, datasets should be made more functional so they can be created lazily.

/** Dataset with a single element.
  *
  * @param  data Data representing the single element of this dataset.
  * @param  name Name for this dataset.
  * @tparam T    Tensor type (i.e., nested structure of tensors).
  * @tparam O    Output type (i.e., nested structure of symbolic tensors).
  * @tparam D    Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S    Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class OutputDataset[T, O, D, S](
    data: O,
    override val name: String = "OutputDataset"
)(implicit
    evStructure: StructureFromOutput.Aux[T, O, D, S],
    evData: Data.Aux[T, O, D, S],
    evFunctionInput: Function.ArgType[O]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = evData.flattenedOutputsFromO(data)
    Op.Builder(opType = "TensorDataset", name = name)
        .addInputList(flattenedOutputs)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: D = evData.dataTypesFromO(data)
  override def outputShapes: S = evData.shapesFromO(data)
}
