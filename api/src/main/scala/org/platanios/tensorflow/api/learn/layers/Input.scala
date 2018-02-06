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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeAuxToDataType, DataTypeToOutput, OutputToTensor}
import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops.io.data.{Data, Iterator}
import org.platanios.tensorflow.api.ops.Op

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Input {
  private[layers] trait API {
    type Input[T, O, DA, D, S] = layers.Input[T, O, DA, D, S]
    val Input: layers.Input.type = layers.Input
  }

  object API extends API
}

case class Input[T, O, DA, D, S](private val _dataType: DA, shape: S, name: String = "Input")(implicit
    val evDAToD: DataTypeAuxToDataType.Aux[DA, D],
    val evDToO: DataTypeToOutput.Aux[D, O],
    val evOToT: OutputToTensor.Aux[O, T],
    val evData: Data.Aux[T, O, D, S]
) {
  val dataType: D = evDAToD.castDataType(_dataType)

  private[this] val cache: mutable.Map[Graph, Iterator[T, O, D, S]] = mutable.Map.empty

  protected def create(): Iterator[T, O, D, S] = Iterator.fromStructure(dataType, shape, name = name)

  final def apply(): Iterator[T, O, D, S] = cache.getOrElse(Op.currentGraph, create())

  def zip[T2, O2, DA2, D2, S2](other: Input[T2, O2, DA2, D2, S2]):
  Input[(T, T2), (O, O2), (DA, DA2), (D, D2), (S, S2)] = {
    implicit val evDA2ToD2: DataTypeAuxToDataType.Aux[DA2, D2] = other.evDAToD
    implicit val evD2ToO2: DataTypeToOutput.Aux[D2, O2] = other.evDToO
    implicit val evO2ToT2: OutputToTensor.Aux[O2, T2] = other.evOToT
    implicit val evData2: Data.Aux[T2, O2, D2, S2] = other.evData
    Input[(T, T2), (O, O2), (DA, DA2), (D, D2), (S, S2)](
      (_dataType, other._dataType), (shape, other.shape), s"${name}_${other.name}/Zip")
  }
}
