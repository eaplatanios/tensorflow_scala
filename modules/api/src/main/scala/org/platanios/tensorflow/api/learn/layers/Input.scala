/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops.data.DatasetIterator
import org.platanios.tensorflow.api.ops.Op

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class Input[T] private(
    private val _dataType: Any,
    private val _shape: Any,
    private val name: String = "Input"
) {
  def dataType[D](implicit evOutputToDataType: OutputToDataType.Aux[T, D]): D = {
    _dataType.asInstanceOf[D]
  }

  def shape[S](implicit evOutputToShape: OutputToShape.Aux[T, S]): S = {
    _shape.asInstanceOf[S]
  }

  private val cache: mutable.Map[Graph, DatasetIterator[T]] = {
    mutable.Map.empty
  }

  protected def create[D, S]()(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): DatasetIterator[T] = {
    DatasetIterator.fromStructure(
      outputDataTypes = dataType,
      outputShapes = shape,
      name = name)
  }

  final def apply[D, S]()(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): DatasetIterator[T] = {
    cache.getOrElse(Op.currentGraph, create())
  }

  def zip[D, S, T2, D2, S2](other: Input[T2])(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S],
      evOutputToDataTypeT2: OutputToDataType.Aux[T2, D2],
      evOutputToShapeT2: OutputToShape.Aux[T2, S2]
  ): Input[(T, T2)] = {
    new Input[(T, T2)](
      _dataType = (dataType, other.dataType),
      _shape = (shape, other.shape),
      name = s"${name}_${other.name}/Zip")
  }
}

object Input {
  def apply[T, D, S](
      dataType: D,
      shape: S,
      name: String = "Input"
  )(implicit
      evDataTypeToOutput: DataTypeToOutput.Aux[D, T],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): Input[T] = {
    new Input[T](dataType, shape, name)
  }

  /** Creates a new [[Input]] without checking that the provided data type and shape are correct. */
  def createUnsafe[T](dataType: Any, shape: Any, name: String = "Input"): Input[T] = {
    new Input[T](dataType, shape, name)
  }

  private[layers] trait API {
    type Input[T] = layers.Input[T]
    val Input: layers.Input.type = layers.Input
  }

  object API extends API
}
