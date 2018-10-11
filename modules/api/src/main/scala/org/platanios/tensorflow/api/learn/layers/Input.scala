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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, StructureFromDataType}
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
  def dataType[D, S](implicit evT: NestedStructure.Aux[T, D, S]): D = {
    _dataType.asInstanceOf[D]
  }

  def shape[D, S](implicit evT: NestedStructure.Aux[T, D, S]): S = {
    _shape.asInstanceOf[S]
  }

  private val cache: mutable.Map[Graph, DatasetIterator[T]] = {
    mutable.Map.empty
  }

  protected def create[D, S]()(implicit evT: NestedStructure.Aux[T, D, S]): DatasetIterator[T] = {
    DatasetIterator.fromStructure(
      outputDataTypes = dataType,
      outputShapes = shape,
      name = name)
  }

  final def apply[D, S]()(implicit evT: NestedStructure.Aux[T, D, S]): DatasetIterator[T] = {
    cache.getOrElse(Op.currentGraph, create())
  }

  def zip[D, S, T2, D2, S2](other: Input[T2])(implicit
      evT: NestedStructure.Aux[T, D, S],
      evT2: NestedStructure.Aux[T2, D2, S2]
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
      evStructure: StructureFromDataType.Aux[_, T, D, S],
      evT: NestedStructure.Aux[T, D, S]
  ): Input[T] = {
    new Input[T](dataType, shape, name)
  }

  private[layers] trait API {
    type Input[T] = layers.Input[T]
    val Input: layers.Input.type = layers.Input
  }

  object API extends API
}
