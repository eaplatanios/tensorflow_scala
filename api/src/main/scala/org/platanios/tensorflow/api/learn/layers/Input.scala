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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops.io.{Data, Iterator}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Input {
  private[layers] trait API {
    type Input[T, O, D, S] = layers.Input[T, O, D, S]
    val Input: layers.Input.type = layers.Input
  }

  object API extends API
}

case class Input[T, O, D, S](dataType: D, shape: S, name: String = "Input")(implicit
    ev: Data.Aux[T, O, D, S]
) {
  private[Input] val evidence: Data.Aux[T, O, D, S] = ev

  private[this] val cache: mutable.Map[Graph, Iterator[T, O, D, S]] = mutable.Map.empty

  protected def create(): Iterator[T, O, D, S] = Iterator.fromStructure(dataType, shape, name)

  final def apply(): Iterator[T, O, D, S] = cache.getOrElse(Op.currentGraph, create())

  def zip[T2, O2, D2, S2](other: Input[T2, O2, D2, S2]):
  Input[(T, T2), (O, O2), (D, D2), (S, S2)] = {
    implicit val ev2: Data.Aux[T2, O2, D2, S2] = other.evidence
    Input[(T, T2), (O, O2), (D, D2), (S, S2)](
      (dataType, other.dataType), (shape, other.shape), s"${name}_${other.name}/Zip")
  }
}
