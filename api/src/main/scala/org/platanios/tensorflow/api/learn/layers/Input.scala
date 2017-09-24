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
  trait API {
    type Input[T, O, D, S] = layers.Input[T, O, D, S]

    def input(dataType: DataType, shape: Shape): Input[Tensor, Output, DataType, Shape] = {
      Input(dataType, shape, "Input")
    }

    def input(dataType: DataType, shape: Shape, name: String): Input[Tensor, Output, DataType, Shape] = {
      Input(dataType, shape, name)
    }
  }

  object API extends API
}

case class Input[T, O, D, S] private[learn](dataType: D, shape: S, name: String)(implicit
    ev: Data.Aux[T, O, D, S]
) {
  private[Input] val evidence: Data.Aux[T, O, D, S] = ev

  private[this] val cache: mutable.Map[Graph, Iterator[T, O, D, S]] = mutable.Map.empty

  protected def create(): Iterator[T, O, D, S] = Iterator.fromStructure(dataType, shape, name)

  final def apply(): Iterator[T, O, D, S] = cache.getOrElse(Op.currentGraph, create())

  private[learn] def zip[T2, O2, D2, S2](other: Input[T2, O2, D2, S2]):
  Input[(T, T2), (O, O2), (D, D2), (S, S2)] = {
    implicit val ev2: Data.Aux[T2, O2, D2, S2] = other.evidence
    Input[(T, T2), (O, O2), (D, D2), (S, S2)](
      (dataType, other.dataType), (shape, other.shape), s"${name}_${other.name}/Zip")
  }
}

//class Inputs private[learn](val dataTypes: Seq[DataType], val shapes: Seq[tf.Shape], val names: Seq[String])
//    extends SupportedInput[Seq[tf.Tensor], Seq[tf.Output]] {
//  override val needsFeeding: Boolean = true
//  override protected def create(): (Seq[tf.Output], Input.DefaultValueSetter[Seq[tf.Tensor], Seq[tf.Output]]) = {
//    val (placeholders, variables) = (dataTypes, shapes, names).zipped.map({
//      case (dataType, shape, name) =>
//        val defaultVariable = tf.createWithVariableScope("InputDefaults") {
//          tf.variable(name = name, dataType = dataType, shape = shape, initializer = tf.zerosInitializer)
//        }
//        val placeholder = tf.placeholderWithDefault(default = defaultVariable.value, shape = shape, name = name)
//        (placeholder, defaultVariable)
//    }).unzip
//    (placeholders, Input.SeqVariableDefaultValueSetter(variables))
//  }
//}
//
//class InputsWithDefaults private[learn](
//    defaults: Seq[tf.Tensor], override val shapes: Seq[tf.Shape], override val names: Seq[String])
//    extends Inputs(defaults.map(_.dataType), shapes, names) {
//  override val needsFeeding: Boolean = false
//  override protected def create(): (Seq[tf.Output], Input.DefaultValueSetter[Seq[tf.Tensor], Seq[tf.Output]]) = {
//    val (placeholders, variables) = (defaults, shapes, names).zipped.map({
//      case (default, shape, name) =>
//        val defaultVariable = tf.createWithVariableScope("InputDefaults") {
//          tf.variable(
//            name = name, dataType = default.dataType, shape = shape, initializer = tf.constantInitializer(default))
//        }
//        val placeholder = tf.placeholderWithDefault(default = defaultVariable.value, shape = shape, name = name)
//        (placeholder, defaultVariable)
//    }).unzip
//    (placeholders, Input.SeqVariableDefaultValueSetter(variables))
//  }
//}
