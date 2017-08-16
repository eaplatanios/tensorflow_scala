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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.{DataType, Shape, learn, tf}
import org.platanios.tensorflow.api.ops.io.Data

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Input {
  trait API {
    type Input = learn.Input

    def input(dataType: DataType, shape: Shape, name: String = "Input"): Input = {
      new Input(dataType = dataType, shape = shape, name = name)
    }
  }

  object API extends API
}

sealed abstract class SupportedInput[O, D, S](implicit ev: Data.Aux[_, O, D, S]) {
  private[this] val cache: mutable.Map[tf.Graph, tf.Iterator[O, D, S]] = mutable.Map.empty

  protected def create(): tf.Iterator[O, D, S]

  final def apply(): tf.Iterator[O, D, S] = cache.getOrElse(tf.currentGraph, create())
}

class Input private[learn](val dataType: DataType, val shape: Shape, val name: String)
    extends SupportedInput[tf.Output, DataType, Shape] {
  override protected def create(): tf.Iterator[tf.Output, DataType, Shape] = {
    tf.iteratorFromStructure(outputDataTypes = dataType, outputShapes = shape)
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
