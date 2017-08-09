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

import org.platanios.tensorflow.api.{learn, tf}
import org.platanios.tensorflow.api.core.client.Feedable

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Input {
  trait API {
    type Input = learn.Input
    type InputWithDefault = learn.InputWithDefault

    def input(dataType: tf.DataType, shape: tf.Shape, name: String = "Input"): Input = {
      new Input(dataType = dataType, shape = shape, name = name)
    }

    def inputWithDefault(default: tf.Tensor, shape: tf.Shape, name: String = "InputWithDefault"): InputWithDefault = {
      new InputWithDefault(default = default, shape = shape, name = name)
    }
  }

  object API extends API

  trait DefaultValueSetter[V, S] {
    def set(value: V): tf.Op

    // The dummy implicit here helps deal with the fact that type erasure of the generic types causes issues for
    // overloaded method that have generically-typed arguments.
    def set(value: S)(implicit d: DummyImplicit): tf.Op
  }

  case class VariableDefaultValueSetter private[learn](variable: tf.Variable)
      extends DefaultValueSetter[tf.Tensor, tf.Output] {
    override def set(value: tf.Tensor): tf.Op = variable.assign(tf.constant(value, variable.dataType)).op
    override def set(value: tf.Output)(implicit d: DummyImplicit): tf.Op = variable.assign(value).op
  }

  case class SeqVariableDefaultValueSetter private[learn](variables: Seq[tf.Variable])
      extends DefaultValueSetter[Seq[tf.Tensor], Seq[tf.Output]] {
    override def set(values: Seq[tf.Tensor]): tf.Op = {
      tf.group(
        variables.zip(values)
            .map({ case (variable, value) => variable.assign(tf.constant(value, variable.dataType)).op })
            .toSet)
    }

    override def set(values: Seq[tf.Output])(implicit d: DummyImplicit): tf.Op = {
      tf.group(variables.zip(values).map({ case (variable, value) => variable.assign(value).op }).toSet)
    }
  }
}

sealed abstract class SupportedInput[V, S](implicit ev: Feedable.Aux[S, V]) {
  private[this] val cache: mutable.Map[tf.Graph, (S, Option[Input.DefaultValueSetter[V, S]])] = mutable.Map.empty

  val needsFeeding: Boolean

  protected def create(): (S, Option[Input.DefaultValueSetter[V, S]])

  final def apply(): (S, Option[Input.DefaultValueSetter[V, S]]) = cache.getOrElse(tf.currentGraph, create())
}

class Input private[learn](val dataType: tf.DataType, val shape: tf.Shape, val name: String)
    extends SupportedInput[tf.Tensor, tf.Output] {
  override val needsFeeding: Boolean = true
  override protected def create(): (tf.Output, Option[Input.DefaultValueSetter[tf.Tensor, tf.Output]]) = {
    if (shape.isFullyDefined) {
      val defaultVariable = tf.createWithVariableScope("InputDefaults") {
        tf.variable(name = name, dataType = dataType, shape = shape, initializer = tf.zerosInitializer)
      }
      val placeholder = tf.placeholderWithDefault(default = defaultVariable.value, shape = shape, name = name)
      (placeholder, Some(Input.VariableDefaultValueSetter(defaultVariable)))
    } else {
      (tf.placeholder(dataType, shape, name), None)
    }
  }
}

class InputWithDefault private[learn](default: tf.Tensor, override val shape: tf.Shape, override val name: String)
    extends Input(default.dataType, shape, name) {
  override val needsFeeding: Boolean = false
  override protected def create(): (tf.Output, Option[Input.DefaultValueSetter[tf.Tensor, tf.Output]]) = {
    if (shape.isFullyDefined) {
      val defaultVariable = tf.createWithVariableScope("InputDefaults") {
        tf.variable(name = name, dataType = dataType, shape = shape, initializer = tf.constantInitializer(default))
      }
      val placeholder = tf.placeholderWithDefault(default = defaultVariable.value, shape = shape, name = name)
      (placeholder, Some(Input.VariableDefaultValueSetter(defaultVariable)))
    } else {
      (tf.placeholderWithDefault(default, shape, name), None)
    }
  }
}

//class Inputs private[learn](val dataTypes: Seq[tf.DataType], val shapes: Seq[tf.Shape], val names: Seq[String])
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
