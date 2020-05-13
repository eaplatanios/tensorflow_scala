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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.{OpSpecification, Output}
import org.platanios.tensorflow.api.ops.variables._

/**
  *
  * '''NOTE:''' Subclasses must implement the `forwardWithoutContext` method. Callers should always use either the
  * `forward` or the `apply` methods.
  *
  * @param  name Name scope (also acting as variable scope) for this layer.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Layer[T, R](
    val name: String = null
) {
  val layerType: String

  def forwardWithoutContext(input: T)(implicit mode: Mode): R

  def forward(input: T)(implicit mode: Mode): R = {
    if (name != null) {
      VariableScope.scope(name, isPure = true) {
        forwardWithoutContext(input)
      }
    } else {
      forwardWithoutContext(input)
    }
  }

  def apply(input: T)(implicit mode: Mode): R = forward(input)

  def >>[S](other: Layer[R, S]): Compose[T, R, S] = compose(other)

  def +(other: Layer[T, R]): Concatenate[T, R] = concatenate(other)

  def ++(others: Seq[Layer[T, R]]): Concatenate[T, R] = concatenate(others: _*)

  def compose[S](other: Layer[R, S]): Compose[T, R, S] = Compose(name, this, other)

  def concatenate(others: Layer[T, R]*): Concatenate[T, R] = Concatenate(name, this +: others)

  def map[MR](mapFn: R => MR): Layer[T, MR] = Map(s"$name/Map", this, mapFn)

  def getParameter[P: TF](
      name: String,
      shape: Shape,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): Output[P] = {
    parameterGetter.value[P](
      name, shape, initializer, regularizer,
      trainable, reuse, collections, cachingDevice)
  }

  final def currentStep: Output[Long] = {
    Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false).value
  }

  override def toString: String = layerType
}

object Layer {
  trait API {
    type Layer[T, R] = layers.Layer[T, R]
  }

  object API extends API
}
