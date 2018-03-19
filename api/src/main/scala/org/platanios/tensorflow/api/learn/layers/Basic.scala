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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

import scala.collection.TraversableLike
import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds

/**
  * @author Emmanouil Antonios Platanios
  */
object Basic {
  private[layers] trait API {
    type Identity[T] = layers.Identity[T]
    type Compose[T, R, S] = layers.Compose[T, R, S]
    type Concatenate[T, R] = layers.Concatenate[T, R]
    type Map[T, R, S, CC[A] <: TraversableLike[A, CC[A]]] = layers.Map[T, R, S, CC]
    type Squeeze = layers.Squeeze
    type Flatten = layers.Flatten
    type Transpose = layers.Transpose
    type OneHot = layers.OneHot

    val Identity : layers.Identity.type  = layers.Identity
    val Squeeze  : layers.Squeeze.type   = layers.Squeeze
    val Flatten  : layers.Flatten.type   = layers.Flatten
    val Transpose: layers.Transpose.type = layers.Transpose
    val OneHot   : layers.OneHot.type    = layers.OneHot
  }

  object API extends API
}

case class Identity[T](override val name: String)
    extends Layer[T, T](name) {
  override val layerType = "Identity"

  override protected def _forward(input: T)(implicit mode: Mode): T = input
}

case class Compose[T, R, S](
    override val name: String,
    layer1: Layer[T, R], layer2: Layer[R, S]
) extends Layer[T, S](name) {
  override val layerType: String = s"Compose[$layer1>>$layer2]"

  override protected def _forward(input: T)(implicit mode: Mode): S = layer2(layer1(input))
}

case class Concatenate[T, R](
    override val name: String,
    layers: Seq[Layer[T, R]]
) extends Layer[T, Seq[R]](name) {
  override val layerType: String = "Concatenate"

  override protected def _forward(input: T)(implicit mode: Mode): Seq[R] = layers.map(_ (input))
}

case class Map[T, R, S, CC[A] <: TraversableLike[A, CC[A]]](
    override val name: String,
    layer: Layer[CC[T], CC[R]],
    mapLayer: Layer[R, S]
)(implicit
    cbfRS: CanBuildFrom[CC[R], S, CC[S]]
) extends Layer[CC[T], CC[S]](name) {
  override val layerType: String = s"Map[$layer]"

  override protected def _forward(input: CC[T])(implicit mode: Mode): CC[S] = {
    layer(input)
        .asInstanceOf[TraversableLike[R, CC[R]]]
        .map[S, CC[S]](mapLayer(_))(cbfRS)
  }
}

case class Squeeze(override val name: String, axes: Seq[Int] = null)
    extends Layer[Output, Output](name) {
  override val layerType: String = if (axes != null) s"Squeeze[${axes.mkString(", ")}]" else "Squeeze"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    ops.Basic.squeeze(input, axes, name = name)
  }
}

case class Flatten(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Flatten"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    if (input.rank == 1)
      input
    else if (input.rank > -1 && input.shape(0) > -1)
      ops.Basic.reshape(input, Shape(input.shape(0), -1), name = name)
    else
      ops.Basic.reshape(input, Shape(-1) + input.shape.asArray.tail.product, name = name)
  }
}

case class Transpose(override val name: String, permutation: Seq[Int])
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Transpose[${permutation.mkString(", ")}]"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    ops.Basic.transpose(input, permutation, name = name)
  }
}

case class OneHot(override val name: String, numberOfLabels: Int)
    extends Layer[Output, Output](name) {
  override val layerType: String = s"OneHot[$numberOfLabels]"

  override protected def _forward(input: Output)(implicit mode: Mode): Output = {
    ops.Basic.oneHot(input, numberOfLabels, name = name)
  }
}
