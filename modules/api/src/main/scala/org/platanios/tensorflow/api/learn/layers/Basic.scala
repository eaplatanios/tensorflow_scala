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
import org.platanios.tensorflow.api.core.types.{IsInt32OrInt64OrUInt8, TF}
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
    type Map[T, R, MR] = layers.Map[T, R, MR]
    type MapSeq[T, R, S, CC[A] <: TraversableLike[A, CC[A]]] = layers.MapSeq[T, R, S, CC]
    type Squeeze[T] = layers.Squeeze[T]
    type Stack[T] = layers.Stack[T]
    type Flatten[T] = layers.Flatten[T]
    type Reshape[T] = layers.Reshape[T]
    type Transpose[T] = layers.Transpose[T]
    type OneHot[T, I] = layers.OneHot[T, I]

    val Identity : layers.Identity.type  = layers.Identity
    val Map      : layers.Map.type       = layers.Map
    val MapSeq   : layers.MapSeq.type    = layers.MapSeq
    val Squeeze  : layers.Squeeze.type   = layers.Squeeze
    val Stack    : layers.Stack.type     = layers.Stack
    val Flatten  : layers.Flatten.type   = layers.Flatten
    val Reshape  : layers.Reshape.type   = layers.Reshape
    val Transpose: layers.Transpose.type = layers.Transpose
    val OneHot   : layers.OneHot.type    = layers.OneHot
  }

  object API extends API
}

case class Identity[T](override val name: String)
    extends Layer[T, T](name) {
  override val layerType = "Identity"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): T = {
    input
  }
}

case class Compose[T, R, S](
    override val name: String,
    layer1: Layer[T, R], layer2: Layer[R, S]
) extends Layer[T, S](name) {
  override val layerType: String = s"Compose[$layer1>>$layer2]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): S = {
    layer2(layer1(input))
  }
}

case class Concatenate[T, R](
    override val name: String,
    layers: Seq[Layer[T, R]]
) extends Layer[T, Seq[R]](name) {
  override val layerType: String = "Concatenate"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): Seq[R] = layers.map(_ (input))
}

case class Map[T, R, MR](
    override val name: String,
    layer: Layer[T, R],
    mapFn: R => MR
) extends Layer[T, MR](name) {
  override val layerType: String = s"Map[$layer]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): MR = {
    mapFn(layer(input))
  }
}

case class MapSeq[T, R, S, CC[A] <: TraversableLike[A, CC[A]]](
    override val name: String,
    layer: Layer[CC[T], CC[R]],
    mapLayer: Layer[R, S]
)(implicit
    cbfRS: CanBuildFrom[CC[R], S, CC[S]]
) extends Layer[CC[T], CC[S]](name) {
  override val layerType: String = s"Map[$layer]"

  override def forwardWithoutContext(input: CC[T])(implicit mode: Mode): CC[S] = {
    layer(input)
        .asInstanceOf[TraversableLike[R, CC[R]]]
        .map[S, CC[S]](mapLayer(_))(cbfRS)
  }
}

case class Squeeze[T: TF](
    override val name: String,
    axes: Seq[Int] = null
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = if (axes != null) s"Squeeze[${axes.mkString(", ")}]" else "Squeeze"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Basic.squeeze(input, axes, name = name)
  }
}

case class Stack[T: TF](
    override val name: String, axis: Int = 0
) extends Layer[Seq[Output[T]], Output[T]](name) {
  override val layerType: String = s"Stack[axis=$axis]"

  override def forwardWithoutContext(
      input: Seq[Output[T]]
  )(implicit mode: Mode): Output[T] = {
    ops.Basic.stack(input, axis, name = name)
  }
}

case class Flatten[T: TF](
    override val name: String
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Flatten"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    if (input.rank == 1) {
      input
    } else if (input.rank > -1 && input.shape(0) > -1) {
      ops.Basic.reshape(input, Shape(input.shape(0), -1), name = name)
    } else {
      ops.Basic.reshape(input, Shape(-1) + input.shape.asArray.tail.product, name = name)
    }
  }
}

case class Reshape[T: TF](
    override val name: String,
    shape: Shape
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Reshape[${shape.asArray.mkString(", ")}]"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Basic.reshape(input, shape, name = name)
  }
}

case class Transpose[T: TF](
    override val name: String,
    permutation: Seq[Int]
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Transpose[${permutation.mkString(", ")}]"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Basic.transpose(input, permutation, name = name)
  }
}

case class OneHot[T: TF, I: TF : IsInt32OrInt64OrUInt8](
    override val name: String,
    numberOfLabels: Int
) extends Layer[Output[I], Output[T]](name) {
  override val layerType: String = s"OneHot[$numberOfLabels]"

  override def forwardWithoutContext(
      input: Output[I]
  )(implicit mode: Mode): Output[T] = {
    ops.Basic.oneHot[T, I](input, numberOfLabels, name = name)
  }
}
