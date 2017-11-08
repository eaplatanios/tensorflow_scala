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
  trait API {
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

case class Identity[T] private[learn](override protected val name: String = "Identity") extends Layer[T, T](name) {
  override val layerType = "Identity"

  override def forward(input: T, mode: Mode): LayerInstance[T, T] = {
    LayerInstance(input, input)
  }
}

case class Compose[T, R, S] private[learn] (
    layer1: Layer[T, R], layer2: Layer[R, S],
    override protected val name: String = "Compose"
) extends Layer[T, S](name) {
  override val layerType: String = s"Compose[$layer1>>$layer2]"

  override def forward(input: T, mode: Mode): LayerInstance[T, S] = {
    val layer1Instance = layer1.forward(input, mode)
    val layer2Instance = layer2.forward(layer1Instance.output, mode)
    LayerInstance(
      input, layer2Instance.output,
      layer1Instance.trainableVariables ++ layer2Instance.trainableVariables,
      layer1Instance.nonTrainableVariables ++ layer2Instance.nonTrainableVariables,
      layer1Instance.graph)
  }
}

case class Concatenate[T, R] private[learn] (
    layers: Seq[Layer[T, R]],
    override protected val name: String = "Concatenate"
) extends Layer[T, Seq[R]](name) {
  override val layerType: String = layers.map(_.uniquifiedName).mkString("+")

  override def forward(input: T, mode: Mode): LayerInstance[T, Seq[R]] = {
    val layerInstances = layers.map(_ (input, mode))
    LayerInstance(
      input, layerInstances.map(_.output),
      layerInstances.flatMap(_.trainableVariables).toSet,
      layerInstances.flatMap(_.nonTrainableVariables).toSet,
      layerInstances.head.graph)
  }
}

case class Map[T, R, S, CC[A] <: TraversableLike[A, CC[A]]] private[learn] (
    layer: Layer[CC[T], CC[R]],
    mapLayer: Layer[R, S],
    override protected val name: String = "Map"
)(implicit
    cbfSS: CanBuildFrom[CC[LayerInstance[R, S]], S, CC[S]],
    cbfLIRS: CanBuildFrom[CC[R], LayerInstance[R, S], CC[LayerInstance[R, S]]]
) extends Layer[CC[T], CC[S]](name) {
  override val layerType: String = s"Map[$layer]"

  override def forward(input: CC[T], mode: Mode): LayerInstance[CC[T], CC[S]] = {
    val layerInstance = layer.forward(input, mode)
    val mappedInstances = layerInstance.output
        .asInstanceOf[TraversableLike[R, CC[R]]]
        .map[LayerInstance[R, S], CC[LayerInstance[R, S]]](mapLayer.forward(_, mode))(cbfLIRS)
    LayerInstance(
      input,
      mappedInstances
          .asInstanceOf[TraversableLike[LayerInstance[R, S], CC[LayerInstance[R, S]]]]
          .map[S, CC[S]](_.output)(cbfSS),
      layerInstance.trainableVariables ++ mappedInstances.flatMap(_.trainableVariables),
      layerInstance.nonTrainableVariables ++ mappedInstances.flatMap(_.nonTrainableVariables),
      layerInstance.graph)
  }
}

case class Squeeze(axes: Seq[Int] = null, override protected val name: String = "Squeeze")
    extends Layer[Output, Output](name) {
  override val layerType: String = if (axes != null) s"Squeeze[${axes.mkString(", ")}]" else "Squeeze"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Basic.squeeze(input, axes, name = uniquifiedName))
  }
}

case class Flatten(override protected val name: String = "Flatten")
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Flatten"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val output = {
      if (input.rank == 1)
        input
      else if (input.rank > -1 && input.shape(0) > -1)
        ops.Basic.reshape(input, Shape(input.shape(0), -1), name = uniquifiedName)
      else
        ops.Basic.reshape(input, Shape(-1) + input.shape.asArray.tail.product, name = uniquifiedName)
    }
    LayerInstance(input, output)
  }
}

case class Transpose(permutation: Seq[Int], override protected val name: String = "Transpose")
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Transpose[${permutation.mkString(", ")}]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Basic.transpose(input, permutation, name = uniquifiedName))
  }
}

case class OneHot(numberOfLabels: Int, override protected val name: String = "OneHot")
    extends Layer[Output, Output](name) {
  override val layerType: String = s"OneHot[$numberOfLabels]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Basic.oneHot(input, numberOfLabels, name = uniquifiedName))
  }
}
