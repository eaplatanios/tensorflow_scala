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

import org.platanios.tensorflow.api.learn._

import scala.collection.generic.CanBuildFrom
import scala.collection.{TraversableLike, mutable}
import scala.language.higherKinds

/**
  * @author Emmanouil Antonios Platanios
  */
trait Layer[T, R] {
  val name: String

  val call: T => R

  def apply(input: T): R = call(input)

  def >>[S](other: Layer[R, S]): ComposedLayer[T, R, S] = compose(other)

  def +(other: Layer[T, R]): ConcatenatedLayer[T, R] = concatenate(other)

  def ++(others: Seq[Layer[T, R]]): ConcatenatedLayer[T, R] = concatenate(others: _*)

  def compose[S](other: Layer[R, S]): ComposedLayer[T, R, S] = ComposedLayer(this, other)

  def concatenate(others: Layer[T, R]*): ConcatenatedLayer[T, R] = ConcatenatedLayer(this +: others)

  override def toString: String = name
}

object Layer {
  def identity[T](name: String = "Identity"): IdentityLayer[T] = IdentityLayer[T](name)

  implicit class MappableLayer[T, R, CC[A] <: TraversableLike[A, CC[A]]] private[learn](
      layer: Layer[CC[T], CC[R]]) extends Layer[CC[T], CC[R]] {
    override val name: String = layer.name
    override val call: CC[T] => CC[R] = layer.call(_)

    def map[S](layer: Layer[CC[T], CC[R]], function: (R) => S)
        (implicit cbf: CanBuildFrom[CC[R], S, CC[S]]): MappedLayer[T, R, S, CC] = {
      MappedLayer[T, R, S, CC](layer, function)(cbf)
    }
  }

  // implicit class IndexableLayer[T, R, CC[A] <: SeqLike[A, CC[A]]] private[learn](
  //     layer: Layer[T, CC[R]]) extends Layer[T, CC[R]] {
  //   override val name: String = layer.name
  //   override def forward(input: T): CC[R] = layer(input)
  //
  //   def output(index: Int): IndexedLayerOutput[T, R, CC] = IndexedLayerOutput[T, R, CC](layer, index)
  // }

  trait API {
    type Layer[T, R] = layers.Layer[T, R]
    type CachedLayer[T, R] = layers.CachedLayer[T, R]
    type IdentityLayer[T] = layers.IdentityLayer[T]
    type ComposedLayer[T, R, S] = layers.ComposedLayer[T, R, S]
    type ConcatenatedLayer[T, R] = layers.ConcatenatedLayer[T, R]
    type MappedLayer[T, R, S, CC[A] <: TraversableLike[A, CC[A]]] = layers.MappedLayer[T, R, S, CC]
    // type IndexedLayerOutput[T, R, CC[A] <: SeqLike[A, CC[A]]] = layers.IndexedLayerOutput[T, R, CC]

    def Identity[T](name: String = "Identity"): layers.IdentityLayer[T] = layers.IdentityLayer[T](name)
  }

  object API extends API
}

trait CachedLayer[T, R] extends Layer[T, R] {
  protected      val cache    : mutable.Map[T, R] = mutable.HashMap.empty[T, R]
  protected lazy val callProxy: T => R            = call
  override final val call     : T => R            = input => cache.getOrElseUpdate(input, callProxy(input))
}

case class IdentityLayer[T] private[learn](override val name: String = "Identity") extends Layer[T, T] {
  override val call: T => T = identity[T]
}

case class ComposedLayer[T, R, S] private[learn](layer1: Layer[T, R], layer2: Layer[R, S]) extends Layer[T, S] {
  override val name: String = s"$layer1>>$layer2"
  override val call: T => S = input => layer2.call(layer1.call(input))
}

case class ConcatenatedLayer[T, R] private[learn](layers: Seq[Layer[T, R]]) extends Layer[T, Seq[R]] {
  override val name: String      = layers.map(_.name).mkString("+")
  override val call: T => Seq[R] = input => layers.map(_ (input))
}

case class MappedLayer[T, R, S, CC[A] <: TraversableLike[A, CC[A]]] private[learn](
    layer: Layer[CC[T], CC[R]], function: R => S)(implicit cbf: CanBuildFrom[CC[R], S, CC[S]])
    extends Layer[CC[T], CC[S]] {
  override val name: String         = s"Mapped[$layer]"
  override val call: CC[T] => CC[S] = input => layer.call(input).map[S, CC[S]](function)(cbf)
}

//case class IndexedLayerOutput[T, R, CC[A] <: SeqLike[A, CC[A]]] private[learn](
//    layer: Layer[T, CC[R]], index: Int) extends Layer[T, R] {
//  override val name: String = s"$layer($index)"
//
//  @throws[IndexOutOfBoundsException]
//  override def forward(input: T): R = {
//    val outputs = layer(input)
//    if (outputs.length > index)
//      outputs(index)
//    else
//      throw new IndexOutOfBoundsException(
//        s"The provided index ($index) is larger than the number of outputs of layer '$layer' (${outputs.length}).")
//  }
//}
