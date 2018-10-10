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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import scala.language.higherKinds

/** Feedables can be fed into a TensorFlow session to fix the values of certain tensors to the provided values.
  *
  * For example, values for placeholder ops can be fed into TensorFlow sessions.
  *
  * Feedables are fed into TensorFlow sessions through the use of [[FeedMap]]s. Any [[Map]] that uses a feedable type as
  * the keys type and its corresponding value type as its values type is a valid feed map.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Feedable[T] {
  type ValueType

  def feed(feedable: T, value: ValueType): Map[Output[_], Tensor[_]]
}

object Feedable {
  type Aux[T, V] = Feedable[T] {type ValueType = V}

  def apply[T, V](implicit ev: Aux[T, V]): Aux[T, V] = ev

  implicit def fromOutput[T: TF]: Aux[Output[T], Tensor[T]] = {
    new Feedable[Output[T]] {
      override type ValueType = Tensor[T]

      override def feed(
          feedable: Output[T],
          value: Tensor[T]
      ): Map[Output[_], Tensor[_]] = {
        Map(feedable -> value)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T: TF]: Aux[OutputIndexedSlices[T], TensorIndexedSlices[T]] = {
    new Feedable[OutputIndexedSlices[T]] {
      override type ValueType = TensorIndexedSlices[T]

      override def feed(
          feedable: OutputIndexedSlices[T],
          value: TensorIndexedSlices[T]
      ): Map[Output[_], Tensor[_]] = {
        Map(
          feedable.indices -> value.indices,
          feedable.values -> value.values,
          feedable.denseShape -> value.denseShape)
      }
    }
  }

  implicit def fromSparseOutput[T: TF]: Aux[SparseOutput[T], SparseTensor[T]] = {
    new Feedable[SparseOutput[T]] {
      override type ValueType = SparseTensor[T]

      override def feed(
          feedable: SparseOutput[T],
          value: SparseTensor[T]
      ): Map[Output[_], Tensor[_]] = {
        Map(
          feedable.indices -> value.indices,
          feedable.values -> value.values,
          feedable.denseShape -> value.denseShape)
      }
    }
  }
}

/** Represents TensorFlow feed maps for sessions.
  *
  * TODO: [CLIENT] !!! Use strings as keys.
  *
  * @param  values Map from tensors in a graph to their values.
  */
class FeedMap private[client](val values: Map[Output[_], Tensor[_]] = Map.empty) {
  def feed[T, V](feedable: T, value: V)(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(values ++ ev.feed(feedable, value))
  }

  def +(value: (Output[_], Tensor[_])): FeedMap = {
    FeedMap(values + value)
  }

  def ++(other: FeedMap): FeedMap = {
    FeedMap(values ++ other.values)
  }

  def isEmpty: Boolean = values.isEmpty

  def nonEmpty: Boolean = values.nonEmpty

  /** Returns `true` if this feed map feeds at least one element that `feedMap` also feeds. */
  def intersects(feedMap: FeedMap): Boolean = {
    values.keySet.exists(feedMap.values.keySet.contains(_))
  }
}

object FeedMap {
  def apply(values: Map[Output[_], Tensor[_]]): FeedMap = {
    new FeedMap(values)
  }

  def apply[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(ev.feed(feed._1, feed._2))
  }

  def apply[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(feeds.flatMap { case (k, v) => ev.feed(k, v) })
  }

  def apply(feedMaps: Seq[FeedMap]): FeedMap = {
    feedMaps.reduce(_ ++ _)
  }

  val empty = new FeedMap()

  private[client] trait Implicits {
    implicit def feedMap(feeds: Map[Output[_], Tensor[_]]): FeedMap = FeedMap(feeds)
    implicit def feedMap[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feed)
    implicit def feedMap[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feeds)
    implicit def feedMap(feedMaps: Seq[FeedMap]): FeedMap = FeedMap(feedMaps)
  }
}
