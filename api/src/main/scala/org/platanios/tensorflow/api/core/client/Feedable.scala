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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor

import shapeless._

import scala.collection.SeqLike
import scala.language.higherKinds

/** Feedables can be fed into a TensorFlow session to fix the values of certain tensors to the provided values.
  *
  * For example, values for placeholder ops can be fed into TensorFlow sessions.
  *
  * Currently supported feedable types are:
  *   - [[Output]] whose fed value should be a single [[Tensor]].
  *   - [[OutputIndexedSlices]] whose fed value should be tuple [[(Tensor, Tensor, Tensor)]].
  *   - [[SparseOutput]] whose fed value should be tuple [[(Tensor, Tensor, Tensor)]].
  *
  * Feedables are fed into TensorFlow sessions through the use of [[FeedMap]]s. Any [[Map]] that uses a feedable type as
  * the keys type and its corresponding value type as its values type is a valid feed map.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Feedable[T] {
  type ValueType
  def feed(feedable: T, value: ValueType): Map[Output, Tensor]
}

object Feedable {
  type Aux[T, V] = Feedable[T] {type ValueType = V}

  def apply[T, V](implicit ev: Aux[T, V]): Aux[T, V] = ev

  implicit val outputFeedable: Aux[Output, Tensor] = new Feedable[Output] {
    override type ValueType = Tensor
    override def feed(feedable: Output, value: ValueType): Map[Output, Tensor] = Map(feedable -> value)
  }

  // TODO: [TENSORS] Switch to something like "TensorIndexedSlices".
  implicit val outputIndexedSlicesFeedable: Aux[OutputIndexedSlices, (Tensor, Tensor, Tensor)] = {
    new Feedable[OutputIndexedSlices] {
      override type ValueType = (Tensor, Tensor, Tensor)
      override def feed(feedable: OutputIndexedSlices, value: ValueType): Map[Output, Tensor] = {
        Map(feedable.indices -> value._1, feedable.values -> value._2, feedable.denseShape -> value._3)
      }
    }
  }

  // TODO: [TENSORS] Switch to something like "SparseTensor".
  implicit val sparseOutputFeedable: Aux[SparseOutput, (Tensor, Tensor, Tensor)] = {
    new Feedable[SparseOutput] {
      override type ValueType = (Tensor, Tensor, Tensor)
      override def feed(feedable: SparseOutput, value: ValueType): Map[Output, Tensor] = {
        Map(feedable.indices -> value._1, feedable.values -> value._2, feedable.denseShape -> value._3)
      }
    }
  }

  implicit def feedableSeq[T, V, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[T, V]): Aux[CC[T], CC[V]] = {
    new Feedable[CC[T]] {
      override type ValueType = CC[V]
      override def feed(feedable: CC[T], value: CC[V]): Map[Output, Tensor] = {
        feedable.toSeq.zip(value.toSeq).foldLeft(Map.empty[Output, Tensor])({
          case (feedMap, pair) => feedMap ++ ev.feed(pair._1, pair._2)
        })
      }
    }
  }

  implicit def feedableArray[T, V](implicit ev: Aux[T, V]): Aux[Array[T], Array[V]] = {
    new Feedable[Array[T]] {
      override type ValueType = Array[V]
      override def feed(feedable: Array[T], value: Array[V]): Map[Output, Tensor] = {
        feedable.toSeq.zip(value.toSeq).foldLeft(Map.empty[Output, Tensor])({
          case (feedMap, pair) => feedMap ++ ev.feed(pair._1, pair._2)
        })
      }
    }
  }

  // Feedable maps are intentionally not allowed because they would be inefficient without any real good use cases.

  implicit val hnil: Aux[HNil, HNil] = new Feedable[HNil] {
    override type ValueType = HNil
    override def feed(feedable: HNil, value: HNil): Map[Output, Tensor] = Map.empty[Output, Tensor]
  }

  implicit def recursiveConstructor[H, R, T <: HList, TO <: HList](implicit
      feedableHead: Lazy[Aux[H, R]],
      feedableTail: Aux[T, TO]
  ): Aux[H :: T, R :: TO] = new Feedable[H :: T] {
    override type ValueType = R :: TO
    override def feed(feedable: H :: T, value: R :: TO): Map[Output, Tensor] = {
      feedableHead.value.feed(feedable.head, value.head) ++ feedableTail.feed(feedable.tail, value.tail)
    }
  }

  implicit def productConstructor[P <: Product, L <: HList, LO <: HList, R](implicit
      genP: Generic.Aux[P, L],
      genR: Generic.Aux[R, LO],
      feedableL: Lazy[Aux[L, LO]]
  ): Aux[P, R] = new Feedable[P] {
    override type ValueType = R
    override def feed(feedable: P, value: R): Map[Output, Tensor] = {
      feedableL.value.feed(genP.to(feedable), genR.to(value))
    }
  }
}

/** Represents TensorFlow feed maps for sessions.
  *
  * TODO: [CLIENT] !!! Use strings as keys.
  *
  * @param  values Map from tensors in a graph to their values.
  */
class FeedMap private[client] (val values: Map[Output, Tensor] = Map.empty) {
  def feed[T, V](feedable: T, value: V)(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(values ++ ev.feed(feedable, value))
  }

  def +(other: FeedMap): FeedMap = FeedMap(values ++ other.values)
}

object FeedMap {
  def apply(values: Map[Output, Tensor]): FeedMap = new FeedMap(values)

  def apply[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(ev.feed(feed._1, feed._2))

  def apply[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(feeds.flatMap { case (k, v) => ev.feed(k, v) })
  }

  def apply(feedMaps: Seq[FeedMap]): FeedMap = feedMaps.reduce(_ + _)

  val empty = new FeedMap()

  implicit def feedMap[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feed)
  implicit def feedMap[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feeds)
  implicit def feedMap(feedMaps: Seq[FeedMap]): FeedMap = FeedMap(feedMaps)
}
