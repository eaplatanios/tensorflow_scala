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

import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.types.DataType

import shapeless._
import shapeless.ops.hlist.Tupler

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
  def feed(feedable: T, value: ValueType): Map[Output, Tensor[DataType]]
}

object Feedable {
  type Aux[T, V] = Feedable[T] {type ValueType = V}

  def apply[T, V](implicit ev: Aux[T, V]): Aux[T, V] = ev

  implicit def outputFeedable[D <: DataType]: Aux[Output, Tensor[D]] = new Feedable[Output] {
    override type ValueType = Tensor[D]
    override def feed(feedable: Output, value: ValueType): Map[Output, Tensor[DataType]] = Map(feedable -> value)
  }

  implicit def outputIndexedSlicesFeedable[D <: DataType]: Aux[OutputIndexedSlices, TensorIndexedSlices[D]] = {
    new Feedable[OutputIndexedSlices] {
      override type ValueType = TensorIndexedSlices[D]
      override def feed(feedable: OutputIndexedSlices, value: ValueType): Map[Output, Tensor[DataType]] = {
        Map(feedable.indices -> value.indices, feedable.values -> value.values, feedable.denseShape -> value.denseShape)
      }
    }
  }

  implicit def sparseOutputFeedable[D <: DataType]: Aux[SparseOutput, SparseTensor[D]] = {
    new Feedable[SparseOutput] {
      override type ValueType = SparseTensor[D]
      override def feed(feedable: SparseOutput, value: ValueType): Map[Output, Tensor[DataType]] = {
        Map(feedable.indices -> value.indices, feedable.values -> value.values, feedable.denseShape -> value.denseShape)
      }
    }
  }

  implicit def feedableArray[T, V](implicit ev: Aux[T, V]): Aux[Array[T], Array[V]] = {
    new Feedable[Array[T]] {
      override type ValueType = Array[V]
      override def feed(feedable: Array[T], value: Array[V]): Map[Output, Tensor[DataType]] = {
        feedable.toSeq.zip(value.toSeq).foldLeft(Map.empty[Output, Tensor[DataType]])({
          case (feedMap, pair) => feedMap ++ ev.feed(pair._1, pair._2)
        })
      }
    }
  }

  implicit def feedableSeq[T, V, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[T, V]): Aux[CC[T], CC[V]] = {
    new Feedable[CC[T]] {
      override type ValueType = CC[V]
      override def feed(feedable: CC[T], value: CC[V]): Map[Output, Tensor[DataType]] = {
        feedable.toSeq.zip(value.toSeq).foldLeft(Map.empty[Output, Tensor[DataType]])({
          case (feedMap, pair) => feedMap ++ ev.feed(pair._1, pair._2)
        })
      }
    }
  }

  // Feedable maps are intentionally not allowed because they would be inefficient without any real good use cases.

  implicit val hnil: Aux[HNil, HNil] = new Feedable[HNil] {
    override type ValueType = HNil
    override def feed(feedable: HNil, value: HNil): Map[Output, Tensor[DataType]] = {
      Map.empty[Output, Tensor[DataType]]
    }
  }

  implicit def recursiveConstructor[H, R, T <: HList, TO <: HList](implicit
      feedableHead: Lazy[Aux[H, R]],
      feedableTail: Aux[T, TO]
  ): Aux[H :: T, R :: TO] = new Feedable[H :: T] {
    override type ValueType = R :: TO
    override def feed(feedable: H :: T, value: R :: TO): Map[Output, Tensor[DataType]] = {
      feedableHead.value.feed(feedable.head, value.head) ++ feedableTail.feed(feedable.tail, value.tail)
    }
  }

  implicit def productConstructor[P, R, L <: HList, LO <: HList](implicit
      genP: Generic.Aux[P, L],
      feedableL: Aux[L, LO],
      tuplerR: Tupler.Aux[LO, R],
      genR: Generic.Aux[R, LO]
  ): Aux[P, R] = new Feedable[P] {
    override type ValueType = R
    override def feed(feedable: P, value: R): Map[Output, Tensor[DataType]] = {
      feedableL.feed(genP.to(feedable), genR.to(value))
    }
  }
}

/** Represents TensorFlow feed maps for sessions.
  *
  * TODO: [CLIENT] !!! Use strings as keys.
  *
  * @param  values Map from tensors in a graph to their values.
  */
class FeedMap private[client](val values: Map[Output, Tensor[DataType]] = Map.empty) {
  def feed[T, V](feedable: T, value: V)(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(values ++ ev.feed(feedable, value))
  }

  def +(value: (Output, Tensor[DataType])): FeedMap = FeedMap(values + value)
  def ++(other: FeedMap): FeedMap = FeedMap(values ++ other.values)

  def isEmpty: Boolean = values.isEmpty
  def nonEmpty: Boolean = values.nonEmpty

  /** Returns `true` if this feed map feeds at least one element that `feedMap` also feeds. */
  def intersects(feedMap: FeedMap): Boolean = values.keySet.exists(feedMap.values.keySet.contains(_))
}

object FeedMap {
  def apply(values: Map[Output, Tensor[DataType]]): FeedMap = new FeedMap(values)

  def apply[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(ev.feed(feed._1, feed._2))

  def apply[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(feeds.flatMap { case (k, v) => ev.feed(k, v) })
  }

  def apply(feedMaps: Seq[FeedMap]): FeedMap = feedMaps.reduce(_ ++ _)

  val empty = new FeedMap()

  implicit def feedMap[T, V](feed: (T, V))(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feed)
  implicit def feedMap[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = FeedMap(feeds)
  implicit def feedMap(feedMaps: Seq[FeedMap]): FeedMap = FeedMap(feedMaps)
}
