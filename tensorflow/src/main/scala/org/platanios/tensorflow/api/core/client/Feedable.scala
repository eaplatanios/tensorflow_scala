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

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors.Tensor

/** Feedables can be fed into a TensorFlow session to fix the values of certain tensors to the provided values.
  *
  * For example, values for placeholder ops can be fed into TensorFlow sessions.
  *
  * Currently supported feedable types are:
  *   - [[Op.Output]] whose fed value should be a single [[Tensor]].
  *   - [[Op.OutputIndexedSlices]] whose fed value should be tuple [[(Tensor, Tensor, Tensor)]].
  *   - [[Op.SparseOutput]] whose fed value should be tuple [[(Tensor, Tensor, Tensor)]].
  *
  * Feedables are fed into TensorFlow sessions through the use of [[FeedMap]]s. Any [[Map]] that uses a feedable type as
  * the keys type and its corresponding value type as its values type is a valid feed map.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Feedable[T] {
  type ValueType
  def feed(feedable: T, value: ValueType): Map[Op.Output, Tensor]
}

object Feedable {
  type Aux[T, V] = Feedable[T] {type ValueType = V}

  def apply[T, V](implicit ev: Aux[T, V]): Aux[T, V] = ev

  implicit val opOutputFeedable: Aux[Op.Output, Tensor] = new Feedable[Op.Output] {
    override type ValueType = Tensor
    override def feed(feedable: Op.Output, value: ValueType): Map[Op.Output, Tensor] = Map(feedable -> value)
  }

  // TODO: [TENSORS] Switch to something like "TensorIndexedSlices".
  implicit val opOutputIndexedSlicesFeedable: Aux[Op.OutputIndexedSlices, (Tensor, Tensor, Tensor)] = {
    new Feedable[Op.OutputIndexedSlices] {
      override type ValueType = (Tensor, Tensor, Tensor)
      override def feed(feedable: Op.OutputIndexedSlices, value: ValueType): Map[Op.Output, Tensor] = {
        Map(feedable.indices -> value._1, feedable.values -> value._2, feedable.denseShape -> value._3)
      }
    }
  }

  // TODO: [TENSORS] Switch to something like "SparseTensor".
  implicit val opSparseOutputFeedable: Aux[Op.SparseOutput, (Tensor, Tensor, Tensor)] = {
    new Feedable[Op.SparseOutput] {
      override type ValueType = (Tensor, Tensor, Tensor)
      override def feed(feedable: Op.SparseOutput, value: ValueType): Map[Op.Output, Tensor] = {
        Map(feedable.indices -> value._1, feedable.values -> value._2, feedable.denseShape -> value._3)
      }
    }
  }
}

/** Represents TensorFlow feed maps for sessions.
  *
  * @param  values Map from tensors in a graph to their values.
  */
class FeedMap private[client] (val values: Map[Op.Output, Tensor] = Map.empty) {
  def feed[T, V](feedable: T, value: V)(implicit ev: Feedable.Aux[T, V]): FeedMap = {
    FeedMap(values ++ ev.feed(feedable, value))
  }

  def +(other: FeedMap): FeedMap = FeedMap(values ++ other.values)
}

object FeedMap {
  def apply(values: Map[Op.Output, Tensor]): FeedMap = new FeedMap(values)

  val empty = new FeedMap()

  trait Implicits {
    implicit def feedMap[T, V](feeds: Map[T, V])(implicit ev: Feedable.Aux[T, V]): FeedMap = {
      FeedMap(feeds.toSeq.map(f => ev.feed(f._1, f._2)).reduce(_ ++ _))
    }
  }

  object Implicits extends Implicits
}
