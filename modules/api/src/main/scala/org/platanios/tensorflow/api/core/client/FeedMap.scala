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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToTensor, TensorStructure}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

import scala.language.higherKinds

/** Represents TensorFlow feed maps for sessions. Feed maps are fed into a TensorFlow session to fix the values of
  * certain tensors to the provided values.
  *
  * TODO: [CLIENT] !!! Use strings as keys.
  *
  * @param  values Map from tensors in a graph to their values.
  */
class FeedMap private[client](val values: Map[Output[_], Tensor[_]] = Map.empty) {
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

  def apply[T, V](feed: (T, V))(implicit
      outputStructure: OutputStructure[T],
      outputToTensor: OutputToTensor.Aux[T, V],
      tensorStructure: TensorStructure[V],
  ): FeedMap = {
    new FeedMap(outputStructure.outputs(feed._1).zip(tensorStructure.tensors(feed._2)).toMap)
  }

  def apply[T, V](feeds: Map[T, V])(implicit
      outputStructure: OutputStructure[T],
      outputToTensor: OutputToTensor.Aux[T, V],
      tensorStructure: TensorStructure[V],
  ): FeedMap = {
    new FeedMap(feeds.flatMap { case (k, v) => outputStructure.outputs(k).zip(tensorStructure.tensors(v)).toMap })
  }

  def apply(feedMaps: Seq[FeedMap]): FeedMap = {
    feedMaps.reduce(_ ++ _)
  }

  val empty = new FeedMap()

  private[client] trait Implicits {
    implicit def feedMap(feeds: Map[Output[_], Tensor[_]]): FeedMap = FeedMap(feeds)

    implicit def feedMap[T, V](feed: (T, V))(implicit
        outputStructure: OutputStructure[T],
        outputToTensor: OutputToTensor.Aux[T, V],
        tensorStructure: TensorStructure[V],
    ): FeedMap = FeedMap(feed)

    implicit def feedMap[T, V](feeds: Map[T, V])(implicit
        outputStructure: OutputStructure[T],
        outputToTensor: OutputToTensor.Aux[T, V],
        tensorStructure: TensorStructure[V],
    ): FeedMap = FeedMap(feeds)

    implicit def feedMap(feedMaps: Seq[FeedMap]): FeedMap = FeedMap(feedMaps)
  }
}
