// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors.Tensor

import scala.collection.mutable

// TODO: [CLIENT] --- Add support for bigger tuples than Tuple6 (up to Tuple22).
// TODO: [CLIENT] Handle nested sequences and maps.

/**
  * @author Emmanouil Antonios Platanios
  */
trait Fetchable[+T] {
  def uniqueFetches: Seq[Op.Output]
  def buildResult(values: Seq[Tensor]): T
}

object Fetchable {
  case class Empty[T]() extends Fetchable[T] {
    override def uniqueFetches: Seq[Op.Output] = Seq.empty
    override def buildResult(values: Seq[Tensor]): T = {
      throw new IllegalArgumentException("Cannot fetch values from an empty fetchable.")
    }
  }

  /** Uniquifies fetches from a sequences of [[Fetchable]]s.
    *
    * This is a helper function used by [[FetchableSeq]] and [[FetchableMap]]. It gathers all the unique fetches from a
    * sequence of fetchables and builds a sequence containing all of them, but without duplicates (`uniqueFetches`).
    *
    * It also returns a nested sequence of integers (`valuesIndices`) indicating at which index in `uniqueFetches` the
    * fetches of the individual fetchables are located. I.e.,
    * {{{
    *   valuesIndices(fetchableIndex)(fetchableFetchIndex) = uniqueFetchesIndex
    * }}}
    *
    * @param  fetchables Sequence of fetchables.
    * @return Tuple containing a sequence containing the unique fetches and a nested sequence of integers containing the
    *         value indices.
    */
  private[client] def uniquifyFetches(fetchables: Seq[Fetchable[_]]): (Seq[Op.Output], Seq[Seq[Int]]) = {
    val uniqueFetches = mutable.ArrayBuffer.empty[Op.Output]
    val seenFetches = mutable.Map.empty[Op.Output, Int]
    val valueIndices = fetchables.map(
      _.uniqueFetches.map(f => seenFetches.getOrElseUpdate(f, {uniqueFetches += f; uniqueFetches.length - 1})))
    (uniqueFetches, valueIndices)
  }

  trait Implicits {
    implicit def fetchableSeq[T](fetchables: Seq[Fetchable[T]]): FetchableSeq[T] = FetchableSeq(fetchables)

    implicit def fetchableMap[K, T](fetchables: Map[K, Fetchable[T]]): FetchableMap[K, T] = FetchableMap[K, T](fetchables)

    implicit def fetchableTuple1[T1](fetchables: Tuple1[Fetchable[T1]]): FetchableTuple1[T1] = FetchableTuple1(fetchables)

    implicit def fetchableTuple2[T1, T2](fetchables: (Fetchable[T1], Fetchable[T2])): FetchableTuple2[T1, T2] = {
      FetchableTuple2(fetchables)
    }

    implicit def fetchableTuple3[T1, T2, T3](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3])): FetchableTuple3[T1, T2, T3] = {
      FetchableTuple3(fetchables)
    }

    implicit def fetchableTuple4[T1, T2, T3, T4](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4])): FetchableTuple4[T1, T2, T3, T4] = {
      FetchableTuple4(fetchables)
    }

    implicit def fetchableTuple5[T1, T2, T3, T4, T5](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4], Fetchable[T5])): FetchableTuple5[T1, T2, T3, T4, T5] = {
      FetchableTuple5(fetchables)
    }
  }

  object Implicits extends Implicits
}

private[client] class FetchableSeq[+T] private (fetchables: Seq[Fetchable[T]]) extends Fetchable[Seq[T]] {
  private[this] val (fetches, indices) = Fetchable.uniquifyFetches(fetchables)

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): Seq[T] = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    fetchables.zip(indices).map(f => f._1.buildResult(f._2.map(values(_))))
  }
}

private[client] object FetchableSeq {
  def apply[T](fetchables: Seq[Fetchable[T]]): FetchableSeq[T] = new FetchableSeq(fetchables)
}

private[client] class FetchableMap[K, +T] private (fetchables: Seq[(K, Fetchable[T])]) extends Fetchable[Map[K, T]] {
  private[this] val (fetches, indices) = Fetchable.uniquifyFetches(fetchables.map(_._2))

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): Map[K, T] = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    fetchables.zip(indices).map(f => f._1._1 -> f._1._2.buildResult(f._2.map(values(_)))).toMap
  }
}

private[client] object FetchableMap {
  def apply[K, T](fetchables: Map[K, Fetchable[T]]): FetchableMap[K, T] = new FetchableMap(fetchables.toSeq)
}

private[client] class FetchableTuple1[+T1] private (fetchables: Tuple1[Fetchable[T1]]) extends Fetchable[Tuple1[T1]] {
  override def uniqueFetches: Seq[Op.Output] = fetchables._1.uniqueFetches

  override def buildResult(values: Seq[Tensor]): Tuple1[T1] = {
    if (values.length != 1)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (1).")
    Tuple1(fetchables._1.buildResult(values))
  }
}

//region Tuples

private[client] object FetchableTuple1 {
  def apply[T1](fetchables: Tuple1[Fetchable[T1]]): FetchableTuple1[T1] = new FetchableTuple1(fetchables)
}

private[client] class FetchableTuple2[+T1, +T2] private (
    fetchables: (Fetchable[T1], Fetchable[T2]))
    extends Fetchable[(T1, T2)] {
  private[this] val (fetches, indices) = Fetchable.uniquifyFetches(Seq(fetchables._1, fetchables._2))

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): (T1, T2) = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    (fetchables._1.buildResult(indices(0).map(values(_))),
        fetchables._2.buildResult(indices(1).map(values(_))))
  }
}

private[client] object FetchableTuple2 {
  def apply[T1, T2](fetchables: (Fetchable[T1], Fetchable[T2])): FetchableTuple2[T1, T2] = {
    new FetchableTuple2(fetchables)
  }
}

private[client] class FetchableTuple3[+T1, +T2, +T3] private (
    fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3]))
    extends Fetchable[(T1, T2, T3)] {
  private[this] val (fetches, indices) = Fetchable.uniquifyFetches(Seq(fetchables._1, fetchables._2, fetchables._3))

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): (T1, T2, T3) = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    (fetchables._1.buildResult(indices(0).map(values(_))),
        fetchables._2.buildResult(indices(1).map(values(_))),
        fetchables._3.buildResult(indices(2).map(values(_))))
  }
}

private[client] object FetchableTuple3 {
  def apply[T1, T2, T3](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3])): FetchableTuple3[T1, T2, T3] = {
    new FetchableTuple3(fetchables)
  }
}

private[client] class FetchableTuple4[+T1, +T2, +T3, +T4] private (
    fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4]))
    extends Fetchable[(T1, T2, T3, T4)] {
  private[this] val (fetches, indices) = {
    Fetchable.uniquifyFetches(Seq(fetchables._1, fetchables._2, fetchables._3, fetchables._4))
  }

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): (T1, T2, T3, T4) = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    (fetchables._1.buildResult(indices(0).map(values(_))),
        fetchables._2.buildResult(indices(1).map(values(_))),
        fetchables._3.buildResult(indices(2).map(values(_))),
        fetchables._4.buildResult(indices(3).map(values(_))))
  }
}

private[client] object FetchableTuple4 {
  def apply[T1, T2, T3, T4](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4])): FetchableTuple4[T1, T2, T3, T4] = {
    new FetchableTuple4(fetchables)
  }
}

private[client] class FetchableTuple5[+T1, +T2, +T3, +T4, +T5] private (
    fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4], Fetchable[T5]))
    extends Fetchable[(T1, T2, T3, T4, T5)] {
  private[this] val (fetches, indices) = {
    Fetchable.uniquifyFetches(Seq(fetchables._1, fetchables._2, fetchables._3, fetchables._4, fetchables._5))
  }

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): (T1, T2, T3, T4, T5) = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    (fetchables._1.buildResult(indices(0).map(values(_))),
        fetchables._2.buildResult(indices(1).map(values(_))),
        fetchables._3.buildResult(indices(2).map(values(_))),
        fetchables._4.buildResult(indices(3).map(values(_))),
        fetchables._5.buildResult(indices(4).map(values(_))))
  }
}

private[client] object FetchableTuple5 {
  def apply[T1, T2, T3, T4, T5](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4], Fetchable[T5])): FetchableTuple5[T1, T2, T3, T4, T5] = {
    new FetchableTuple5(fetchables)
  }
}

private[client] class FetchableTuple6[+T1, +T2, +T3, +T4, +T5, +T6] private (
    fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4], Fetchable[T5], Fetchable[T6]))
    extends Fetchable[(T1, T2, T3, T4, T5, T6)] {
  private[this] val (fetches, indices) = {
    Fetchable.uniquifyFetches(Seq(fetchables._1, fetchables._2, fetchables._3, fetchables._4, fetchables._5, fetchables._6))
  }

  override def uniqueFetches: Seq[Op.Output] = fetches

  override def buildResult(values: Seq[Tensor]): (T1, T2, T3, T4, T5, T6) = {
    if (fetches.length != values.length)
      throw new IllegalArgumentException(
        s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
    (fetchables._1.buildResult(indices(0).map(values(_))),
        fetchables._2.buildResult(indices(1).map(values(_))),
        fetchables._3.buildResult(indices(2).map(values(_))),
        fetchables._4.buildResult(indices(3).map(values(_))),
        fetchables._5.buildResult(indices(4).map(values(_))),
        fetchables._6.buildResult(indices(5).map(values(_))))
  }
}

private[client] object FetchableTuple6 {
  def apply[T1, T2, T3, T4, T5, T6](fetchables: (Fetchable[T1], Fetchable[T2], Fetchable[T3], Fetchable[T4], Fetchable[T5], Fetchable[T6])): FetchableTuple6[T1, T2, T3, T4, T5, T6] = {
    new FetchableTuple6(fetchables)
  }
}

//endregion Tuples
