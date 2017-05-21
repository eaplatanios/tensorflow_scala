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

import scala.collection.{MapLike, SeqLike, mutable}
import scala.language.higherKinds
import scala.reflect.ClassTag

// TODO: [CLIENT] !!! Add support for tuples.

/** Fetchables can be executed within a TensorFlow session and have their results returned from [[Session.run]].
  *
  * For example, the result of any mathematical operation can be fetched.
  *
  * Currently supported executable types are:
  *   - Single [[Op.Output]], [[Op.OutputIndexedSlices]], [[Op.SparseOutput]] object.
  *   - Sequences of other [[Fetchable]]s (e.g., `Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(output1, Seq(output1, output2))`).
  *     - Note that, for that reason, even though `Seq(List(output1), List(output1, output2))` is supported,
  *       `Seq(Seq(output1), List(output1, output2))` is not.
  *     - A sequence containing both [[Op.Output]]s and [[Op.SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[Fetchable]]s.
  * Internally, the fetchable provided to a session will be de-duplicated to prevent redundant computation. This means
  * that ops that appear more than once in the fetchable, will only be executed once by the session.
  *
  * Fetchables guarantee that the returned result of a computation will match the structure of the provided fetchable.
  * For example, if a `Seq(List(output1), List(output1, output2))` is provided as the [[Session.run]] fetchable, then
  * the result will have the following structure `Seq(List(tensor1), List(tensor1, tensor2))`.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Fetchable[T] {
  type R
  def process(fetchable: T): (Seq[Op.Output], Seq[Tensor] => R)
}

object Fetchable {
  type Aux[T, Res] = Fetchable[T] {type R = Res}

  // def apply[T](implicit instance: Fetchable[T]): Aux[T, instance.R] = instance

  def instance[T, Res](processor: T => (Seq[Op.Output], Seq[Tensor] => Res)): Aux[T, Res] = {
    new Fetchable[T] {
      override type R = Res
      override def process(fetchable: T): (Seq[Op.Output], Seq[Tensor] => R) = processor(fetchable)
    }
  }

  implicit val opOutputFetchable: Aux[Op.Output, Tensor] = instance(f => (Seq(f), v => v.head))

  implicit val opOutputIndexedSlicesFetchable: Aux[Op.OutputIndexedSlices, (Tensor, Tensor, Tensor)] = {
    instance(f => (Seq(f.indices, f.values, f.denseShape), v => (v(0), v(1), v(2))))
  }

  implicit val opSparseOutputFetchable: Aux[Op.SparseOutput, (Tensor, Tensor, Tensor)] = {
    instance(f => (Seq(f.indices, f.values, f.denseShape), v => (v(0), v(1), v(2))))
  }

  implicit def fetchableSeq[T, R, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[T, R]): Aux[CC[T], Seq[R]] = {
    // TODO: [CLIENT] Return CC type instead of Seq.
    instance(f => {
      val (fetches, indices, resultsBuilders) = Fetchable.uniquifyFetches(f.toSeq)
      def resultsBuilder(values: Seq[Tensor]): Seq[R] = {
        if (fetches.length != values.length)
          throw new IllegalArgumentException(
            s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
        resultsBuilders.zip(indices).map(f => f._1(f._2.map(values(_))))
      }
      (fetches, resultsBuilder)
    })
  }

  implicit def fetchableArray[T, R: ClassTag](implicit ev: Aux[T, R]): Aux[Array[T], Array[R]] = {
    instance(f => {
      val (fetches, indices, resultsBuilders) = Fetchable.uniquifyFetches(f)
      def resultsBuilder(values: Seq[Tensor]): Array[R] = {
        if (fetches.length != values.length)
          throw new IllegalArgumentException(
            s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
        resultsBuilders.zip(indices).map(f => f._1(f._2.map(values(_)))).toArray
      }
      (fetches, resultsBuilder)
    })
  }

  implicit def fetchableMap[T, R, CC[K, V] <: MapLike[K, V, CC[K, V]] with Map[K, V]](
      implicit ev: Aux[T, R]): Aux[CC[String, T], Map[String, R]] = {
    // TODO: [CLIENT] Return CC type instead of Map.
    instance(f => {
      val ff = f.toSeq
      val (fetches, indices, resultsBuilders) = Fetchable.uniquifyFetches(ff.map(_._2))
      def resultsBuilder(values: Seq[Tensor]): Map[String, R] = {
        if (fetches.length != values.length)
          throw new IllegalArgumentException(
            s"The number of values (${values.length}) must match the number of unique fetches (${fetches.length}).")
        ff.map(_._1).zip(resultsBuilders).zip(indices).map(f => f._1._1 -> f._1._2(f._2.map(values(_)))).toMap
      }
      (fetches, resultsBuilder)
    })
  }

  /** Uniquifies fetches from a sequences of [[Fetchable]]s.
    *
    * This is a helper function used by `fetchableSeq` and `fetchableMap`. It gathers all the unique fetches from a
    * sequence of fetchables and builds a sequence containing all of them, but without duplicates (`fetches`).
    *
    * It also returns a nested sequence of integers (`indices`) indicating at which index in `uniqueFetches` the fetches
    * of the individual fetchables are located. I.e.,
    * {{{
    *   indices(fetchableIndex)(fetchableFetchIndex) = fetchesIndex
    * }}}
    *
    * @param  fetchables Sequence of fetchables.
    * @return Tuple containing a sequence containing the unique fetches and a nested sequence of integers containing the
    *         value indices.
    */
  private[client] def uniquifyFetches[T, R](fetchables: Seq[T])
      (implicit ev: Fetchable.Aux[T, R]): (Seq[Op.Output], Seq[Seq[Int]], Seq[Seq[Tensor] => R]) = {
    val fetches = mutable.ArrayBuffer.empty[Op.Output]
    val seenFetches = mutable.Map.empty[Op.Output, Int]
    val processed = fetchables.map(ev.process)
    val indices = processed.map(_._1.map(f => seenFetches.getOrElseUpdate(f, {fetches += f; fetches.length - 1})))
    (fetches, indices, processed.map(_._2))
  }
}
