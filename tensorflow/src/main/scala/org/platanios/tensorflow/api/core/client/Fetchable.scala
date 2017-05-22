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
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike, mutable}
import scala.language.higherKinds
import scala.reflect.ClassTag

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
  *   - Products of other [[Fetchable]]s (e.g., tuples).
  *     - Note that with tuples, heterogeneous types are supported, due to the tuple type being a kind of heterogeneous
  *       collection.
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
  type ResultType

  def numberOfFetches(fetchable: T): Int
  def fetches(fetchable: T): Seq[Op.Output]
  def resultsBuilder(fetchable: T, values: Seq[Tensor]): ResultType = segment(fetchable, values)._1
  def segment(fetchable: T, values: Seq[Tensor]): (ResultType, Seq[Tensor])
}

object Fetchable {
  private[client] def process[F, R](fetchable: F)(implicit ev: Aux[F, R]): (Seq[Op.Output], Seq[Tensor] => R) = {
    val fetches = ev.fetches(fetchable)
    val (uniqueFetches, indices) = Fetchable.uniquifyFetches(fetches)
    val resultsBuilder = (values: Seq[Tensor]) => ev.resultsBuilder(fetchable, indices.map(values(_)))
    (uniqueFetches, resultsBuilder)
  }

  private[Fetchable] def uniquifyFetches(fetches: Seq[Op.Output]): (Seq[Op.Output], Seq[Int]) = {
    val uniqueFetches = mutable.ArrayBuffer.empty[Op.Output]
    val seenFetches = mutable.Map.empty[Op.Output, Int]
    val indices = fetches.map(f => seenFetches.getOrElseUpdate(f, {uniqueFetches += f; uniqueFetches.length - 1}))
    (uniqueFetches, indices)
  }

  type Aux[T, R] = Fetchable[T] {type ResultType = R}

  implicit val opOutputFetchable: Aux[Op.Output, Tensor] = new Fetchable[Op.Output] {
    override type ResultType = Tensor
    override def numberOfFetches(fetchable: Op.Output): Int = 1
    override def fetches(fetchable: Op.Output): Seq[Op.Output] = Seq(fetchable)
    override def segment(fetchable: Op.Output, values: Seq[Tensor]): (Tensor, Seq[Tensor]) = (values.head, values.tail)
  }

  implicit def fetchableSeq[T, R, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[T, R]): Aux[CC[T], Seq[R]] = {
    new Fetchable[CC[T]] {
      // TODO: [CLIENT] Return CC type instead of Seq.
      override type ResultType = Seq[R]
      override def numberOfFetches(fetchable: CC[T]): Int = fetchable.map(ev.numberOfFetches).sum
      override def fetches(fetchable: CC[T]): Seq[Op.Output] = fetchable.flatMap(ev.fetches).toSeq
      override def segment(fetchable: CC[T], values: Seq[Tensor]): (Seq[R], Seq[Tensor]) = {
        val n = numberOfFetches(fetchable)
        (fetchable.zip(Collections.segment(values.take(n), fetchable.map(ev.numberOfFetches).toSeq))
            .map(f => ev.resultsBuilder(f._1, f._2)).toSeq, values.drop(n))
      }
    }
  }

  implicit def fetchableArray[T, R: ClassTag](implicit ev: Aux[T, R]): Aux[Array[T], Array[R]] = {
    new Fetchable[Array[T]] {
      override type ResultType = Array[R]
      override def numberOfFetches(fetchable: Array[T]): Int = fetchable.map(ev.numberOfFetches).sum
      override def fetches(fetchable: Array[T]): Seq[Op.Output] = fetchable.flatMap(ev.fetches).toSeq
      override def segment(fetchable: Array[T], values: Seq[Tensor]): (Array[R], Seq[Tensor]) = {
        val n = numberOfFetches(fetchable)
        (fetchable.zip(Collections.segment(values.take(n), fetchable.map(ev.numberOfFetches).toSeq))
            .map(f => ev.resultsBuilder(f._1, f._2)).toArray, values.drop(n))
      }
    }
  }

  implicit def fetchableMap[T, R, CC[K, V] <: MapLike[K, V, CC[K, V]] with Map[K, V]]
  (implicit ev: Aux[T, R]): Aux[CC[String, T], Map[String, R]] = {
    new Fetchable[CC[String, T]] {
      // TODO: [CLIENT] Return CC type instead of Map.
      // TODO: [CLIENT] Make sure key-value pairs order is handled correctly here.
      override type ResultType = Map[String, R]
      override def numberOfFetches(fetchable: CC[String, T]): Int = fetchable.values.map(ev.numberOfFetches).sum
      override def fetches(fetchable: CC[String, T]): Seq[Op.Output] = fetchable.values.flatMap(ev.fetches).toSeq
      override def segment(fetchable: CC[String, T], values: Seq[Tensor]): (Map[String, R], Seq[Tensor]) = {
        val n = numberOfFetches(fetchable)
        (fetchable.keys.zip(
          fetchable.values
              .zip(Collections.segment(values.take(n), fetchable.values.map(ev.numberOfFetches).toSeq))
              .map(f => ev.resultsBuilder(f._1, f._2))).toMap, values.drop(n))
      }
    }
  }

  implicit val hnil: Aux[HNil, HNil] = new Fetchable[HNil] {
    override type ResultType = HNil
    override def numberOfFetches(fetchable: HNil): Int = 0
    override def fetches(fetchable: HNil): Seq[Op.Output] = Seq.empty
    override def segment(fetchable: HNil, values: Seq[Tensor]): (HNil, Seq[Tensor]) = (HNil, values)
  }

  implicit def recursiveConstructor[H, R, T <: HList, TO <: HList](implicit
      fetchableHead: Aux[H, R],
      fetchableTail: Aux[T, TO]
  ): Aux[H :: T, R :: TO] = new Fetchable[H :: T] {
    override type ResultType = R :: TO

    override def numberOfFetches(fetchable: H :: T): Int = {
      fetchableHead.numberOfFetches(fetchable.head) + fetchableTail.numberOfFetches(fetchable.tail)
    }

    override def fetches(fetchable: H :: T): Seq[Op.Output] = {
      fetchableHead.fetches(fetchable.head) ++ fetchableTail.fetches(fetchable.tail)
    }

    override def segment(fetchable: H :: T, tensors: Seq[Tensor]): (R :: TO, Seq[Tensor]) = {
      val (headOut, headRemaining) = fetchableHead.segment(fetchable.head, tensors)
      val (tailOut, tailRemaining) = fetchableTail.segment(fetchable.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }
  }

  // This also covers `Op.OutputIndexedSlices` and `Op.SparseOutput` as they are case classes (i.e., products).
  implicit def productConstructor[P <: Product, L <: HList, LO <: HList, R](implicit
      gen: Generic.Aux[P, L],
      fetchableL: Aux[L, LO],
      tupler: Tupler.Aux[LO, R]
  ): Aux[P, R] = new Fetchable[P] {
    override type ResultType = R
    override def numberOfFetches(fetchable: P): Int = fetchableL.numberOfFetches(gen.to(fetchable))
    override def fetches(fetchable: P): Seq[Op.Output] = fetchableL.fetches(gen.to(fetchable))
    override def segment(p: P, tensors: Seq[Tensor]): (R, Seq[Tensor]) = {
      val (out, remaining) = fetchableL.segment(gen.to(p), tensors)
      (tupler(out), remaining)
    }
  }
}
