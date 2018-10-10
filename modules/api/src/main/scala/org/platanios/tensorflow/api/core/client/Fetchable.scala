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
import org.platanios.tensorflow.api.tensors.{Tensor, TensorIndexedSlices, SparseTensor}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.breakOut
import scala.collection.mutable
import scala.language.higherKinds
import scala.reflect.ClassTag

/** Fetchables can be executed within a TensorFlow session and have their results returned from [[Session.run]].
  *
  * For example, the result of any mathematical operation can be fetched.
  *
  * Currently supported fetchable types are:
  *   - Single [[Output]], [[OutputIndexedSlices]], [[SparseOutput]] object.
  *   - Sequences of other [[Fetchable]]s (e.g., `Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(output1, Seq(output1, output2))`).
  *     - Note that, for that reason, even though `Seq(List(output1), List(output1, output2))` is supported,
  *       `Seq(Seq(output1), List(output1, output2))` is not.
  *     - A sequence containing both [[Output]]s and [[SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[Fetchable]]s.
  *   - Maps with arbitrary key types and [[Fetchable]] value types.
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
  def fetches(fetchable: T): Seq[Output[Any]]
  def resultsBuilder(fetchable: T, values: Seq[Tensor[Any]]): ResultType = segment(fetchable, values)._1
  def segment(fetchable: T, values: Seq[Tensor[Any]]): (ResultType, Seq[Tensor[Any]])
}

object Fetchable {
  private[client] def process[F, R](
      fetchable: F
  )(implicit ev: Aux[F, R]): (Seq[Output[Any]], Seq[Tensor[Any]] => R) = {
    val fetches = ev.fetches(fetchable)
    val (uniqueFetches, indices) = Fetchable.uniquifyFetches(fetches)
    val resultsBuilder = (values: Seq[Tensor[Any]]) => {
      ev.resultsBuilder(fetchable, indices.map(values(_)))
    }
    (uniqueFetches, resultsBuilder)
  }

  private[Fetchable] def uniquifyFetches(
      fetches: Seq[Output[Any]]
  ): (Seq[Output[Any]], Seq[Int]) = {
    val uniqueFetches = mutable.ArrayBuffer.empty[Output[Any]]
    val seenFetches = mutable.Map.empty[Output[_], Int]
    val indices = fetches.map(f => seenFetches.getOrElseUpdate(f, {
      uniqueFetches += f
      uniqueFetches.length - 1
    }))
    (uniqueFetches, indices)
  }

  type Aux[T, R] = Fetchable[T] {
    type ResultType = R
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new Fetchable[Unit] {
      override type ResultType = Unit

      override def numberOfFetches(fetchable: Unit): Int = {
        0
      }

      override def fetches(fetchable: Unit): Seq[Output[Any]] = {
        Seq.empty
      }

      override def segment(
          fetchable: Unit,
          values: Seq[Tensor[Any]]
      ): (Unit, Seq[Tensor[Any]]) = {
        ((), values)
      }
    }
  }

  implicit def fromOutput[T]: Aux[Output[T], Tensor[T]] = {
    new Fetchable[Output[T]] {
      override type ResultType = Tensor[T]

      override def numberOfFetches(fetchable: Output[T]): Int = {
        1
      }

      override def fetches(fetchable: Output[T]): Seq[Output[Any]] = {
        Seq(fetchable)
      }

      override def segment(
          fetchable: Output[T],
          values: Seq[Tensor[Any]]
      ): (Tensor[T], Seq[Tensor[Any]]) = {
        (values.head.asInstanceOf[Tensor[T]], values.tail)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: Aux[OutputIndexedSlices[T], TensorIndexedSlices[T]] = {
    new Fetchable[OutputIndexedSlices[T]] {
      override type ResultType = TensorIndexedSlices[T]

      override def numberOfFetches(fetchable: OutputIndexedSlices[T]): Int = {
        3
      }

      override def fetches(fetchable: OutputIndexedSlices[T]): Seq[Output[Any]] = {
        Seq(fetchable.indices, fetchable.values, fetchable.denseShape)
      }

      override def segment(
          fetchable: OutputIndexedSlices[T],
          values: Seq[Tensor[Any]]
      ): (TensorIndexedSlices[T], Seq[Tensor[Any]]) = {
        (TensorIndexedSlices(
          indices = values(0).asInstanceOf[Tensor[Long]],
          values = values(1).asInstanceOf[Tensor[T]],
          denseShape = values(2).asInstanceOf[Tensor[Long]]),
            values.drop(3))
      }
    }
  }

  implicit def fromSparseOutput[T]: Aux[SparseOutput[T], SparseTensor[T]] = {
    new Fetchable[SparseOutput[T]] {
      override type ResultType = SparseTensor[T]

      override def numberOfFetches(fetchable: SparseOutput[T]): Int = {
        3
      }

      override def fetches(fetchable: SparseOutput[T]): Seq[Output[Any]] = {
        Seq(fetchable.indices, fetchable.values, fetchable.denseShape)
      }

      override def segment(
          fetchable: SparseOutput[T],
          values: Seq[Tensor[Any]]
      ): (SparseTensor[T], Seq[Tensor[Any]]) = {
        (SparseTensor(
          indices = values(0).asInstanceOf[Tensor[Long]],
          values = values(1).asInstanceOf[Tensor[T]],
          denseShape = values(2).asInstanceOf[Tensor[Long]]),
            values.drop(3))
      }
    }
  }

  implicit def fromOption[T, R](implicit ev: Aux[T, R]): Aux[Option[T], Option[R]] = {
    new Fetchable[Option[T]] {
      override type ResultType = Option[R]

      override def numberOfFetches(fetchable: Option[T]): Int = {
        fetchable.map(ev.numberOfFetches).getOrElse(0)
      }

      override def fetches(fetchable: Option[T]): Seq[Output[Any]] = {
        fetchable.map(ev.fetches).getOrElse(Seq.empty)
      }

      override def segment(
          fetchable: Option[T],
          values: Seq[Tensor[Any]]
      ): (Option[R], Seq[Tensor[Any]]) = {
        fetchable match {
          case Some(f) =>
            val (result, remaining) = ev.segment(f, values)
            (Some(result), remaining)
          case None => (None, values)
        }
      }
    }
  }

  implicit def fromArray[T, R: ClassTag](implicit ev: Aux[T, R]): Aux[Array[T], Array[R]] = {
    new Fetchable[Array[T]] {
      override type ResultType = Array[R]

      override def numberOfFetches(fetchable: Array[T]): Int = {
        fetchable.map(ev.numberOfFetches).sum
      }

      override def fetches(fetchable: Array[T]): Seq[Output[Any]] = {
        fetchable.flatMap(ev.fetches).toSeq
      }

      override def segment(
          fetchable: Array[T],
          values: Seq[Tensor[Any]]
      ): (Array[R], Seq[Tensor[Any]]) = {
        val n = numberOfFetches(fetchable)
        val segmented = Collections.segment(
          values.take(n), fetchable.map(ev.numberOfFetches).toSeq)
        (fetchable.zip(segmented)
            .map(f => ev.resultsBuilder(f._1, f._2)), values.drop(n))
      }
    }
  }

  implicit def fromSeq[T, R](implicit ev: Aux[T, R]): Aux[Seq[T], Seq[R]] = {
    new Fetchable[Seq[T]] {
      override type ResultType = Seq[R]

      override def numberOfFetches(fetchable: Seq[T]): Int = {
        fetchable.map(ev.numberOfFetches).sum
      }

      override def fetches(fetchable: Seq[T]): Seq[Output[Any]] = {
        fetchable.flatMap(ev.fetches).toSeq
      }

      override def segment(
          fetchable: Seq[T],
          values: Seq[Tensor[Any]]
      ): (Seq[R], Seq[Tensor[Any]]) = {
        val n = numberOfFetches(fetchable)
        val segmented = Collections.segment(
          values.take(n), fetchable.map(ev.numberOfFetches).toSeq)
        (fetchable.zip(segmented)(breakOut)
            .map(f => ev.resultsBuilder(f._1, f._2)), values.drop(n))
      }
    }
  }

  implicit def fromMap[T, R, MK](implicit ev: Aux[T, R]): Aux[Map[MK, T], Map[MK, R]] = {
    new Fetchable[Map[MK, T]] {
      // TODO: [CLIENT] Make sure key-value pairs order is handled correctly here.
      override type ResultType = Map[MK, R]

      override def numberOfFetches(fetchable: Map[MK, T]): Int = {
        fetchable.values.map(ev.numberOfFetches).sum
      }

      override def fetches(fetchable: Map[MK, T]): Seq[Output[Any]] = {
        fetchable.values.flatMap(ev.fetches).toSeq
      }

      override def segment(
          fetchable: Map[MK, T],
          values: Seq[Tensor[Any]]
      ): (Map[MK, R], Seq[Tensor[Any]]) = {
        val n = numberOfFetches(fetchable)
        val segmented = Collections.segment(
          values.take(n), fetchable.values.map(ev.numberOfFetches).toSeq)
        (fetchable.keys.zip(
          fetchable.values.zip(segmented)
              .map(f => ev.resultsBuilder(f._1, f._2))).toMap, values.drop(n))
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil] = {
    new Fetchable[HNil] {
      override type ResultType = HNil

      override def numberOfFetches(fetchable: HNil): Int = {
        0
      }

      override def fetches(fetchable: HNil): Seq[Output[Any]] = {
        Seq.empty
      }

      override def segment(
          fetchable: HNil,
          values: Seq[Tensor[Any]]
      ): (HNil, Seq[Tensor[Any]]) = {
        (HNil, values)
      }
    }
  }

  implicit def fromHList[H, HO, T <: HList, TO <: HList](implicit
      evH: Strict[Aux[H, HO]],
      evT: Aux[T, TO]
  ): Aux[H :: T, HO :: TO] = {
    new Fetchable[H :: T] {
      override type ResultType = HO :: TO

      override def numberOfFetches(fetchable: H :: T): Int = {
        evH.value.numberOfFetches(fetchable.head) +
            evT.numberOfFetches(fetchable.tail)
      }

      override def fetches(fetchable: H :: T): Seq[Output[Any]] = {
        evH.value.fetches(fetchable.head) ++
            evT.fetches(fetchable.tail)
      }

      override def segment(
          fetchable: H :: T,
          tensors: Seq[Tensor[Any]]
      ): (HO :: TO, Seq[Tensor[Any]]) = {
        val (headOut, headRemaining) = evH.value.segment(fetchable.head, tensors)
        val (tailOut, tailRemaining) = evT.segment(fetchable.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromCoproduct[H, HO, T <: Coproduct, TO <: Coproduct](implicit
      evH: Strict[Aux[H, HO]],
      evT: Aux[T, TO]
  ): Aux[H :+: T, HO :+: TO] = {
    new Fetchable[H :+: T] {
      override type ResultType = HO :+: TO

      override def numberOfFetches(fetchable: H :+: T): Int = {
        fetchable match {
          case Inl(h) => evH.value.numberOfFetches(h)
          case Inr(t) => evT.numberOfFetches(t)
        }
      }

      override def fetches(fetchable: H :+: T): Seq[Output[Any]] = {
        fetchable match {
          case Inl(h) => evH.value.fetches(h)
          case Inr(t) => evT.fetches(t)
        }
      }

      override def segment(fetchable: H :+: T, values: Seq[Tensor[Any]]): (HO :+: TO, Seq[Tensor[Any]]) = {
        fetchable match {
          case Inl(h) =>
            val (result, remaining) = evH.value.segment(h, values)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.segment(t, values)
            (Inr(result), remaining)
        }
      }
    }
  }

  implicit def fromProduct[P <: Product, R <: Product, L <: HList, LO <: HList](implicit
      gen: Generic.Aux[P, L],
      evL: Strict[Aux[L, LO]],
      tupler: Tupler.Aux[LO, R]
  ): Aux[P, R] = {
    new Fetchable[P] {
      override type ResultType = R

      override def numberOfFetches(fetchable: P): Int = {
        evL.value.numberOfFetches(gen.to(fetchable))
      }

      override def fetches(fetchable: P): Seq[Output[Any]] = {
        evL.value.fetches(gen.to(fetchable))
      }

      override def segment(
          p: P,
          tensors: Seq[Tensor[Any]]
      ): (R, Seq[Tensor[Any]]) = {
        val (out, remaining) = evL.value.segment(gen.to(p), tensors)
        (tupler(out), remaining)
      }
    }
  }
}
