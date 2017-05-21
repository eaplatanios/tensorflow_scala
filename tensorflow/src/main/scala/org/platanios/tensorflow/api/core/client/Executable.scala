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

import scala.collection.TraversableLike
import scala.language.higherKinds

/** Executables can be executed within a TensorFlow session, but their results (if any) are not returned.
  *
  * For example, variable initializers are executable.
  *
  * Currently supported executable types are:
  *   - Single [[Op]] object.
  *   - Single [[Op.OutputLike]] object.
  *   - Traversables of other [[Executable]]s (e.g., `Set`s, `List`s, etc.).
  *     - Note that, even though `Set(List(op1), List(op1, op2))` is supported, `Set(Set(op1), List(op1, op2))` is not.
  *     - Traversables that are not homogeneous are not supported (e.g., `Set(op1, Set(op1, op2))`).
  *   - Arrays of other [[Executable]]s.
  *   - Tuples of other [[Executable]]s.
  *     - Note that with tuples, heterogeneous types are supported, due to the tuple type being a kind of heterogeneous
  *       collection.
  * Internally, the executable provided to a session will be de-duplicated to prevent redundant computation. This means
  * that ops that appear more than once in the executable, will only be executed once by the session.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Executable[T] {
  /** Target ops to execute. */
  def ops(executable: T): Set[Op]
}

object Executable {
  implicit val opExecutable = new Executable[Op] {
    override def ops(executable: Op): Set[Op] = Set(executable)
  }

  implicit def opOutputLikeExecutable[O <: Op.OutputLike] = new Executable[O] {
    override def ops(executable: O): Set[Op] = Set(executable.op)
  }

  implicit def arrayExecutable[T: Executable] = new Executable[Array[T]] {
    override def ops(executable: Array[T]): Set[Op] = {
      executable.flatMap(e => implicitly[Executable[T]].ops(e)).toSet
    }
  }

  implicit def traversableExecutable[T: Executable, CC[A] <: TraversableLike[A, CC[A]]] = new Executable[CC[T]] {
    override def ops(executable: CC[T]): Set[Op] = {
      executable.flatMap(e => implicitly[Executable[T]].ops(e)).toSet
    }
  }

  implicit def tuple1Executable[T1: Executable] = new Executable[Tuple1[T1]] {
    override def ops(executable: Tuple1[T1]): Set[Op] = {
      implicitly[Executable[T1]].ops(executable._1)
    }
  }

  implicit def tuple2Executable[T1: Executable, T2: Executable] = new Executable[(T1, T2)] {
    override def ops(executable: (T1, T2)): Set[Op] = {
      implicitly[Executable[T1]].ops(executable._1) ++
          implicitly[Executable[T2]].ops(executable._2)
    }
  }

  implicit def tuple3Executable[T1: Executable, T2: Executable, T3: Executable] = new Executable[(T1, T2, T3)] {
    override def ops(executable: (T1, T2, T3)): Set[Op] = {
      implicitly[Executable[T1]].ops(executable._1) ++
          implicitly[Executable[T2]].ops(executable._2) ++
          implicitly[Executable[T3]].ops(executable._3)
    }
  }

  implicit def tuple4Executable[T1: Executable, T2: Executable, T3: Executable, T4: Executable] =
    new Executable[(T1, T2, T3, T4)] {
      override def ops(executable: (T1, T2, T3, T4)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4)
      }
    }

  implicit def tuple5Executable[T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable] =
    new Executable[(T1, T2, T3, T4, T5)] {
      override def ops(executable: (T1, T2, T3, T4, T5)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5)
      }
    }

  implicit def tuple6Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6)
      }
    }

  implicit def tuple7Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7)
      }
    }

  implicit def tuple8Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8)
      }
    }

  implicit def tuple9Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9)
      }
    }

  implicit def tuple10Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10)
      }
    }

  implicit def tuple11Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11)
      }
    }

  implicit def tuple12Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12)
      }
    }

  implicit def tuple13Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13)
      }
    }

  implicit def tuple14Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14)
      }
    }

  implicit def tuple15Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15)
      }
    }

  implicit def tuple16Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16)] {
      override def ops(executable: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16)
      }
    }

  implicit def tuple17Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17)
      }
    }

  implicit def tuple18Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable, T18: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17) ++
            implicitly[Executable[T18]].ops(executable._18)
      }
    }

  implicit def tuple19Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable, T18: Executable, T19: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17) ++
            implicitly[Executable[T18]].ops(executable._18) ++
            implicitly[Executable[T19]].ops(executable._19)
      }
    }

  implicit def tuple20Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable, T18: Executable, T19: Executable, T20: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17) ++
            implicitly[Executable[T18]].ops(executable._18) ++
            implicitly[Executable[T19]].ops(executable._19) ++
            implicitly[Executable[T20]].ops(executable._20)
      }
    }

  implicit def tuple21Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable, T18: Executable, T19: Executable, T20: Executable,
  T21: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17) ++
            implicitly[Executable[T18]].ops(executable._18) ++
            implicitly[Executable[T19]].ops(executable._19) ++
            implicitly[Executable[T20]].ops(executable._20) ++
            implicitly[Executable[T21]].ops(executable._21)
      }
    }

  implicit def tuple22Executable[
  T1: Executable, T2: Executable, T3: Executable, T4: Executable, T5: Executable, T6: Executable, T7: Executable,
  T8: Executable, T9: Executable, T10: Executable, T11: Executable, T12: Executable, T13: Executable, T14: Executable,
  T15: Executable, T16: Executable, T17: Executable, T18: Executable, T19: Executable, T20: Executable, T21: Executable,
  T22: Executable] =
    new Executable[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
        T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22)] {
      override def ops(executable: (
          T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
              T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22)): Set[Op] = {
        implicitly[Executable[T1]].ops(executable._1) ++
            implicitly[Executable[T2]].ops(executable._2) ++
            implicitly[Executable[T3]].ops(executable._3) ++
            implicitly[Executable[T4]].ops(executable._4) ++
            implicitly[Executable[T5]].ops(executable._5) ++
            implicitly[Executable[T6]].ops(executable._6) ++
            implicitly[Executable[T7]].ops(executable._7) ++
            implicitly[Executable[T8]].ops(executable._8) ++
            implicitly[Executable[T9]].ops(executable._9) ++
            implicitly[Executable[T10]].ops(executable._10) ++
            implicitly[Executable[T11]].ops(executable._11) ++
            implicitly[Executable[T12]].ops(executable._12) ++
            implicitly[Executable[T13]].ops(executable._13) ++
            implicitly[Executable[T14]].ops(executable._14) ++
            implicitly[Executable[T15]].ops(executable._15) ++
            implicitly[Executable[T16]].ops(executable._16) ++
            implicitly[Executable[T17]].ops(executable._17) ++
            implicitly[Executable[T18]].ops(executable._18) ++
            implicitly[Executable[T19]].ops(executable._19) ++
            implicitly[Executable[T20]].ops(executable._20) ++
            implicitly[Executable[T21]].ops(executable._21) ++
            implicitly[Executable[T22]].ops(executable._22)
      }
    }
}
