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

import org.platanios.tensorflow.api.ops._

import shapeless._

import scala.collection.TraversableLike
import scala.language.higherKinds

/** Executables can be executed within a TensorFlow session, but their results (if any) are not returned.
  *
  * For example, variable initializers are executable.
  *
  * Currently supported executable types are:
  *   - Single [[Op]] object.
  *   - Single [[OutputLike]] object.
  *   - Traversables of other [[Executable]]s (e.g., `Set`s, `List`s, etc.).
  *     - Note that, even though `Set(List(op1), List(op1, op2))` is supported, `Set(Set(op1), List(op1, op2))` is not.
  *     - Traversables that are not homogeneous are not supported (e.g., `Set(op1, Set(op1, op2))`).
  *   - Arrays of other [[Executable]]s.
  *   - Products of other [[Executable]]s (e.g., tuples).
  *     - Note that with tuples, heterogeneous types are supported, due to the tuple type being a kind of heterogeneous
  *       collection.
  * Internally, the executable provided to a session will be de-duplicated to prevent redundant computation. This means
  * that ops that appear more than once in the executable, will only be executed once by the session.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Executable[T] {
  /** Target ops to execute. */
  def ops(executable: T): Set[UntypedOp]
}

object Executable {
  def apply[T: Executable]: Executable[T] = implicitly[Executable[T]]

  implicit val fromUnit: Executable[Unit] = {
    new Executable[Unit] {
      /** Target ops to execute. */
      override def ops(executable: Unit): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromOp[I, O]: Executable[Op[I, O]] = {
    new Executable[Op[I, O]] {
      override def ops(executable: Op[I, O]): Set[UntypedOp] = {
        Set(executable)
      }
    }
  }

  implicit def fromOutput[T]: Executable[Output[T]] = {
    new Executable[Output[T]] {
      override def ops(executable: Output[T]): Set[UntypedOp] = {
        Set(executable.op)
      }
    }
  }

  implicit def fromOption[T: Executable]: Executable[Option[T]] = {
    new Executable[Option[T]] {
      override def ops(executable: Option[T]): Set[UntypedOp] = {
        executable.map(e => Executable[T].ops(e)).getOrElse(Set.empty)
      }
    }
  }

  implicit def fromArray[T: Executable]: Executable[Array[T]] = {
    new Executable[Array[T]] {
      override def ops(executable: Array[T]): Set[UntypedOp] = {
        executable.flatMap(e => Executable[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSeq[T: Executable]: Executable[Seq[T]] = {
    new Executable[Seq[T]] {
      override def ops(executable: Seq[T]): Set[UntypedOp] = {
        executable.flatMap(e => Executable[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSet[T: Executable]: Executable[Set[T]] = {
    new Executable[Set[T]] {
      override def ops(executable: Set[T]): Set[UntypedOp] = {
        executable.flatMap(e => Executable[T].ops(e))
      }
    }
  }

  implicit val fromHNil: Executable[HNil] = {
    new Executable[HNil] {
      override def ops(executable: HNil): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromHList[H, T <: HList](implicit
      evH: Strict[Executable[H]],
      evT: Executable[T]
  ): Executable[H :: T] = {
    new Executable[H :: T] {
      override def ops(executable: H :: T): Set[UntypedOp] = {
        evH.value.ops(executable.head) ++
            evT.ops(executable.tail)
      }
    }
  }

  implicit def fromCoproduct[H, T <: Coproduct](implicit
      evH: Strict[Executable[H]],
      evT: Executable[T]
  ): Executable[H :+: T] = {
    new Executable[H :+: T] {
      override def ops(executable: H :+: T): Set[UntypedOp] = {
        executable match {
          case Inl(h) => evH.value.ops(h)
          case Inr(t) => evT.ops(t)
        }
      }
    }
  }

  implicit def fromProduct[P <: Product, L <: HList](implicit
      gen: Generic.Aux[P, L],
      executableL: Strict[Executable[L]]
  ): Executable[P] = {
    new Executable[P] {
      override def ops(executable: P): Set[UntypedOp] = {
        executableL.value.ops(gen.to(executable))
      }
    }
  }
}
