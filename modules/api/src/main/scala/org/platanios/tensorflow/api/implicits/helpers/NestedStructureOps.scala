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

package org.platanios.tensorflow.api.implicits.helpers

import org.platanios.tensorflow.api.ops._

import shapeless._

import scala.language.higherKinds

/** Executables can be executed within a TensorFlow session, but their results (if any) are not returned.
  *
  * For example, variable initializers are executable.
  *
  * Currently supported executable types are:
  *   - Single [[Op]] object.
  *   - Single [[OutputLike]] object.
  *   - Traversables of other [[NestedStructureOps]]s (e.g., `Set`s, `List`s, etc.).
  *     - Note that, even though `Set(List(op1), List(op1, op2))` is supported, `Set(Set(op1), List(op1, op2))` is not.
  *     - Traversables that are not homogeneous are not supported (e.g., `Set(op1, Set(op1, op2))`).
  *   - Arrays of other [[NestedStructureOps]]s.
  *   - Products of other [[NestedStructureOps]]s (e.g., tuples).
  *     - Note that with tuples, heterogeneous types are supported, due to the tuple type being a kind of heterogeneous
  *       collection.
  * Internally, the executable provided to a session will be de-duplicated to prevent redundant computation. This means
  * that ops that appear more than once in the executable, will only be executed once by the session.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait NestedStructureOps[T] {
  /** Target ops to execute. */
  def ops(executable: T): Set[UntypedOp]
}

object NestedStructureOps {
  def apply[T: NestedStructureOps]: NestedStructureOps[T] = implicitly[NestedStructureOps[T]]

  implicit val fromUnit: NestedStructureOps[Unit] = {
    new NestedStructureOps[Unit] {
      /** Target ops to execute. */
      override def ops(executable: Unit): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromOp[I, O]: NestedStructureOps[Op[I, O]] = {
    new NestedStructureOps[Op[I, O]] {
      override def ops(executable: Op[I, O]): Set[UntypedOp] = {
        Set(executable)
      }
    }
  }

  implicit def fromOutput[T]: NestedStructureOps[Output[T]] = {
    new NestedStructureOps[Output[T]] {
      override def ops(executable: Output[T]): Set[UntypedOp] = {
        Set(executable.op)
      }
    }
  }

  implicit def fromOption[T: NestedStructureOps]: NestedStructureOps[Option[T]] = {
    new NestedStructureOps[Option[T]] {
      override def ops(executable: Option[T]): Set[UntypedOp] = {
        executable.map(e => NestedStructureOps[T].ops(e)).getOrElse(Set.empty)
      }
    }
  }

  implicit def fromArray[T: NestedStructureOps]: NestedStructureOps[Array[T]] = {
    new NestedStructureOps[Array[T]] {
      override def ops(executable: Array[T]): Set[UntypedOp] = {
        executable.flatMap(e => NestedStructureOps[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSeq[T: NestedStructureOps]: NestedStructureOps[Seq[T]] = {
    new NestedStructureOps[Seq[T]] {
      override def ops(executable: Seq[T]): Set[UntypedOp] = {
        executable.flatMap(e => NestedStructureOps[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSet[T: NestedStructureOps]: NestedStructureOps[Set[T]] = {
    new NestedStructureOps[Set[T]] {
      override def ops(executable: Set[T]): Set[UntypedOp] = {
        executable.flatMap(e => NestedStructureOps[T].ops(e))
      }
    }
  }

  implicit val fromHNil: NestedStructureOps[HNil] = {
    new NestedStructureOps[HNil] {
      override def ops(executable: HNil): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromHList[H, T <: HList](implicit
      evH: Strict[NestedStructureOps[H]],
      evT: NestedStructureOps[T]
  ): NestedStructureOps[H :: T] = {
    new NestedStructureOps[H :: T] {
      override def ops(executable: H :: T): Set[UntypedOp] = {
        evH.value.ops(executable.head) ++
            evT.ops(executable.tail)
      }
    }
  }

  implicit def fromCoproduct[H, T <: Coproduct](implicit
      evH: Strict[NestedStructureOps[H]],
      evT: NestedStructureOps[T]
  ): NestedStructureOps[H :+: T] = {
    new NestedStructureOps[H :+: T] {
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
      executableL: Strict[NestedStructureOps[L]]
  ): NestedStructureOps[P] = {
    new NestedStructureOps[P] {
      override def ops(executable: P): Set[UntypedOp] = {
        executableL.value.ops(gen.to(executable))
      }
    }
  }
}
