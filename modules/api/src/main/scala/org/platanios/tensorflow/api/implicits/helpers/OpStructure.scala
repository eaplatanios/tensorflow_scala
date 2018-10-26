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
sealed trait OpStructure[T] {
  /** Target ops to execute. */
  def ops(executable: T): Set[UntypedOp]
}

object OpStructure extends NestedStructureOpsLowPriority {
  def apply[T](implicit ev: OpStructure[T]): OpStructure[T] = {
    ev
  }
}

trait NestedStructureOpsLowPriority {
  implicit val fromUnit: OpStructure[Unit] = {
    new OpStructure[Unit] {
      /** Target ops to execute. */
      override def ops(executable: Unit): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromOp[I, O]: OpStructure[Op[I, O]] = {
    new OpStructure[Op[I, O]] {
      override def ops(executable: Op[I, O]): Set[UntypedOp] = {
        Set(executable)
      }
    }
  }

  implicit def fromOutput[T]: OpStructure[Output[T]] = {
    new OpStructure[Output[T]] {
      override def ops(executable: Output[T]): Set[UntypedOp] = {
        Set(executable.op)
      }
    }
  }

  implicit def fromOption[T: OpStructure]: OpStructure[Option[T]] = {
    new OpStructure[Option[T]] {
      override def ops(executable: Option[T]): Set[UntypedOp] = {
        executable.map(e => OpStructure[T].ops(e)).getOrElse(Set.empty)
      }
    }
  }

  implicit def fromArray[T: OpStructure]: OpStructure[Array[T]] = {
    new OpStructure[Array[T]] {
      override def ops(executable: Array[T]): Set[UntypedOp] = {
        executable.flatMap(e => OpStructure[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSeq[T: OpStructure]: OpStructure[Seq[T]] = {
    new OpStructure[Seq[T]] {
      override def ops(executable: Seq[T]): Set[UntypedOp] = {
        executable.flatMap(e => OpStructure[T].ops(e)).toSet
      }
    }
  }

  implicit def fromSet[T: OpStructure]: OpStructure[Set[T]] = {
    new OpStructure[Set[T]] {
      override def ops(executable: Set[T]): Set[UntypedOp] = {
        executable.flatMap(e => OpStructure[T].ops(e))
      }
    }
  }

  implicit val fromHNil: OpStructure[HNil] = {
    new OpStructure[HNil] {
      override def ops(executable: HNil): Set[UntypedOp] = {
        Set.empty
      }
    }
  }

  implicit def fromHList[H, T <: HList](implicit
      evH: Strict[OpStructure[H]],
      evT: OpStructure[T]
  ): OpStructure[H :: T] = {
    new OpStructure[H :: T] {
      override def ops(executable: H :: T): Set[UntypedOp] = {
        evH.value.ops(executable.head) ++
            evT.ops(executable.tail)
      }
    }
  }

  implicit def fromProduct[P <: Product, L <: HList](implicit
      gen: Generic.Aux[P, L],
      executableL: Strict[OpStructure[L]]
  ): OpStructure[P] = {
    new OpStructure[P] {
      override def ops(executable: P): Set[UntypedOp] = {
        executableL.value.ops(gen.to(executable))
      }
    }
  }
}
