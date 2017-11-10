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

import org.platanios.tensorflow.api.ops.{Op, Output, OutputLike}

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
  def ops(executable: T): Set[Op]
}

object Executable {
  def apply[T: Executable]: Executable[T] = implicitly[Executable[T]]

  implicit val opExecutable: Executable[Op] = new Executable[Op] {
    override def ops(executable: Op): Set[Op] = Set(executable)
  }

  implicit def outputLikeExecutable: Executable[OutputLike] = new Executable[OutputLike] {
    override def ops(executable: OutputLike): Set[Op] = Set(executable.op)
  }

  implicit def outputExecutable: Executable[Output] = new Executable[Output] {
    override def ops(executable: Output): Set[Op] = Set(executable.op)
  }

  implicit def arrayExecutable[T: Executable]: Executable[Array[T]] = new Executable[Array[T]] {
    override def ops(executable: Array[T]): Set[Op] = executable.flatMap(e => Executable[T].ops(e)).toSet
  }

  implicit def traversableExecutable[T: Executable, CC[A] <: TraversableLike[A, CC[A]]]: Executable[CC[T]] = {
    new Executable[CC[T]] {
      override def ops(executable: CC[T]): Set[Op] = executable.flatMap(e => Executable[T].ops(e)).toSet
    }
  }

  implicit val hnil: Executable[HNil] = new Executable[HNil] {
    override def ops(executable: HNil): Set[Op] = Set.empty[Op]
  }

  implicit def recursiveConstructor[H, T <: HList](implicit
      executableHead: Executable[H],
      executableTail: Executable[T]
  ): Executable[H :: T] = new Executable[H :: T] {
    override def ops(executable: H :: T): Set[Op] = {
      executableHead.ops(executable.head) ++ executableTail.ops(executable.tail)
    }
  }

  // This also covers `OutputIndexedSlices` and `SparseOutput` as they are case classes (i.e., products).
  implicit def productConstructor[P <: Product, L <: HList](implicit
      gen: Generic.Aux[P, L],
      executableL: Executable[L]
  ): Executable[P] = new Executable[P] {
    override def ops(executable: P): Set[Op] = executableL.ops(gen.to(executable))
  }
}
