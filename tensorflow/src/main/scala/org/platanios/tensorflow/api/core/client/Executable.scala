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
  *   - Sequences of other [[Executable]]s.
  * Internally, the executable provided to a session will be de-duplicated to prevent redundant computation. This means
  * that ops that appear more than once in the executable, will only be executed once by the session.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Executable[T] {
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
}
