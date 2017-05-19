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

/** Executables can be executed within a TensorFlow session, but their results (if any) are not returned.
  *
  * For example, variable initializers are executable.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Executable {
  /** Target ops to execute. */
  def ops: Set[Op]
}

object Executable {
  object Empty extends Executable {
    override def ops: Set[Op] = Set.empty
  }

  trait Implicits {
    implicit def executableSet(executables: Set[Executable]): Executable = new ExecutableSet(executables)
    implicit def executableSeq(executables: Seq[Executable]): Executable = new ExecutableSeq(executables)
  }

  object Implicits extends Implicits
}

private[client] class ExecutableSet private[client] (executables: Set[Executable]) extends Executable {
  override def ops: Set[Op] = executables.flatMap(_.ops)
}

private[client] class ExecutableSeq private[client] (executables: Seq[Executable]) extends Executable {
  override def ops: Set[Op] = executables.flatMap(_.ops).toSet
}
