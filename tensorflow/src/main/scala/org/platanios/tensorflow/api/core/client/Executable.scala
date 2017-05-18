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
