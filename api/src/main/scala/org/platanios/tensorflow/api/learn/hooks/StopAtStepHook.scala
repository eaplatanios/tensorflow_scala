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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.core.client.{Executable, Fetchable, Session}
import org.platanios.tensorflow.api.learn.{Coordinator, GlobalStep}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Hook that requests to stop iterating at a specified step.
  *
  * This hook requests to stop iterating after either a number of steps have been executed (when `restartCounting` is
  * set to `true`) or a global step value has been reached (when `restartCounting` is set to `false`).
  *
  * @param  numSteps        Number of steps after which to stop iterating.
  * @param  restartCounting If `true`, the number of steps is counted starting at global step value when initializing
  *                         this hook. Otherwise, the iteration stops when the global step exceeds `numSteps` in value.
  *                         For example, in that case, if global step is 10 when the hook is initialized and `numSteps`
  *                         is `100`, the iteration will repeat `90` times. If `restartCounting` was set to `true`, in
  *                         that case, it would repeat `100` times.
  *
  * @author Emmanouil Antonios Platanios
  */
case class StopAtStepHook(numSteps: Int, restartCounting: Boolean) extends Hook {
  private[this] var globalStep: Variable    = _
  private[this] var lastStep  : Option[Int] = if (restartCounting) None else Some(numSteps)

  override def begin(): Unit = {
    globalStep = GlobalStep.get(Op.currentGraph).getOrElse(throw new IllegalStateException(
      "A global step variable should be created in order to use the 'StopAtStepHook'."))
  }

  override def afterSessionCreation(session: Session, coordinator: Coordinator): Unit = lastStep match {
    case Some(_) => ()
    case None => lastStep = Some(session.run(fetches = globalStep.value).scalar.asInstanceOf[Int] + numSteps)
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    Some(Hook.SessionRunArgs(fetches = Seq(globalStep.value)))
  }

  @throws[IllegalStateException]
  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    val currentStep = runResult.values.head.scalar.asInstanceOf[Int]
    if (currentStep >= lastStep.get)
      runContext.requestStop()
  }
}

object StopAtStepHook {
  private[StopAtStepHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Stop at Step"))
}
