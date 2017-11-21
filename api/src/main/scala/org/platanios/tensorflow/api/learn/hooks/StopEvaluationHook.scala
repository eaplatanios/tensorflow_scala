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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.{Executable, Fetchable, Session}
import org.platanios.tensorflow.api.learn.{Counter, SessionCreator}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Hook that requests to stop evaluating after a certain number of steps have been executed.
  *
  * @param  maxSteps Maximum number of steps to execute (i.e., maximum number of batches to process).
  *
  * @author Emmanouil Antonios Platanios
  */
private[learn] case class StopEvaluationHook(maxSteps: Long = -1L) extends Hook {
  private[this] var step    : Variable = _
  private[this] var lastStep: Long     = if (maxSteps == -1L) -1L else 0L

  override def begin(sessionCreator: SessionCreator): Unit = {
    step = Counter.get(Graph.Keys.EVAL_STEP, local = true, Op.currentGraph).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.EVAL_STEP.name} variable should be created in order to use the 'StopEvaluationHook'."))
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = {
      if (maxSteps != -1L)
        maxSteps + session.run(fetches = step.value).scalar.asInstanceOf[Long]
      else
        maxSteps
    }
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  @throws[IllegalStateException]
  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    val step = runResult.values(0).scalar.asInstanceOf[Long]
    if (lastStep != -1 && step >= lastStep) {
      StopEvaluationHook.logger.info("Evaluation stop requested: Exceeded maximum number of steps.")
      runContext.requestStop()
    }
  }
}

object StopEvaluationHook {
  private[StopEvaluationHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Evaluation Termination"))
}
