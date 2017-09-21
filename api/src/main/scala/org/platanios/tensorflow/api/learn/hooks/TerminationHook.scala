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
import org.platanios.tensorflow.api.learn.{Coordinator, Counter, TerminationCriteria}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

// TODO: !!! [HOOKS] [TERMINATION] Currently not using the convergence criteria.

/** Hook that requests to stop iterating when certain termination criteria are satisfied.
  *
  * @param  terminationCriteria Termination criteria to use.
  *
  * @author Emmanouil Antonios Platanios
  */
private[learn] case class TerminationHook private[learn] (terminationCriteria: TerminationCriteria) extends Hook {
  private[this] var epoch    : Variable = _
  private[this] var iteration: Variable = _

  private[this] var lastEpoch: Option[Long] = {
    if (terminationCriteria.restartCounting)
      None
    else
      terminationCriteria.maxEpochs
  }

  private[this] var lastIteration: Option[Long] = {
    if (terminationCriteria.restartCounting)
      None
    else
      terminationCriteria.maxIterations
  }

  override def begin(): Unit = {
    epoch = Counter.get(Graph.Keys.GLOBAL_EPOCH, Op.currentGraph).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_EPOCH.name} variable should be created in order to use the 'StopAtStepHook'."))
    iteration = Counter.get(Graph.Keys.GLOBAL_ITERATION, Op.currentGraph).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_ITERATION.name} variable should be created in order to use the 'StopAtStepHook'."))
  }

  override def afterSessionCreation(session: Session, coordinator: Coordinator): Unit = {
    if (terminationCriteria.restartCounting) {
      val (e, i) = session.run(fetches = (epoch.value, iteration.value))
      lastEpoch = terminationCriteria.maxEpochs.map(_ + e.scalar.asInstanceOf[Long])
      lastIteration = terminationCriteria.maxIterations.map(_ + i.scalar.asInstanceOf[Long])
    }
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    Some(Hook.SessionRunArgs(fetches = Seq(epoch.value, iteration.value)))
  }

  @throws[IllegalStateException]
  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    val currentEpoch = runResult.values(0).scalar.asInstanceOf[Long]
    val currentIteration = runResult.values(1).scalar.asInstanceOf[Long]
    if (lastEpoch.exists(currentEpoch >= _) || lastIteration.exists(currentIteration >= _))
      runContext.requestStop()
  }
}

object TerminationHook {
  private[TerminationHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Termination"))
}
