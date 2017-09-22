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
  private[this] var epoch: Variable = _
  private[this] var step : Variable = _

  private[this] var lastEpoch: Option[Long] = {
    if (terminationCriteria.restartCounting)
      None
    else
      terminationCriteria.maxEpochs
  }

  private[this] var lastStep: Option[Long] = {
    if (terminationCriteria.restartCounting)
      None
    else
      terminationCriteria.maxSteps
  }

  override def begin(): Unit = {
    if (terminationCriteria.maxEpochs.isDefined)
      epoch = Counter.get(Graph.Keys.GLOBAL_EPOCH, Op.currentGraph).getOrElse(throw new IllegalStateException(
        s"A ${Graph.Keys.GLOBAL_EPOCH.name} variable should be created in order to use the 'StopAtStepHook'."))
    if (terminationCriteria.maxSteps.isDefined)
      step = Counter.get(Graph.Keys.GLOBAL_STEP, Op.currentGraph).getOrElse(throw new IllegalStateException(
        s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'StopAtStepHook'."))
  }

  override def afterSessionCreation(session: Session, coordinator: Coordinator): Unit = {
    if (terminationCriteria.restartCounting &&
        (terminationCriteria.maxEpochs.isDefined || terminationCriteria.maxSteps.isDefined)) {
      (terminationCriteria.maxEpochs, terminationCriteria.maxSteps) match {
        case (Some(_), Some(_)) =>
          val (e, s) = session.run(fetches = (epoch.value, step.value))
          lastEpoch = terminationCriteria.maxEpochs.map(_ + e.scalar.asInstanceOf[Long])
          lastStep = terminationCriteria.maxSteps.map(_ + s.scalar.asInstanceOf[Long])
        case (Some(_), None) =>
          val e = session.run(fetches = epoch.value)
          lastEpoch = terminationCriteria.maxEpochs.map(_ + e.scalar.asInstanceOf[Long])
        case (None, Some(_)) =>
          val s = session.run(fetches = step.value)
          lastStep = terminationCriteria.maxSteps.map(_ + s.scalar.asInstanceOf[Long])
        case (None, None) => () // Impossible branch.
      }
    }
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    val fetches = {
      if (terminationCriteria.maxEpochs.isDefined && terminationCriteria.maxSteps.isDefined)
        Seq(epoch.value, step.value)
      else if (terminationCriteria.maxEpochs.isDefined)
        Seq(epoch.value)
      else if (terminationCriteria.maxSteps.isDefined)
        Seq(step.value)
      else
        Seq.empty[Output]
    }
    Some(Hook.SessionRunArgs(fetches = fetches))
  }

  @throws[IllegalStateException]
  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    var converged = false
    if (terminationCriteria.maxEpochs.isDefined) {
      val currentEpoch = runResult.values(0).scalar.asInstanceOf[Long]
      converged ||= lastEpoch.exists(currentEpoch >= _)
    }
    if (terminationCriteria.maxEpochs.isDefined && terminationCriteria.maxSteps.isDefined) {
      val currentIteration = runResult.values(1).scalar.asInstanceOf[Long]
      converged ||= lastStep.exists(currentIteration >= _)
    } else if (terminationCriteria.maxSteps.isDefined) {
      val currentIteration = runResult.values(0).scalar.asInstanceOf[Long]
      converged ||= lastStep.exists(currentIteration >= _)
    }
    if (converged)
      runContext.requestStop()
  }
}

object TerminationHook {
  private[TerminationHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Termination"))
}
