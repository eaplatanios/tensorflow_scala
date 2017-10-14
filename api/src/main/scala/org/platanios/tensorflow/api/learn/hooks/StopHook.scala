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
import org.platanios.tensorflow.api.learn.{Counter, StopCriteria}
import org.platanios.tensorflow.api.ops.{Math, Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/** Hook that requests to stop iterating when certain stopping criteria are satisfied.
  *
  * @param  criteria Termination criteria to use.
  *
  * @author Emmanouil Antonios Platanios
  */
private[learn] case class StopHook private[learn] (criteria: StopCriteria) extends Hook {
  private[this] var epoch: Variable = _
  private[this] var step : Variable = _
  private[this] var loss : Output   = _

  private[this] var lastEpoch       : Option[Long] = if (criteria.restartCounting) None else criteria.maxEpochs
  private[this] var lastStep        : Option[Long] = if (criteria.restartCounting) None else criteria.maxSteps
  private[this] var lastLoss        : Float        = Float.MaxValue
  private[this] var numStepsBelowTol: Int          = 0

  private[this] var sessionFetches : Seq[Output] = _
  private[this] var epochFetchIndex: Int         = _
  private[this] var stepFetchIndex : Int         = _
  private[this] var lossFetchIndex : Int         = _

  override def begin(): Unit = {
    numStepsBelowTol = 0
    val fetches = mutable.ListBuffer.empty[Output]
    if (criteria.needEpoch) {
      epoch = Counter.get(Graph.Keys.GLOBAL_EPOCH, local = false, Op.currentGraph).getOrElse(
        throw new IllegalStateException(
          s"A ${Graph.Keys.GLOBAL_EPOCH.name} variable should be created in order to use the 'StopHook'."))
      epochFetchIndex = fetches.size
      fetches.append(epoch.value)
    }
    if (criteria.needStep) {
      step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false, Op.currentGraph).getOrElse(
        throw new IllegalStateException(
          s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'StopHook'."))
      stepFetchIndex = fetches.size
      fetches.append(step.value)
    }
    if (criteria.needLoss) {
      loss = Math.addN(Op.currentGraph.getCollection(Graph.Keys.LOSSES).toSeq)
      lossFetchIndex = fetches.size
      fetches.append(loss)
    }
    sessionFetches = fetches
  }

  override def afterSessionCreation(session: Session): Unit = {
    if (criteria.needEpoch)
      lastEpoch = criteria.maxEpochs.map(_ + session.run(fetches = epoch.value).scalar.asInstanceOf[Long])
    if (criteria.needStep)
      lastStep = criteria.maxSteps.map(_ + session.run(fetches = step.value).scalar.asInstanceOf[Long])
    if (criteria.needLoss)
      lastLoss = session.run(fetches = loss).scalar.asInstanceOf[Float]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    Some(Hook.SessionRunArgs(fetches = sessionFetches))
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
    if (criteria.maxEpochs.isDefined) {
      val epoch = runResult.values(epochFetchIndex).scalar.asInstanceOf[Long]
      if (lastEpoch.exists(epoch >= _)) {
        converged = true
        StopHook.logger.info("Stop requested: Exceeded maximum number of epochs.")
      }
    }
    if (criteria.maxSteps.isDefined) {
      val step = runResult.values(stepFetchIndex).scalar.asInstanceOf[Long]
      if (lastStep.exists(step >= _)) {
        converged = true
        StopHook.logger.info("Stop requested: Exceeded maximum number of steps.")
      }
    }
    if (criteria.absLossChangeTol.isDefined || criteria.relLossChangeTol.isDefined) {
      val loss = runResult.values(lossFetchIndex).scalar.asInstanceOf[Float]
      val lossDiff = scala.math.abs(lastLoss - loss)
      if (criteria.absLossChangeTol.exists(lossDiff < _) ||
          criteria.relLossChangeTol.exists(scala.math.abs(lossDiff / lastLoss) < _)) {
        numStepsBelowTol += 1
      } else {
        numStepsBelowTol = 0
      }
      if (numStepsBelowTol > criteria.maxStepBelowTol) {
        StopHook.logger.info("Stop requested: Loss value converged.")
        converged = true
      }
    }
    if (converged)
      runContext.requestStop()
  }
}

object StopHook {
  private[StopHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Termination"))
}
