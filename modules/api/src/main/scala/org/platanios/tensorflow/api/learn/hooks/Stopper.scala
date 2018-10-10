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
private[learn] class Stopper protected (protected var criteria: StopCriteria) extends Hook {
  override type StateF = (Option[Output[Long]], Option[Output[Long]], Option[Output[Float]])
  override type StateR = (Option[Tensor[Long]], Option[Tensor[Long]], Option[Tensor[Float]])

  private var epoch: Variable[Long] = _
  private var step : Variable[Long] = _
  private var loss : Output[Float]  = _

  private var startTime       : Long         = 0L
  private var lastEpoch       : Option[Long] = None
  private var lastStep        : Option[Long] = None
  private var lastLoss        : Float        = Float.MaxValue
  private var numStepsBelowTol: Int          = 0

  private var sessionFetches: StateF = _

  /** Updates the stop criteria used by this stop hook. This method is used by in-memory estimators. */
  def updateCriteria(criteria: StopCriteria): Unit = {
    this.criteria = criteria
  }

  /** Resets the state of this hook and should be called before initiating training. This method is used by in-memory
    * estimators. */
  def reset(session: Session): Unit = {
    startTime = System.currentTimeMillis()
    if (criteria.needEpoch) {
      val _lastEpoch = session.run(fetches = epoch.value).scalar.asInstanceOf[Long]
      lastEpoch = if (criteria.restartCounting) criteria.maxEpochs.map(_ + _lastEpoch) else criteria.maxEpochs
    }
    if (criteria.needStep) {
      val _lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
      lastStep = if (criteria.restartCounting) criteria.maxSteps.map(_ + _lastStep) else criteria.maxSteps
    }
    if (criteria.needLoss)
      lastLoss = session.run(fetches = loss).scalar.asInstanceOf[Float]
    numStepsBelowTol = 0
  }

  override protected def begin(): Unit = {
    if (criteria.maxSeconds.isDefined)
      startTime = System.currentTimeMillis()

    val epoch = {
      if (criteria.needEpoch) {
        this.epoch = Counter.get(Graph.Keys.GLOBAL_EPOCH, local = false, Op.currentGraph).getOrElse(
          throw new IllegalStateException(
            s"A ${Graph.Keys.GLOBAL_EPOCH.name} variable should be created in order to use the 'StopHook'."))
        Some(this.epoch.value)
      } else {
        None
      }
    }

    val step = {
      if (criteria.needStep) {
        this.step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false, Op.currentGraph).getOrElse(
          throw new IllegalStateException(
            s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'StopHook'."))
        Some(this.step.value)
      } else {
        None
      }
    }

    val loss = {
      if (criteria.needLoss) {
        this.loss = Math.addN(Op.currentGraph.getCollection(Graph.Keys.LOSSES).toSeq.map(_.toFloat))
        Some(this.loss)
      } else {
        None
      }
    }

    sessionFetches = (epoch, step, loss)
  }

  override protected def afterSessionCreation(session: Session): Unit = {
    reset(session)
  }

  override protected def beforeSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R]
  )(implicit
      evFetchable: Fetchable.Aux[F, R],
      evExecutable: Executable[E]
  ): Option[Hook.SessionRunArgs[StateF, StateE, StateR]] = {
    Some(Hook.SessionRunArgs(fetches = sessionFetches))
  }

  @throws[IllegalStateException]
  override protected def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[StateR]
  )(implicit
      evFetchable: Fetchable.Aux[F, R],
      evExecutable: Executable[E]
  ): Unit = {
    var converged = false
    runResult.result match {
      case (Some(e), _, _) if lastEpoch.exists(e.scalar >= _) =>
        Stopper.logger.debug("Stop requested: Exceeded maximum number of epochs.")
        converged = true
      case (_, Some(s), _) if lastStep.exists(s.scalar >= _) =>
        Stopper.logger.debug("Stop requested: Exceeded maximum number of steps.")
        converged = true
      case (_, _, Some(l)) =>
        val lossDiff = scala.math.abs(lastLoss - l.scalar)
        if (criteria.absLossChangeTol.exists(lossDiff < _) ||
            criteria.relLossChangeTol.exists(scala.math.abs(lossDiff / lastLoss) < _)) {
          numStepsBelowTol += 1
        } else {
          numStepsBelowTol = 0
        }
        if (numStepsBelowTol > criteria.maxStepBelowTol) {
          Stopper.logger.debug("Stop requested: Loss value converged.")
          converged = true
        }
      case _ => ()
    }
    criteria.maxSeconds.foreach(maxSeconds => {
      if (System.currentTimeMillis() - startTime >= maxSeconds) {
        Stopper.logger.debug("Stop requested: Exceeded maximum number of seconds.")
        converged = true
      }
    })
    if (converged)
      runContext.requestStop()
  }
}

object Stopper {
  private[Stopper] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Termination"))

  def apply(criteria: StopCriteria): Stopper = new Stopper(criteria)
}
