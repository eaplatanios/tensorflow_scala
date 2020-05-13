/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToTensor}
import org.platanios.tensorflow.api.learn.{Counter, StopCriteria}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.math.Math
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
  private var epoch: Output[Long] = _
  private var step : Output[Long] = _
  private var loss : Output[Float]  = _

  private var startTime       : Long         = 0L
  private var lastEpoch       : Option[Long] = None
  private var lastStep        : Option[Long] = None
  private var lastLoss        : Float        = Float.MaxValue
  private var numStepsBelowTol: Int          = 0

  private var sessionFetches : Seq[Output[Any]] = _
  private var epochFetchIndex: Int              = _
  private var stepFetchIndex : Int              = _
  private var lossFetchIndex : Int              = _

  /** Updates the stop criteria used by this stop hook. This method is used by in-memory estimators. */
  def updateCriteria(criteria: StopCriteria): Unit = {
    this.criteria = criteria
  }

  /** Resets the state of this hook and should be called before initiating training. This method is used by in-memory
    * estimators. */
  def reset(session: Session): Unit = {
    startTime = System.currentTimeMillis()
    if (criteria.needEpoch) {
      val _lastEpoch = session.run(fetches = epoch).scalar
      lastEpoch = if (criteria.restartCounting) criteria.maxEpochs.map(_ + _lastEpoch) else criteria.maxEpochs
    }
    if (criteria.needStep) {
      val _lastStep = session.run(fetches = step).scalar
      lastStep = if (criteria.restartCounting) criteria.maxSteps.map(_ + _lastStep) else criteria.maxSteps
    }
    if (criteria.needLoss)
      lastLoss = session.run(fetches = loss).scalar
    numStepsBelowTol = 0
  }

  override protected def begin(): Unit = {
    val fetches = mutable.ListBuffer.empty[Output[Any]]
    if (criteria.maxSeconds.isDefined)
      startTime = System.currentTimeMillis()
    if (criteria.needEpoch) {
      epoch = Counter.get(Graph.Keys.GLOBAL_EPOCH, local = false, Op.currentGraph).getOrElse(
        throw new IllegalStateException(
          s"A ${Graph.Keys.GLOBAL_EPOCH.name} variable should be created in order to use the 'StopHook'.")
      ).value
      epochFetchIndex = fetches.size
      fetches.append(epoch)
    }
    if (criteria.needStep) {
      step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false, Op.currentGraph).getOrElse(
        throw new IllegalStateException(
          s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'StopHook'.")
      ).value
      stepFetchIndex = fetches.size
      fetches.append(step)
    }
    if (criteria.needLoss) {
      loss = Math.addN(Op.currentGraph.getCollection(Graph.Keys.LOSSES).toSeq.map(_.toFloat))
      lossFetchIndex = fetches.size
      fetches.append(loss)
    }
    sessionFetches = fetches.toSeq
  }

  override protected def afterSessionCreation(session: Session): Unit = {
    reset(session)
  }

  override protected def beforeSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV]
  )(implicit
      evOutputToTensorC: OutputToTensor.Aux[C, CV]
  ): Option[Hook.SessionRunArgs[Seq[Output[Any]], Seq[Tensor[Any]]]] = {
    Some(Hook.SessionRunArgs(fetches = sessionFetches))
  }

  @throws[IllegalStateException]
  override protected def afterSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]]
  )(implicit evOutputToTensorC: OutputToTensor.Aux[C, CV]): Unit = {

    var converged = false
    if (criteria.maxEpochs.isDefined) {
      val epoch = runResult.result(epochFetchIndex).scalar.asInstanceOf[Long]
      if (lastEpoch.exists(epoch >= _)) {
        Stopper.logger.debug("Stop requested: Exceeded maximum number of epochs.")
        converged = true
      }
    }
    if (criteria.maxSteps.isDefined) {
      val step = runResult.result(stepFetchIndex).scalar.asInstanceOf[Long]
      if (lastStep.exists(step >= _)) {
        Stopper.logger.debug("Stop requested: Exceeded maximum number of steps.")
        converged = true
      }
    }
    criteria.maxSeconds.foreach(maxSeconds => {
      if (System.currentTimeMillis() - startTime >= maxSeconds) {
        Stopper.logger.debug("Stop requested: Exceeded maximum number of seconds.")
        converged = true
      }
    })
    if (criteria.absLossChangeTol.isDefined || criteria.relLossChangeTol.isDefined) {
      val loss = runResult.result(lossFetchIndex).scalar.asInstanceOf[Float]
      val lossDiff = scala.math.abs(lastLoss - loss)
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
    }
    if (converged)
      runContext.requestStop()
  }
}

object Stopper {
  private[Stopper] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Termination"))

  def apply(criteria: StopCriteria): Stopper = new Stopper(criteria)
}
