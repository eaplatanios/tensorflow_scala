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
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Output, UntypedOp}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import org.tensorflow.framework.RunOptions

/** Hook that may be triggered at certain steps or time points.
  *
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to trigger this hook at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, the hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then the global step must be computable without using a feed map for the
  *                      [[Session.run()]] call (which should always be the case by default).
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class TriggeredHook(
    trigger: HookTrigger = StepHookTrigger(10),
    triggerAtEnd: Boolean = true
)(implicit
    evStructureOptionS: NestedStructure.Aux[Option[Seq[Output[Any]]], Option[Seq[Tensor[Any]]], _, _]
) extends Hook {
  protected val internalTrigger: HookTrigger    = trigger.copy()
  protected var step           : Variable[Long] = _
  protected var lastStep       : Long           = 0L
  protected var shouldTrigger  : Boolean        = false

  override private[learn] def internalBegin(): Unit = {
    internalTrigger.reset()
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use a triggered hook."))
    super.internalBegin()
  }

  override private[learn] def internalAfterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar
    if (lastStep == 0L)
      lastStep = -1L
    super.internalAfterSessionCreation(session)
  }

  override protected def beforeSessionRun[C, CV](
      runContext: Hook.SessionRunContext[C, CV]
  )(implicit
      evStructureC: NestedStructure.Aux[C, CV, _, _]
  ): Option[Hook.SessionRunArgs[Seq[Output[Any]], Seq[Tensor[Any]]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt + 1)
    if (shouldTrigger) {
      if (internalTrigger.lastTriggerStep().isEmpty)
        onFirstTrigger(runContext)
      Some(Hook.SessionRunArgs(
        fetches = step.value.asInstanceOf[Output[Any]] +: fetches,
        targets = targets,
        options = runOptions,
        wantMetadata = wantMetadata))
    } else {
      Some(Hook.SessionRunArgs(
        fetches = Seq(step.value),
        targets = Set.empty))
    }
  }

  override protected def afterSessionRun[C, CV](
      runContext: Hook.SessionRunContext[C, CV],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]]
  )(implicit
      evStructureC: NestedStructure.Aux[C, CV, _, _]
  ): Unit = {
    val result = runResult.result
    lastStep = result.head.scalar.asInstanceOf[Long]
    if (shouldTrigger) {
      val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt)
      val innerResult = Hook.SessionRunResult(result.tail, runResult.runMetadata)
      onTrigger(lastStep, elapsed, innerResult, runContext.session)
    }
  }

  override private[learn] def internalEnd(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1)) {
      val lastStep = session.run(fetches = step.value).scalar
      val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt)
      val results = session.run(fetches = fetches)
      onTrigger(lastStep, elapsed, Hook.SessionRunResult(results, None), session)
    }
    super.internalEnd(session)
  }

  protected def fetches: Seq[Output[Any]]
  protected def targets: Set[UntypedOp]

  protected def runOptions: Option[RunOptions] = None
  protected def wantMetadata: Boolean = false

  protected def onFirstTrigger[C, CV](
      runContext: Hook.SessionRunContext[C, CV]
  )(implicit evStructureC: NestedStructure.Aux[C, CV, _, _]): Unit = {
    ()
  }

  protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]],
      session: Session
  ): Unit = {
    ()
  }
}
