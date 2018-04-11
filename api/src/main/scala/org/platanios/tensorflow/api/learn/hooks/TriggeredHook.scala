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
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Op, Output}
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
) extends Hook {
  protected     val internalTrigger: HookTrigger = trigger.copy()
  private[this] var step           : Variable    = _
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override private[learn] def internalBegin(): Unit = {
    internalTrigger.reset()
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use a triggered hook."))
    super.internalBegin()
  }

  override private[learn] def internalAfterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
    if (lastStep == 0L)
      lastStep = -1L
    super.internalAfterSessionCreation(session)
  }

  override final protected def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt + 1)
    if (shouldTrigger) {
      if (internalTrigger.lastTriggerStep().isEmpty)
        onFirstTrigger(runContext)(executableEv, fetchableEv)
      Some(Hook.SessionRunArgs(fetches = step.value +: fetches, options = runOptions, wantMetadata = wantMetadata))
    } else {
      Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
    }
  }

  override final protected def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    lastStep = runResult.values(0).scalar.asInstanceOf[Long]
    if (shouldTrigger) {
      val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt)
      onTrigger(lastStep, elapsed, runResult.copy(values = runResult.values.tail), runContext.session)
    }
  }

  override private[learn] def internalEnd(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1)) {
      val lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
      val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt)
      onTrigger(lastStep, elapsed, Hook.SessionRunResult(session.run(fetches = fetches), None), session)
    }
    super.internalEnd(session)
  }

  protected def fetches: Seq[Output] = Seq.empty[Output]
  protected def runOptions: Option[RunOptions] = None
  protected def wantMetadata: Boolean = false

  protected def onFirstTrigger[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = ()

  protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]],
      session: Session
  ): Unit = ()
}
