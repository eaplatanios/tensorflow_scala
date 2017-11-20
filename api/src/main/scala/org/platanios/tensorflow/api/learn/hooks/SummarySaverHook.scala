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
import org.platanios.tensorflow.api.io.events.{SummaryFileWriter, SummaryFileWriterCache}
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Op, Output, Summary}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import org.tensorflow.util.SessionLog

import java.nio.file.Path

/** Saves summaries to files based on a [[HookTrigger]].
  *
  * @param  directory    Directory in which to save the summaries.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to save the summary values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then all summaries must be computable without using a feed map for the
  *                      [[Session.run()]] call.
  * @param  collection   Graph collection from which to obtain the summaries. Defaults to `Graph.Keys.SUMMARIES`.
  *
  * @author Emmanouil Antonios Platanios
  */
case class SummarySaverHook(
    directory: Path,
    trigger: HookTrigger = StepHookTrigger(10),
    triggerAtEnd: Boolean = true,
    collection: Graph.Key[Output] = Graph.Keys.SUMMARIES
) extends Hook {
  private[this] var step         : Variable                  = _
  private[this] var summary      : Option[Output]            = None
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override def begin(): Unit = {
    internalTrigger.reset()
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'SummarySaverHook'."))
    summary = Summary.mergeAll(collection)
    if (summary.isDefined)
      summaryWriter = Some(SummaryFileWriterCache.get(directory))
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt) && summary.isDefined
    if (shouldTrigger) {
      Some(Hook.SessionRunArgs(fetches = Seq(step.value, summary.get)))
    } else {
      Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
    }
  }

  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    saveSummaries(runResult.values)
  }

  override def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1))
      saveSummaries(session.run(fetches = Seq(step.value, summary.get)))
    summaryWriter.foreach(_.flush())
  }

  private[this] def saveSummaries(fetches: Seq[Tensor]): Unit = {
    summaryWriter.foreach(writer => {
      if (lastStep == 0L)
        writer.writeSessionLog(SessionLog.newBuilder().setStatus(SessionLog.SessionStatus.START).build(), lastStep)
      lastStep = fetches(0).scalar.asInstanceOf[Long]
      if (shouldTrigger) {
        internalTrigger.updateLastTrigger(lastStep.toInt - 1)
        writer.writeSummaryString(fetches(1).scalar.asInstanceOf[String], lastStep)
        writer.flush()
      }
    })
  }
}
