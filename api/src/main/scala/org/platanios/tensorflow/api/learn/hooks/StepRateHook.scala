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
import org.platanios.tensorflow.api.io.{SummaryFileWriter, SummaryFileWriterCache}
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.Summary

import java.nio.file.Path

/** Saves summaries to files based on a [[HookTrigger]].
  *
  * @param  log              If `true`, the step rate is logged using the current logging configuration.
  * @param  summaryDirectory If provided, summaries for the step rate will be saved in this directory. This is useful
  *                          for visualization using TensorBoard, for example.
  * @param  trigger          Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only
  *                          want to trigger this hook at the end of a run and not during, then you should set
  *                          `trigger` to [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd     If `true`, the hook will be triggered at the end of the run. Note that if this flag is set
  *                          to `true`, then the global step must be computable without using a feed map for the
  *                          [[Session.run()]] call (which should always be the case by default).
  * @param  tag              Tag to use for the step rate when logging and saving summaries.
  *
  * @author Emmanouil Antonios Platanios
  */
case class StepRateHook(
    log: Boolean = true,
    summaryDirectory: Path = null,
    trigger: HookTrigger = StepHookTrigger(10),
    triggerAtEnd: Boolean = true,
    tag: String = "Steps/Sec"
) extends Hook {
  require(log || summaryDirectory != null, "At least one of 'log' and 'summaryDirectory' needs to be provided.")

  private[this] var step         : Variable                  = _
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override def begin(): Unit = {
    internalTrigger.reset()
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'StepRateHook'."))
    summaryWriter = Option(summaryDirectory).map(SummaryFileWriterCache.get)
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt) && (log || summaryWriter.isDefined)
    Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    saveStepRateSummary(runResult.values)
  }

  override def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1))
      saveStepRateSummary(session.run(fetches = Seq(step.value)))
    summaryWriter.foreach(_.flush())
  }

  private[this] def saveStepRateSummary(fetches: Seq[Tensor]): Unit = {
    lastStep = fetches(0).scalar.asInstanceOf[Long]
    if (shouldTrigger) {
      internalTrigger.updateLastTrigger(lastStep.toInt - 1).foreach(elapsed => {
        val stepRate = elapsed._2.toDouble / elapsed._1
        if (log)
          StepRateHook.logger.info(f"$tag: $stepRate%.2f")
        summaryWriter.foreach(_.writeSummary(
          Summary.newBuilder()
              .addValue(Summary.Value.newBuilder()
                            .setTag(tag)
                            .setSimpleValue(stepRate.toFloat))
                  .build(), lastStep))
      })
    }
  }
}

object StepRateHook {
  private[StepRateHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Step Rate"))
}
