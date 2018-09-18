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

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/** Saves summaries to files based on a [[HookTrigger]].
  *
  * @param  log          If `true`, the step rate is logged using the current logging configuration.
  * @param  summaryDir   If provided, summaries for the step rate will be saved in this directory. This is useful for
  *                      visualization using TensorBoard, for example.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to trigger this hook at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, the hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then the global step must be computable without using a feed map for the
  *                      [[Session.run()]] call (which should always be the case by default).
  * @param  tag          Tag to use for the step rate when logging and saving summaries.
  *
  * @author Emmanouil Antonios Platanios
  */
class StepRateLogger protected (
    val log: Boolean = true,
    val summaryDir: Path = null,
    val trigger: HookTrigger = StepHookTrigger(10),
    val triggerAtEnd: Boolean = true,
    val tag: String = "Steps/Sec"
) extends TriggeredHook(trigger, triggerAtEnd) with SummaryWriterHookAddOn {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor[_]]],
      session: Session
  ): Unit = {
    elapsed.foreach(elapsed => {
      val stepRate = elapsed._2.toDouble / elapsed._1
      if (log)
        StepRateLogger.logger.info(f"$tag: $stepRate%.2f")
      writeSummary(step, tag, stepRate.toFloat)
    })
  }
}

object StepRateLogger {
  private[StepRateLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Step Rate"))

  def apply(
      log: Boolean = true,
      summaryDir: Path = null,
      trigger: HookTrigger = StepHookTrigger(10),
      triggerAtEnd: Boolean = true,
      tag: String = "Steps/Sec"
  ): StepRateLogger = {
    new StepRateLogger(log, summaryDir, trigger, triggerAtEnd, tag)
  }
}
