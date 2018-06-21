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

import org.platanios.tensorflow.api.core.client.{Session, Timeline}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.RunOptions

import java.nio.file.{Files, Path, StandardOpenOption}

/** Hook that saves Chrome trace files for visualizing execution timelines of TensorFlow steps.
  *
  * @param  workingDir   Directory in which to create the trace file. The file will be named `trace{step}.json`. Note
  *                      that this hook will overwrite any existing files with that name, in this directory.
  * @param  showDataFlow If `true`, add flow events to the trace connecting producers and consumers of tensors.
  * @param  showMemory   If `true`, add object snapshot events to the trace showing sizes and lifetimes of tensors.
  * @param  prettyJson   If `true`, produces human-readable JSON output.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the [[Session.run()]]
  *                      call.
  *
  * @author Emmanouil Antonios Platanios
  */
class TimelineHook protected (
    val workingDir: Path,
    val showDataFlow: Boolean = false,
    val showMemory: Boolean = false,
    val prettyJson: Boolean = false,
    val trigger: HookTrigger = StepHookTrigger(1000),
    val triggerAtEnd: Boolean = true
) extends TriggeredHook(trigger, triggerAtEnd) {
  override protected def runOptions: Option[RunOptions] = {
    Some(RunOptions.newBuilder().setTraceLevel(RunOptions.TraceLevel.FULL_TRACE).build())
  }

  override protected def wantMetadata: Boolean = true

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor[DataType]]],
      session: Session
  ): Unit = {
    TimelineHook.logger.info("Saving timeline.")
    val file = workingDir.resolve(s"trace$step.json")
    val stepStatistics = runResult.runMetadata.get.getStepStats
    val chromeTraceJSON = Timeline.generateChromeTrace(stepStatistics, showDataFlow, showMemory, prettyJson)
    val fileWriter = Files.newBufferedWriter(file, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    fileWriter.write(chromeTraceJSON)
    fileWriter.flush()
    fileWriter.close()
    TimelineHook.logger.info(s"Saved timeline to '$file'.")
  }
}

object TimelineHook {
  private[TimelineHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Timeline"))

  def apply(
      workingDir: Path,
      showDataFlow: Boolean = false,
      showMemory: Boolean = false,
      prettyJson: Boolean = false,
      trigger: HookTrigger = StepHookTrigger(1000),
      triggerAtEnd: Boolean = true
  ): TimelineHook = {
    new TimelineHook(workingDir, showDataFlow, showMemory, prettyJson, trigger, triggerAtEnd)
  }
}
