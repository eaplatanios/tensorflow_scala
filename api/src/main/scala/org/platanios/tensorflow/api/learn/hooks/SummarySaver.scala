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
import org.platanios.tensorflow.api.io.events.{SummaryFileWriter, SummaryFileWriterCache}
import org.platanios.tensorflow.api.ops.{Output, Summary}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

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
class SummarySaver protected (
    val directory: Path,
    val trigger: HookTrigger = StepHookTrigger(10),
    val triggerAtEnd: Boolean = true,
    val collection: Graph.Key[Output] = Graph.Keys.SUMMARIES
) extends TriggeredHook(trigger, triggerAtEnd) {
  private[this] var summary      : Option[Output]            = None
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  override protected def begin(): Unit = {
    summary = Summary.mergeAll(collection)
    if (summary.isDefined)
      summaryWriter = Some(SummaryFileWriterCache.get(directory))
  }

  override protected def end(session: Session): Unit = summaryWriter.foreach(_.flush())

  override protected def fetches: Seq[Output] = summary.map(Seq(_)).getOrElse(Seq.empty)

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor[DataType]]],
      session: Session
  ): Unit = {
    summaryWriter.foreach(writer => {
      if (step == 0L)
        writer.writeSessionLog(SessionLog.newBuilder().setStatus(SessionLog.SessionStatus.START).build(), step)
      writer.writeSummaryString(runResult.values(0).scalar.asInstanceOf[String], step)
      writer.flush()
    })
  }
}

object SummarySaver {
  def apply(
      directory: Path,
      trigger: HookTrigger = StepHookTrigger(10),
      triggerAtEnd: Boolean = true,
      collection: Graph.Key[Output] = Graph.Keys.SUMMARIES
  ): SummarySaver = {
    new SummarySaver(directory, trigger, triggerAtEnd, collection)
  }
}
