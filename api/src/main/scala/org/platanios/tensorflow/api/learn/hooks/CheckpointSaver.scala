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
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.io.events.{SummaryFileWriter, SummaryFileWriterCache}
import org.platanios.tensorflow.api.learn.SessionWrapper
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Saver
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.util.SessionLog

import java.nio.file.{Files, Path}

/** Saves checkpoints to files based on a [[HookTrigger]]. Checkpoints include the current graph, as well as the trained
  * values of all variables, so far.
  *
  * @param  directory          Directory in which to save the checkpoints.
  * @param  trigger            Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only
  *                            want to save the summary values at the end of a run and not during, then you should set
  *                            `trigger` to [[NoHookTrigger]] and `triggerAtEnd` to `true`.
  * @param  triggerAtEnd       If `true`, this hook will be triggered at the end of the run. Note that if this flag is
  *                            set to `true`, then all summaries must be computable without using a feed map for the
  *                            [[Session.run()]] call.
  * @param  checkpointBaseName Base name for the checkpoint files.
  *
  * @author Emmanouil Antonios Platanios
  */
case class CheckpointSaver(
    directory: Path,
    trigger: HookTrigger = StepHookTrigger(1000),
    triggerAtEnd: Boolean = true,
    checkpointBaseName: String = "model.ckpt"
) extends TriggeredHook(trigger, triggerAtEnd) {
  override private[learn] val priority: Int = 1000

  private[this] val savePath: Path = directory.resolve(checkpointBaseName)

  private[this] var saver        : Option[Saver]             = None
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  override protected def begin(): Unit = {
    val savers = Op.currentGraph.getCollection(Graph.Keys.SAVERS)
    if (savers.isEmpty || savers.size > 1)
      throw InvalidArgumentException("There should exist one (and only one) saver in the graph.")
    saver = Some(savers.head)
    summaryWriter = Some(SummaryFileWriterCache.get(directory))
  }

  override protected def end(session: Session): Unit = {
    summaryWriter.foreach(_.flush())
  }

  override protected def onFirstTrigger[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    // We save the graph and the saver at the first call of `beforeSessionRun`. We cannot do this in `begin()` because
    // we let other hooks change the graph and add variables in their `begin()` methods. The graph is finalized after
    // all `begin()` calls.
    val graphDef = runContext.session.graph.toGraphDef
    val metaGraphDef = runContext.session.graph.toMetaGraphDef(saverDef = saver.map(_.toSaverDef()).orNull)
    Files.write(directory.resolve("graph.pbtxt"), graphDef.toByteArray)
    summaryWriter.foreach(_.writeGraphDef(graphDef))
    summaryWriter.foreach(_.writeMetaGraphDef(metaGraphDef))
  }

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]],
      session: Session
  ): Unit = {
    CheckpointSaver.logger.info(s"Saving checkpoint for step $step.")
    session match {
      case s: SessionWrapper => s.disableHooks()
      case _ => ()
    }
    saver.foreach(_.save(session, savePath, Some(step.toInt)))
    session match {
      case s: SessionWrapper => s.enableHooks()
      case _ => ()
    }
    summaryWriter.foreach(_.writeSessionLog(
      SessionLog.newBuilder()
          .setStatus(SessionLog.SessionStatus.CHECKPOINT)
          .setCheckpointPath(savePath.toAbsolutePath.toString)
          .build()))
  }
}

object CheckpointSaver {
  private[CheckpointSaver] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Checkpoint Saver"))
}
