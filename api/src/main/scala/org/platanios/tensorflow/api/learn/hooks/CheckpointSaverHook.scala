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
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.{Saver, Variable}
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
case class CheckpointSaverHook(
    directory: Path,
    trigger: HookTrigger = StepHookTrigger(1000),
    triggerAtEnd: Boolean = true,
    checkpointBaseName: String = "model.ckpt"
) extends Hook {
  private[this] val savePath: Path = directory.resolve(checkpointBaseName)

  private[this] var step         : Variable                  = _
  private[this] var saver        : Option[Saver]             = None
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override def begin(): Unit = {
    internalTrigger.reset()
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw InvalidArgumentException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'CheckpointSaverHook'."))
    val savers = Op.currentGraph.getCollection(Graph.Keys.SAVERS)
    if (savers.isEmpty || savers.size > 1)
      throw InvalidArgumentException("There should exist one (and only one) saver in the graph.")
    saver = Some(savers.head)
    summaryWriter = Some(SummaryFileWriterCache.get(directory))
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    if (internalTrigger.lastTriggerStep().isEmpty) {
      // We save the graph and the saver at the first call of `beforeSessionRun`. We cannot do this in `begin()` because
      // we let other hooks change the graph and add variables in their `begin()` methods. The graph is finalized after
      // all `begin()` calls.
      val graphDef = runContext.session.graph.toGraphDef
      val metaGraphDef = runContext.session.graph.toMetaGraphDef(saverDef = saver.map(_.toSaverDef()).orNull)
      Files.write(directory.resolve("graph.pbtxt"), graphDef.toByteArray)
      summaryWriter.foreach(_.writeGraphDef(graphDef))
      summaryWriter.foreach(_.writeMetaGraphDef(metaGraphDef))
    }
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt)
    Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    lastStep = runResult.values(0).scalar.asInstanceOf[Long]
    if (shouldTrigger)
      save(runContext.session)
  }

  override def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1))
      save(session)
    summaryWriter.foreach(_.flush())
  }

  private[this] def save(session: Session): Unit = {
    internalTrigger.updateLastTrigger(lastStep.toInt - 1)
    CheckpointSaverHook.logger.info(s"Saving checkpoint for step $lastStep.")
    saver.foreach(_.save(session, savePath, Some(lastStep.toInt)))
    summaryWriter.foreach(_.writeSessionLog(
      SessionLog.newBuilder()
          .setStatus(SessionLog.SessionStatus.CHECKPOINT)
          .setCheckpointPath(savePath.toAbsolutePath.toString)
          .build()))
  }
}

object CheckpointSaverHook {
  private[CheckpointSaverHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Checkpoint Saver"))
}
