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
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Logs the values of the provided tensors based on a [[HookTrigger]], or at the end of a run (i.e., end of a
  * [[Session]]'s usage. The tensors will be printed using `INFO` logging level/severity. If you are not seeing the
  * logs, you might want to changing the logging level in your logging configuration file.
  *
  * Note that if `logAtEnd` is `true`, `tensors` should not include any tensor whose evaluation produces a side effect,
  * such as consuming additional inputs.
  *
  * @param  tensors      Map from tags to tensor names. The tags are used to identify the tensors in the log.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the [[Session.run()]]
  *                      call.
  * @param  formatter    Function used to format the strings being logged that takes a `Map[String, Tensor]` as input,
  *                      with the keys corresponding to tags, and returns a string to log. Defaults to a simple summary
  *                      of all the tensors in the map.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorLoggingHook(
    tensors: Map[String, String],
    trigger: HookTrigger = StepHookTrigger(1),
    triggerAtEnd: Boolean = false,
    formatter: (Map[String, Tensor]) => String = null)
    extends Hook {
  private[this] val tensorTags: Seq[String] = tensors.keys.toSeq
  private[this] val tensorNames: Seq[String] = tensors.values.toSeq
  private[this] var outputs: Seq[Output] = _
  private[this] var step         : Variable                  = _

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger: Boolean = false

  override def begin(): Unit = {
    step = Counter.get(Graph.Keys.GLOBAL_STEP, Op.currentGraph).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'TensorLoggingHook'."))
    internalTrigger.reset()
    shouldTrigger = false
    // Convert tensor names to op outputs.
    outputs = tensorNames.map(t => Op.currentGraph.getOutputByName(t))
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = internalTrigger.shouldTriggerForStep(lastStep.toInt)
    if (shouldTrigger)
      Some(Hook.SessionRunArgs(fetches = step.value +: outputs))
    else
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
      logTensors(tensorTags.zip(runResult.values.tail))
  }

  override def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1))
      logTensors(tensorTags.zip(session.run(fetches = outputs)))
  }

  /** Logs the provided tensor values. */
  private[this] def logTensors(tensors: Seq[(String, Tensor)]): Unit = {
    if (formatter != null) {
      TensorLoggingHook.logger.info(formatter(tensors.toMap))
    } else {
      val valuesLog = tensors.map(t => {
        s"${t._1} = ${t._2.summarize(flattened = true, includeInfo = false)}"
      }).mkString(", ")
      val log = internalTrigger.updateLastTrigger(lastStep.toInt - 1).map(_._1) match {
        case Some(s) => f"($s%.3f s) $valuesLog"
        case None => s"( N/A ) $valuesLog"
      }
      TensorLoggingHook.logger.info(log)
    }
  }
}

object TensorLoggingHook {
  private[TensorLoggingHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Tensor Logging"))
}
