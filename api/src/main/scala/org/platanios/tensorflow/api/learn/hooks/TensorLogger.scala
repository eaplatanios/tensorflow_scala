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

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.ops.{Op, Output}
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
class TensorLogger protected (
    val tensors: Map[String, String],
    val trigger: HookTrigger = StepHookTrigger(1),
    val triggerAtEnd: Boolean = true,
    val formatter: (Map[String, Tensor]) => String = null
) extends TriggeredHook(trigger, triggerAtEnd) {
  private[this] val tensorTags : Seq[String] = tensors.keys.toSeq
  private[this] val tensorNames: Seq[String] = tensors.values.toSeq
  private[this] var outputs    : Seq[Output] = _

  override protected def begin(): Unit = {
    // Convert tensor names to op outputs.
    outputs = tensorNames.map(t => Op.currentGraph.getOutputByName(t))
  }

  override protected def fetches: Seq[Output] = outputs

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]],
      session: Session
  ): Unit = {
    val tensors = tensorTags.zip(runResult.values.tail)
    if (formatter != null) {
      TensorLogger.logger.info(formatter(tensors.toMap))
    } else {
      val valuesLog = tensors.map(t => {
        s"${t._1} = ${t._2.summarize(flattened = true, includeInfo = false)}"
      }).mkString(", ")
      val log = elapsed.map(_._1) match {
        case Some(s) => f"($s%9.3f s) $valuesLog"
        case None => s"(    N/A    ) $valuesLog"
      }
      TensorLogger.logger.info(log)
    }
  }
}

object TensorLogger {
  private[TensorLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Tensor Logging"))

  def apply(
      tensors: Map[String, String],
      trigger: HookTrigger = StepHookTrigger(1),
      triggerAtEnd: Boolean = true,
      formatter: (Map[String, Tensor]) => String = null
  ): TensorLogger = {
    new TensorLogger(tensors, trigger, triggerAtEnd, formatter)
  }
}
