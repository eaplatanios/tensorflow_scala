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

/** Hook that logs the loss function value.
  *
  * @param  log          If `true`, the loss will also be logged at the `INFO` level, using the current logging
  *                      configuration.
  * @param  summaryDir   Summary directory to use optionally, for saving scalar summaries of the loss. These can be
  *                      useful for visualizing training using TensorBoard.
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the [[Session.run()]]
  *                      call.
  * @param  formatter    Function used to format the message that is being logged. It takes the time taken since the
  *                      last logged message, the current step, and the current loss value, as input, and returns a
  *                      string to log.
  *
  * @author Emmanouil Antonios Platanios
  */
class LossLogger protected (
    val log: Boolean = true,
    val summaryDir: Path = null,
    val trigger: HookTrigger = StepHookTrigger(1),
    val triggerAtEnd: Boolean = true,
    val formatter: (Option[Double], Long, Float) => String = null
) extends TriggeredHook(trigger, triggerAtEnd)
    with ModelDependentHook[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]
    with SummaryWriterHookAddOn {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")

  override type InnerStateF = Output[Float]
  override type InnerStateR = Tensor[Float]

  private[this] var loss: Output[Float] = _

  override protected def begin(): Unit = {
    loss = modelInstance.loss.map(_.castTo[Float]).orNull
  }

  override protected def fetches: Output[Float] = loss

  override protected def onTrigger(
      step: Long,
      elapsed: Option[(Double, Int)],
      runResult: Hook.SessionRunResult[InnerStateR],
      session: Session
  ): Unit = {
    val loss = runResult.result.scalar
    val log = {
      if (formatter != null) {
        formatter(elapsed.map(_._1), step, loss)
      } else {
        elapsed.map(_._1) match {
          case Some(s) => f"($s%9.3f s) Step: $step%6d, Loss: $loss%.4f"
          case None => f"(    N/A    ) Step: $step%6d, Loss: $loss%.4f"
        }
      }
    }
    LossLogger.logger.info(log)
    writeSummary(step, "Loss", loss)
  }
}

object LossLogger {
  private[LossLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Loss Logger"))

  def apply(
      log: Boolean = true,
      summaryDir: Path = null,
      trigger: HookTrigger = StepHookTrigger(1),
      triggerAtEnd: Boolean = true,
      formatter: (Option[Double], Long, Float) => String = null
  ): LossLogger = {
    new LossLogger(log, summaryDir, trigger, triggerAtEnd, formatter)
  }
}
