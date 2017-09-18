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

import org.platanios.tensorflow.api.core.client.Executable
import org.platanios.tensorflow.api.core.client.Fetchable.Aux
import org.platanios.tensorflow.api.learn.Hook
import org.platanios.tensorflow.api.learn.Hook.{SessionRunArgs, SessionRunContext}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Monitors the provided tensor and stops training if, at any point, it contains any `NaN` values.
  *
  * This hook can either fail with an exception or just stop training.
  *
  * @param  tensorName Name of the tensor to monitor.
  * @param  failOnNaN  If `true`, an exception is thrown when `NaN` values are encountered. Otherwise, training stops.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorNaNHook(tensorName: String, failOnNaN: Boolean = true) extends Hook[Output, Traversable[Op], Tensor] {
  private[this] var output: Output = _

  override def begin(): Unit = output = Op.currentGraph.getOutputByName(tensorName)

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Aux[F, R]
  ): Option[Hook.SessionRunArgs[Output, Traversable[Op], Tensor]] = {
    Some(SessionRunArgs(fetches = output))
  }

  @throws[IllegalStateException]
  override def afterSessionRun[F, E, R](runContext: SessionRunContext[F, E, R], runValues: Tensor)(implicit
      executableEv: Executable[E],
      fetchableEv: Aux[F, R]): Unit = {
    if (runValues.isNaN.any().scalar.asInstanceOf[Boolean]) {
      val message = s"Encountered NaN values in tensor: $output."
      if (failOnNaN) {
        TensorNaNHook.logger.error(message)
        throw new IllegalStateException(message)
      } else {
        TensorNaNHook.logger.warn(message)
        // We do not raise an error but we request to stop iterating without throwing an exception.
        runContext.requestStop()
      }
    }
  }
}

object TensorNaNHook {
  private[TensorNaNHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Tensor NaN"))
}
