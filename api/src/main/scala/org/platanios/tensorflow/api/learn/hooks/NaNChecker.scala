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

import org.platanios.tensorflow.api.core.client.{Executable, Fetchable}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Monitors the provided tensors and stops training if, at any point, any of them contains any `NaN` values.
  *
  * This hook can either fail with an exception or just stop training.
  *
  * @param  tensorNames Names of the tensors to monitor.
  * @param  failOnNaN   If `true`, an exception is thrown when `NaN` values are encountered. Otherwise, training stops.
  *
  * @author Emmanouil Antonios Platanios
  */
class NaNChecker protected (
    val tensorNames: Set[String],
    val failOnNaN: Boolean = true
) extends Hook {
  private[this] var outputs: Seq[Output] = _

  override protected def begin(): Unit = {
    // Convert tensor names to op outputs.
    outputs = tensorNames.map(Op.currentGraph.getOutputByName).toSeq
  }

  override protected def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    Some(Hook.SessionRunArgs(fetches = outputs))
  }

  @throws[IllegalStateException]
  override protected def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    runResult.values.filter(_.isNaN.any().scalar.asInstanceOf[Boolean]).foreach(value => {
      val message = s"Encountered NaN values in tensor: $value."
      if (failOnNaN) {
        NaNChecker.logger.error(message)
        throw new IllegalStateException(message)
      } else {
        NaNChecker.logger.warn(message)
        // We do not raise an error but we request to stop iterating without throwing an exception.
        runContext.requestStop()
      }
    })
  }
}

object NaNChecker {
  private[NaNChecker] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Tensor NaN"))

  def apply(tensorNames: Set[String], failOnNaN: Boolean = true): NaNChecker = {
    new NaNChecker(tensorNames, failOnNaN)
  }
}
