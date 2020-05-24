/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToTensor}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.tensors.{Tensor, ops}
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
  private var outputs: Seq[Output[Any]] = _

  override protected def begin(): Unit = {
    // Convert tensor names to op outputs.
    outputs = tensorNames.map(Op.currentGraph.getOutputByName).toSeq
  }

  override protected def beforeSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV]
  )(implicit
      evOutputToTensorC: OutputToTensor.Aux[C, CV]
  ): Option[Hook.SessionRunArgs[Seq[Output[Any]], Seq[Tensor[Any]]]] = {
    Some(Hook.SessionRunArgs(fetches = outputs))
  }

  @throws[IllegalStateException]
  override protected def afterSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]]
  )(implicit evOutputToTensorC: OutputToTensor.Aux[C, CV]): Unit = {
    // TODO: [TYPES] !!! Remove the cast once we start using static types everywhere.
    runResult.result.filter(r => ops.Math.any(ops.Math.isNaN(r.toFloat)).scalar).foreach(value => {
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
