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

import org.platanios.tensorflow.api.config.TensorBoardConfig
import org.platanios.tensorflow.api.core.client.{Executable, Fetchable, Session}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.Try

/** Launches a TensorBoard server for the duration of a run.
  *
  * This can be useful when running on a server or a distributed environment and want to monitor the run.
  *
  * @param  tensorBoardConfig TensorBoard configuration to use.
  *
  * @author Emmanouil Antonios Platanios
  */
private[learn] class TensorBoardHook protected (val tensorBoardConfig: TensorBoardConfig) extends Hook {
  override type StateF = Unit
  override type StateE = Unit
  override type StateR = Unit

  override protected implicit val evFetchableState: Fetchable.Aux[StateF, StateR] = {
    implicitly[Fetchable.Aux[StateF, StateR]]
  }

  override protected implicit val evExecutableState: Executable[StateE] = {
    implicitly[Executable[StateE]]
  }

  private[this] var tensorBoardProcess: Option[Process] = None

  override protected def begin(): Unit = tensorBoardProcess = {
    Option(tensorBoardConfig).flatMap(config => {
      TensorBoardHook.logger.info(
        s"Launching TensorBoard in '${config.host}:${config.port}' " +
            s"for log directory '${config.logDir.toAbsolutePath}'.")
      val processOrError = Try(config.processBuilder.start())
      processOrError.failed.foreach(e => {
        TensorBoardHook.logger.warn(e.getMessage)
        TensorBoardHook.logger.warn(
          "Could not launch TensorBoard. Please make sure it is installed correctly and in your PATH.")
      })
      processOrError.toOption
    })
  }

  override protected def end(session: Session): Unit = {
    tensorBoardProcess.foreach(process => {
      TensorBoardHook.logger.info("Killing the TensorBoard service.")
      process.destroy()
    })
  }
}

private[learn] object TensorBoardHook {
  private[TensorBoardHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / TensorBoard"))

  def apply(tensorBoardConfig: TensorBoardConfig): TensorBoardHook = new TensorBoardHook(tensorBoardConfig)
}
