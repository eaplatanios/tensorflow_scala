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

package org.platanios.tensorflow.api.config

import org.platanios.tensorflow.api.learn.estimators.Estimator

import java.nio.file.Path

/** TensorBoard configuration, which can be used when training using [[Estimator]]s.
  *
  * @param  logDir         Directory containing the logs and summaries that the TensorBoard instance should use.
  * @param  host           Host to use for the TensorBoard service.
  * @param  port           Port to use for the TensorBoard service.
  * @param  reloadInterval Interval at which the backend reloads more data in seconds.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TensorBoardConfig(
    logDir: Path,
    host: String = "localhost",
    port: Int = 6006,
    reloadInterval: Int = 5) {
  private[api] val processBuilder = new ProcessBuilder(
    "python", "-m", "tensorboard.main",
    "--logdir", logDir.toAbsolutePath.toString,
    "--host", host,
    "--port", port.toString,
    "--reload_interval", reloadInterval.toString)
}
