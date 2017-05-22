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

package org.platanios.tensorflow.data.loaders

import java.io.IOException
import java.net.URL
import java.nio.file.{Files, Path}

import com.typesafe.scalalogging.Logger

/**
  * @author Emmanouil Antonios Platanios
  */
trait Loader {
  protected val logger: Logger

  def maybeDownload(path: Path, url: String, bufferSize: Int = 8192): Boolean = {
    if (Files.exists(path)) {
      false
    } else {
      try {
        logger.info(s"Downloading file '$url'.")
        // TODO: [DATA] Add progress bar.
        val inputStream = new URL(url).openStream()
        val outputStream = Files.newOutputStream(path)
        val buffer = new Array[Byte](bufferSize)
        Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
        outputStream.close()
        logger.info(s"Downloaded file '$url'.")
        true
      } catch {
        case e: IOException =>
          logger.error(s"Could not download file '$url'", e)
          throw e
      }
    }
  }
}
