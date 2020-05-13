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

package org.platanios.tensorflow.data

import com.typesafe.scalalogging.Logger

import java.io.IOException
import java.net.URL
import java.nio.file.{Files, Path}

import scala.collection.compat.immutable.LazyList
import scala.io.Source
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
trait Loader {
  protected val logger: Logger

  protected val googleDriveConfirmTokenRegex: Regex = {
    """<a id="uc-download-link".*href="/uc\?export=download&amp;(confirm=.*)&amp;id=.*">Download anyway</a>""".r
  }

  def maybeDownload(path: Path, url: String, bufferSize: Int = 8192): Boolean = {
    if (Files.exists(path)) {
      false
    } else {
      try {
        logger.info(s"Downloading file '$url'.")
        Files.createDirectories(path.getParent)
        download(path, url, bufferSize)

        // Small hack to deal with downloading large Google Drive files.
        if (Files.size(path) < 1024 * 1024 && url.contains("drive.google.com")) {
          val content = Source.fromFile(path.toFile).getLines().mkString("\n")
          googleDriveConfirmTokenRegex.findFirstMatchIn(content) match {
            case Some(confirmToken) => download(path, s"$url&${confirmToken.group(1)}", bufferSize)
            case None => ()
          }
        }

        logger.info(s"Downloaded file '$url'.")
        true
      } catch {
        case e: IOException =>
          logger.error(s"Could not download file '$url'", e)
          throw e
      }
    }
  }

  protected def download(path: Path, url: String, bufferSize: Int = 8192): Unit = {
    val connection = new URL(url).openConnection()
    val contentLength = connection.getContentLengthLong
    val inputStream = connection.getInputStream
    val outputStream = Files.newOutputStream(path)
    val buffer = new Array[Byte](bufferSize)
    var progress = 0L
    var progressLogTime = System.currentTimeMillis
    LazyList.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(numBytes => {
      outputStream.write(buffer, 0, numBytes)
      progress += numBytes
      val time = System.currentTimeMillis
      if (time - progressLogTime >= 1e4) {
        if (contentLength > 0) {
          val numBars = Math.floorDiv(10 * progress, contentLength).toInt
          logger.info(s"[${"=" * numBars}${" " * (10 - numBars)}] $progress / $contentLength bytes downloaded.")
          progressLogTime = time
        } else {
          logger.info(s"$progress bytes downloaded.")
          progressLogTime = time
        }
      }
    })
    outputStream.close()
  }
}
