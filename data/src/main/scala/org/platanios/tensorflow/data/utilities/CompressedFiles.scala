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

package org.platanios.tensorflow.data.utilities

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.utils.IOUtils

import java.io.{File, FileOutputStream, InputStream}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

/**
  * @author Emmanouil Antonios Platanios
  */
object CompressedFiles {
  def decompressTGZ(tgzFilePath: Path, destinationPath: Path, bufferSize: Int = 8192): Unit = {
    decompressTGZStream(Files.newInputStream(tgzFilePath), destinationPath, bufferSize)
  }

  def decompressTar(tarFilePath: Path, destinationPath: Path, bufferSize: Int = 8192): Unit = {
    decompressTarStream(Files.newInputStream(tarFilePath), destinationPath, bufferSize)
  }

  def decompressTGZStream(tgzStream: InputStream, destinationPath: Path, bufferSize: Int = 8192): Unit = {
    decompressTarStream(new GZIPInputStream(tgzStream), destinationPath, bufferSize)
  }

  def decompressTarStream(tarStream: InputStream, destinationPath: Path, bufferSize: Int = 8192): Unit = {
    val inputStream = new TarArchiveInputStream(tarStream)
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      if (!entry.isDirectory) {
        val currentFile = new File(destinationPath.toAbsolutePath.toString, entry.getName)
        val parentFile = currentFile.getParentFile
        if (!parentFile.exists)
          parentFile.mkdirs()
        IOUtils.copy(inputStream, new FileOutputStream(currentFile))
      }
      entry = inputStream.getNextTarEntry
    }
    inputStream.close()
  }
}
