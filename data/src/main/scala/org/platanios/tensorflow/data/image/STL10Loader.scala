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

package org.platanios.tensorflow.data.image

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.Loader

import com.typesafe.scalalogging.Logger
import org.apache.commons.compress.archivers.tar._
import org.slf4j.LoggerFactory

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

/**
  * @author Emmanouil Antonios Platanios
  */
object STL10Loader extends Loader {
  val url                    : String = "http://ai.stanford.edu/~acoates/stl10/"
  val compressedFilename     : String = "stl10_binary.tar.gz"
  val trainImagesFilename    : String = "stl10_binary/train_X.bin"
  val trainLabelsFilename    : String = "stl10_binary/train_y.bin"
  val testImagesFilename     : String = "stl10_binary/test_X.bin"
  val testLabelsFilename     : String = "stl10_binary/test_y.bin"
  val unlabeledImagesFilename: String = "stl10_binary/unlabeled_X.bin"

  val numTrain     : Int = 5000
  val numTest      : Int = 8000
  val numUnlabeled : Int = 100000
  val imageWidth   : Int = 96
  val imageHeight  : Int = 96
  val imageChannels: Int = 3

  override protected val logger = Logger(LoggerFactory.getLogger("STL10 Data Loader"))

  def load(path: Path, bufferSize: Int = 8192, loadUnlabeled: Boolean = true): STL10Dataset = {
    // Download the data, if necessary.
    maybeDownload(path.resolve(compressedFilename), url + compressedFilename, bufferSize)

    // Load the data.
    val dataset = extractFiles(path.resolve(compressedFilename), bufferSize, loadUnlabeled)
    logger.info(s"Finished loading the STL-10 dataset.")
    dataset
  }

  private[this] def extractFiles(path: Path, bufferSize: Int = 8192, loadUnlabeled: Boolean = true): STL10Dataset = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new TarArchiveInputStream(new GZIPInputStream(Files.newInputStream(path)))
    var dataset = STL10Dataset(null, null, null, null, null)
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      if (entry.getName == unlabeledImagesFilename) {
        if (loadUnlabeled) {
          // TODO: Make this more efficient.
          // We have to split this tensor in two parts because its size exceeds the maximum allowed byte buffer size.
          val halfShape = Shape(numUnlabeled / 2, imageChannels, imageHeight, imageWidth)
          var outputStream = new ByteArrayOutputStream()
          val buffer = new Array[Byte](bufferSize)
          var numBytesRead = 0
          Stream.continually(inputStream.read(buffer))
              .takeWhile(_ => numBytesRead <= entry.getSize / 2)
              .foreach(numBytes => {
                outputStream.write(buffer, 0, numBytes)
                numBytesRead += numBytes
              })
          var byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
          outputStream.close()
          val tensor1 =
            Tensor.fromBuffer(UINT8, halfShape, byteBuffer.capacity(), byteBuffer).transpose(Tensor(0, 3, 2, 1))
          outputStream = new ByteArrayOutputStream()
          Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
          byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
          outputStream.close()
          val tensor2 =
            Tensor.fromBuffer(UINT8, halfShape, byteBuffer.capacity(), byteBuffer).transpose(Tensor(0, 3, 2, 1))
          dataset = dataset.copy(unlabeledImages = tfi.concatenate(Seq(tensor1, tensor2), axis = 0))
        }
      } else if (Set(
        trainImagesFilename, trainLabelsFilename,
        testImagesFilename, testLabelsFilename).contains(entry.getName)) {
        val outputStream = new ByteArrayOutputStream(entry.getSize.toInt)
        val buffer = new Array[Byte](bufferSize)
        Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
        val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
        outputStream.close()
        entry.getName match {
          case name if name == trainImagesFilename =>
            val shape = Shape(numTrain, imageChannels, imageHeight, imageWidth)
            val tensor = Tensor.fromBuffer(UINT8, shape, entry.getSize, byteBuffer).transpose(Tensor(0, 3, 2, 1))
            dataset = dataset.copy(trainImages = tensor)
          case name if name == trainLabelsFilename =>
            val tensor = Tensor.fromBuffer(UINT8, Shape(numTrain), entry.getSize, byteBuffer) - Tensor(UINT8, 1)
            dataset = dataset.copy(trainLabels = tensor)
          case name if name == testImagesFilename =>
            val shape = Shape(numTest, imageChannels, imageHeight, imageWidth)
            val tensor = Tensor.fromBuffer(UINT8, shape, entry.getSize, byteBuffer).transpose(Tensor(0, 3, 2, 1))
            dataset = dataset.copy(testImages = tensor)
          case name if name == testLabelsFilename =>
            val tensor = Tensor.fromBuffer(UINT8, Shape(numTest), entry.getSize, byteBuffer) - Tensor(UINT8, 1)
            dataset = dataset.copy(testLabels = tensor)
          case _ => ()
        }
      }
      entry = inputStream.getNextTarEntry
    }
    inputStream.close()
    dataset
  }
}

case class STL10Dataset(
    trainImages: Tensor,
    trainLabels: Tensor,
    testImages: Tensor,
    testLabels: Tensor,
    unlabeledImages: Tensor)
