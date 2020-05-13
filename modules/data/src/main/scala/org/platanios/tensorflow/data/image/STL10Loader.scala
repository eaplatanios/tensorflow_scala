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

package org.platanios.tensorflow.data.image

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.UByte
import org.platanios.tensorflow.data.Loader
import org.platanios.tensorflow.data.utilities.UniformSplit

import com.typesafe.scalalogging.Logger
import org.apache.commons.compress.archivers.tar._
import org.slf4j.LoggerFactory

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

import scala.collection.compat.immutable.LazyList
import scala.collection.mutable

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

  private def extractFiles(path: Path, bufferSize: Int = 8192, loadUnlabeled: Boolean = true): STL10Dataset = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new TarArchiveInputStream(new GZIPInputStream(Files.newInputStream(path)))
    var dataset = STL10Dataset(null, null, null, null, null)
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      if (entry.getName == unlabeledImagesFilename) {
        if (loadUnlabeled) {
          // TODO: Make this more efficient.
          // We have to split this tensor in parts because its size exceeds the maximum allowed byte buffer size.
          val maxNumBytesPerPart = Int.MaxValue / 2
          val parts = mutable.ListBuffer.empty[Tensor[UByte]]
          val dataType = UINT8
          val buffer = new Array[Byte](bufferSize)
          var numRemainingBytes = entry.getSize
          while (numRemainingBytes > maxNumBytesPerPart) {
            val numElementsToRead = math.floor(maxNumBytesPerPart / dataType.byteSize.get).toInt
            val numBytesToRead = numElementsToRead * dataType.byteSize.get
            val numSamplesToRead = numElementsToRead / (imageChannels * imageHeight * imageWidth)
            val shape = Shape(numSamplesToRead, imageChannels, imageHeight, imageWidth)
            val outputStream = new ByteArrayOutputStream()
            var numBytesRead = 0
            LazyList.continually(inputStream.read(buffer))
                .takeWhile(_ => numBytesRead <= numBytesToRead)
                .foreach(numBytes => {
                  outputStream.write(buffer, 0, numBytes)
                  numBytesRead += numBytes
                })
            val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
            outputStream.close()
            val tensor = Tensor.fromBuffer[UByte](shape, byteBuffer.capacity(), byteBuffer)
            parts.append(tensor.transpose(Tensor(0, 3, 2, 1)))
            numRemainingBytes -= numBytesToRead
          }
          dataset = dataset.copy(unlabeledImages = tfi.concatenate(parts.toSeq, axis = 0))
        }
      } else if (Set(
        trainImagesFilename, trainLabelsFilename,
        testImagesFilename, testLabelsFilename).contains(entry.getName)) {
        val outputStream = new ByteArrayOutputStream(entry.getSize.toInt)
        val buffer = new Array[Byte](bufferSize)
        LazyList.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
        val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
        outputStream.close()
        entry.getName match {
          case name if name == trainImagesFilename =>
            val shape = Shape(numTrain, imageChannels, imageHeight, imageWidth)
            val tensor = Tensor.fromBuffer[UByte](shape, entry.getSize, byteBuffer).transpose(Tensor(0, 3, 2, 1))
            dataset = dataset.copy(trainImages = tensor)
          case name if name == trainLabelsFilename =>
            val tensor = Tensor.fromBuffer[UByte](Shape(numTrain), entry.getSize, byteBuffer) - Tensor.ones[UByte](Shape())
            dataset = dataset.copy(trainLabels = tensor)
          case name if name == testImagesFilename =>
            val shape = Shape(numTest, imageChannels, imageHeight, imageWidth)
            val tensor = Tensor.fromBuffer[UByte](shape, entry.getSize, byteBuffer).transpose(Tensor(0, 3, 2, 1))
            dataset = dataset.copy(testImages = tensor)
          case name if name == testLabelsFilename =>
            val tensor = Tensor.fromBuffer[UByte](Shape(numTest), entry.getSize, byteBuffer) - Tensor.ones[UByte](Shape())
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
    trainImages: Tensor[UByte],
    trainLabels: Tensor[UByte],
    testImages: Tensor[UByte],
    testLabels: Tensor[UByte],
    unlabeledImages: Tensor[UByte]
) {
  def splitRandomly(trainPortion: Float, seed: Option[Long] = None): STL10Dataset = {
    if (trainPortion == 1.0f) {
      this
    } else {
      val allImages = tfi.concatenate(Seq(trainImages, testImages), axis = 0)
      val allLabels = tfi.concatenate(Seq(trainLabels, testLabels), axis = 0)
      val split = UniformSplit(allLabels.shape(0), seed)
      val (trainIndices, testIndices) = split(trainPortion)
      copy(
        trainImages = allImages.gather[Int](trainIndices),
        trainLabels = allLabels.gather[Int](trainIndices),
        testImages = allImages.gather[Int](testIndices),
        testLabels = allLabels.gather[Int](testIndices))
    }
  }
}
