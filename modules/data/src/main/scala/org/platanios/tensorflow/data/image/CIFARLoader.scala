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
import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.slf4j.LoggerFactory

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

import scala.collection.compat.immutable.LazyList
import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
object CIFARLoader extends Loader {
  sealed trait DatasetType {
    val name              : String
    val url               : String
    val compressedFilename: String
    val trainFilenames    : Seq[String]
    val testFilename      : String

    private[image] val entryByteSize: Int
  }

  case object CIFAR_10 extends DatasetType {
    override val name              : String      = "CIFAR-10"
    override val url               : String      = "http://www.cs.toronto.edu/~kriz/"
    override val compressedFilename: String      = "cifar-10-binary.tar.gz"
    override val trainFilenames    : Seq[String] = (1 to 5).map(i => s"data_batch_$i.bin")
    override val testFilename      : String      = "test_batch.bin"

    private[image] override val entryByteSize: Int = 3073
  }

  case object CIFAR_100 extends DatasetType {
    override val name              : String      = "CIFAR-100"
    override val url               : String      = "http://www.cs.toronto.edu/~kriz/"
    override val compressedFilename: String      = "cifar-100-binary.tar.gz"
    override val trainFilenames    : Seq[String] = Seq("train.bin")
    override val testFilename      : String      = "test.bin"

    private[image] override val entryByteSize: Int = 3074
  }

  override protected val logger = Logger(LoggerFactory.getLogger("CIFAR Data Loader"))

  def load(path: Path, datasetType: DatasetType = CIFAR_10, bufferSize: Int = 8192): CIFARDataset = {
    val url = datasetType.url
    val compressedFilename = datasetType.compressedFilename

    // Download the data, if necessary.
    maybeDownload(path.resolve(compressedFilename), url + compressedFilename, bufferSize)

    // Load the data.
    val dataset = extractFiles(path.resolve(compressedFilename), datasetType, bufferSize)
    logger.info(s"Finished loading the ${datasetType.name} dataset.")
    dataset
  }

  private def extractFiles(
      path: Path,
      datasetType: DatasetType = CIFAR_10,
      bufferSize: Int = 8192
  ): CIFARDataset = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new TarArchiveInputStream(new GZIPInputStream(Files.newInputStream(path)))
    var dataset = CIFARDataset(datasetType, null, null, null, null)
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      if (datasetType.trainFilenames.exists(entry.getName.endsWith(_))) {
        val (images, labels) = readImagesAndLabels(inputStream, entry, datasetType, bufferSize)
        val trainImages = {
          if (dataset.trainImages == null)
            images
          else
            tfi.concatenate(Seq(dataset.trainImages, images), axis = 0)
        }
        val trainLabels = {
          if (dataset.trainLabels == null)
            labels
          else
            tfi.concatenate(Seq(dataset.trainLabels, labels), axis = 0)
        }
        dataset = dataset.copy(trainImages = trainImages, trainLabels = trainLabels)
      } else if (entry.getName.endsWith(datasetType.testFilename)) {
        val (images, labels) = readImagesAndLabels(inputStream, entry, datasetType, bufferSize)
        dataset = dataset.copy(testImages = images, testLabels = labels)
      }
      entry = inputStream.getNextTarEntry
    }
    inputStream.close()
    dataset
  }

  private def readImagesAndLabels(
      inputStream: TarArchiveInputStream,
      entry: TarArchiveEntry,
      datasetType: DatasetType = CIFAR_10,
      bufferSize: Int = 8192
  ): (Tensor[UByte], Tensor[UByte]) = {
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    LazyList.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    outputStream.close()
    val numSamples = entry.getSize.toInt / datasetType.entryByteSize
    val combinedShape = Shape(numSamples, datasetType.entryByteSize)
    val combined = Tensor.fromBuffer[UByte](combinedShape, entry.getSize.toInt, byteBuffer)
    datasetType match {
      case CIFAR_10 => (combined(::, 1 ::).reshape(Shape(-1, 32, 32, 3)), combined(::, 0))
      case CIFAR_100 => (combined(::, 2 ::).reshape(Shape(-1, 32, 32, 3)), combined(::, 0 :: 2))
    }
  }
}

case class CIFARDataset(
    datasetType: CIFARLoader.DatasetType,
    trainImages: Tensor[UByte],
    trainLabels: Tensor[UByte],
    testImages: Tensor[UByte],
    testLabels: Tensor[UByte]
) {
  def splitRandomly(trainPortion: Float, seed: Option[Long] = None): CIFARDataset = {
    if (trainPortion == 1.0f) {
      this
    } else {
      val allImages = tfi.concatenate(Seq(trainImages, testImages), axis = 0)
      val allLabels = tfi.concatenate(Seq(trainLabels, testLabels), axis = 0)
      val split = UniformSplit(allLabels.shape(0), seed)
      val (trainIndices, testIndices) = split(trainPortion)
      copy(
        trainImages = allImages.gather[Int](trainIndices, axis = 0),
        trainLabels = allLabels.gather[Int](trainIndices, axis = 0),
        testImages = allImages.gather[Int](testIndices, axis = 0),
        testLabels = allLabels.gather[Int](testIndices, axis = 0))
    }
  }
}
