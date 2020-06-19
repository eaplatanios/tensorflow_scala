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
import org.slf4j.LoggerFactory

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

import scala.collection.compat.immutable.LazyList

/**
  * @author Emmanouil Antonios Platanios
  */
object MNISTLoader extends Loader {
  sealed trait DatasetType {
    val name               : String
    val url                : String
    val trainImagesFilename: String
    val trainLabelsFilename: String
    val testImagesFilename : String
    val testLabelsFilename : String
  }

  case object MNIST extends DatasetType {
    override val name               : String = "MNIST"
    override val url                : String = "http://yann.lecun.com/exdb/mnist/"
    override val trainImagesFilename: String = "train-images-idx3-ubyte.gz"
    override val trainLabelsFilename: String = "train-labels-idx1-ubyte.gz"
    override val testImagesFilename : String = "t10k-images-idx3-ubyte.gz"
    override val testLabelsFilename : String = "t10k-labels-idx1-ubyte.gz"
  }

  case object FASHION_MNIST extends DatasetType {
    override val name               : String = "FASHION-MNIST"
    override val url                : String = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    override val trainImagesFilename: String = "train-images-idx3-ubyte.gz"
    override val trainLabelsFilename: String = "train-labels-idx1-ubyte.gz"
    override val testImagesFilename : String = "t10k-images-idx3-ubyte.gz"
    override val testLabelsFilename : String = "t10k-labels-idx1-ubyte.gz"
  }

  override protected val logger = Logger(LoggerFactory.getLogger("MNIST Data Loader"))

  def load(path: Path, datasetType: DatasetType = MNIST, bufferSize: Int = 8192): MNISTDataset = {
    val trainImagesPath = path.resolve(datasetType.trainImagesFilename)
    val trainLabelsPath = path.resolve(datasetType.trainLabelsFilename)
    val testImagesPath = path.resolve(datasetType.testImagesFilename)
    val testLabelsPath = path.resolve(datasetType.testLabelsFilename)

    // Download the data, if necessary.
    maybeDownload(trainImagesPath, datasetType.url + datasetType.trainImagesFilename, bufferSize)
    maybeDownload(trainLabelsPath, datasetType.url + datasetType.trainLabelsFilename, bufferSize)
    maybeDownload(testImagesPath, datasetType.url + datasetType.testImagesFilename, bufferSize)
    maybeDownload(testLabelsPath, datasetType.url + datasetType.testLabelsFilename, bufferSize)

    // Load the data.
    val trainImages = extractImages(trainImagesPath, bufferSize)
    val trainLabels = extractLabels(trainLabelsPath, bufferSize)
    val testImages = extractImages(testImagesPath, bufferSize)
    val testLabels = extractLabels(testLabelsPath, bufferSize)

    logger.info(s"Finished loading the ${datasetType.name} dataset.")

    MNISTDataset(datasetType, trainImages, trainLabels, testImages, testLabels)
  }

  private def extractImages(path: Path, bufferSize: Int = 8192): Tensor[UByte] = {
    logger.info(s"Extracting images from file '$path'.")
    val inputStream = new GZIPInputStream(Files.newInputStream(path))
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    LazyList.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    outputStream.close()
    inputStream.close()
    val magicNumber = (byteBuffer.getInt & 0xffffffffL).toInt
    if (magicNumber != 2051)
      throw new IllegalStateException(s"Invalid magic number '$magicNumber' in MNIST image file '$path'.")
    val numberOfImages = (byteBuffer.getInt & 0xffffffffL).toInt
    val numberOfRows = (byteBuffer.getInt & 0xffffffffL).toInt
    val numberOfColumns = (byteBuffer.getInt & 0xffffffffL).toInt
    val numBytes = byteBuffer.limit() - 16
    val tensor = Tensor.fromBuffer[UByte](Shape(numberOfImages, numberOfRows, numberOfColumns), numBytes, byteBuffer)
    outputStream.close()
    inputStream.close()
    tensor
  }

  private def extractLabels(path: Path, bufferSize: Int = 8192): Tensor[UByte] = {
    logger.info(s"Extracting labels from file '$path'.")
    val inputStream = new GZIPInputStream(Files.newInputStream(path))
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    LazyList.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    outputStream.close()
    inputStream.close()
    val magicNumber = (byteBuffer.getInt & 0xffffffffL).toInt
    if (magicNumber != 2049)
      throw new IllegalStateException(s"Invalid magic number '$magicNumber' in MNIST labels file '$path'.")
    val numberOfLabels = (byteBuffer.getInt & 0xffffffffL).toInt
    val numBytes = byteBuffer.limit() - 8
    val tensor = Tensor.fromBuffer[UByte](Shape(numberOfLabels), numBytes, byteBuffer)
    outputStream.close()
    inputStream.close()
    tensor
  }
}

case class MNISTDataset(
    datasetType: MNISTLoader.DatasetType,
    trainImages: Tensor[UByte],
    trainLabels: Tensor[UByte],
    testImages: Tensor[UByte],
    testLabels: Tensor[UByte]
) {
  def splitRandomly(trainPortion: Float, seed: Option[Long] = None): MNISTDataset = {
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
