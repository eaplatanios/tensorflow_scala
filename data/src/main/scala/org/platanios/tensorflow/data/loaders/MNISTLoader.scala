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

import org.platanios.tensorflow.api._

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
object MNISTLoader extends Loader {
  override protected val logger = Logger(LoggerFactory.getLogger("MNIST Data Loader"))

  private[this] val DEFAULT_URL           = "http://yann.lecun.com/exdb/mnist/"
  private[this] val TRAIN_IMAGES_FILENAME = "train-images-idx3-ubyte.gz"
  private[this] val TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
  private[this] val TEST_IMAGES_FILENAME  = "t10k-images-idx3-ubyte.gz"
  private[this] val TEST_LABELS_FILENAME  = "t10k-labels-idx1-ubyte.gz"

  def load(path: Path, url: String = DEFAULT_URL, trainImagesFilename: String = TRAIN_IMAGES_FILENAME,
      trainLabelsFilename: String = TRAIN_LABELS_FILENAME, testImagesFilename: String = TEST_IMAGES_FILENAME,
      testLabelsFilename: String = TEST_LABELS_FILENAME, bufferSize: Int = 8192): MNISTDataSet = {
    val trainImagesPath = path.resolve(trainImagesFilename)
    val trainLabelsPath = path.resolve(trainLabelsFilename)
    val testImagesPath = path.resolve(testImagesFilename)
    val testLabelsPath = path.resolve(testLabelsFilename)

    // Download the data, if necessary.
    maybeDownload(trainImagesPath, url + trainImagesFilename, bufferSize)
    maybeDownload(trainLabelsPath, url + trainLabelsFilename, bufferSize)
    maybeDownload(testImagesPath, url + testImagesFilename, bufferSize)
    maybeDownload(testLabelsPath, url + testLabelsFilename, bufferSize)

    // Load the data.
    val trainImages = extractImages(trainImagesPath, bufferSize)
    val trainLabels = extractLabels(trainLabelsPath, bufferSize)
    val testImages = extractImages(testImagesPath, bufferSize)
    val testLabels = extractLabels(testLabelsPath, bufferSize)

    MNISTDataSet(trainImages, trainLabels, testImages, testLabels)
  }

  private[this] def extractImages(path: Path, bufferSize: Int = 8192): tf.Tensor = {
    logger.info(s"Extracting images from file '$path'.")
    val inputStream = new GZIPInputStream(Files.newInputStream(path))
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    outputStream.close()
    inputStream.close()
    val magicNumber = (byteBuffer.getInt & 0xffffffffL).toInt
    if (magicNumber != 2051)
      throw new IllegalStateException(s"Invalid magic number '$magicNumber' in MNIST image file '$path'.")
    val numberOfImages = (byteBuffer.getInt & 0xffffffffL).toInt
    val numberOfRows = (byteBuffer.getInt & 0xffffffffL).toInt
    val numberOfColumns = (byteBuffer.getInt & 0xffffffffL).toInt
    tf.Tensor.fromBuffer(tf.UINT8, tf.shape(numberOfImages, numberOfRows, numberOfColumns), byteBuffer)
  }

  private[this] def extractLabels(path: Path, bufferSize: Int = 8192): tf.Tensor = {
    logger.info(s"Extracting labels from file '$path'.")
    val inputStream = new GZIPInputStream(Files.newInputStream(path))
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val byteBuffer = ByteBuffer.wrap(outputStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    outputStream.close()
    inputStream.close()
    val magicNumber = (byteBuffer.getInt & 0xffffffffL).toInt
    if (magicNumber != 2049)
      throw new IllegalStateException(s"Invalid magic number '$magicNumber' in MNIST labels file '$path'.")
    val numberOfLabels = (byteBuffer.getInt & 0xffffffffL).toInt
    tf.Tensor.fromBuffer(tf.UINT8, tf.shape(numberOfLabels), byteBuffer)
  }
}

case class MNISTDataSet(trainImages: tf.Tensor, trainLabels: tf.Tensor, testImages: tf.Tensor, testLabels: tf.Tensor)
