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

import java.nio.file.Path

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

object FashionMNISTLoader extends Loader {
  override protected val logger = Logger(LoggerFactory.getLogger("Fashion MNIST Data Loader"))

  private[this] val DEFAULT_URL           = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
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
    val trainImages = MNISTLoader.extractImages(trainImagesPath, bufferSize)
    val trainLabels = MNISTLoader.extractLabels(trainLabelsPath, bufferSize)
    val testImages = MNISTLoader.extractImages(testImagesPath, bufferSize)
    val testLabels = MNISTLoader.extractLabels(testLabelsPath, bufferSize)

    MNISTDataSet(trainImages, trainLabels, testImages, testLabels)
  }
}

case class FashionMNISTDataSet(trainImages: Tensor, trainLabels: Tensor, testImages: Tensor, testLabels: Tensor)
