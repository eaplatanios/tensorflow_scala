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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.utilities.UniformSplit

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.zip.ZipInputStream

/** Loader for datasets obtained from the
  * [[http://manikvarma.org/downloads/XC/XMLRepository.html Extreme Classification Repository]].
  *
  * @author Emmanouil Antonios Platanios
  */
object XCLoader extends Loader {
  sealed trait DatasetType {
    val name: String
    val url : String

    def compressedFilename: String = s"$name.zip"

    val numFeatures       : Int
    val numLabels         : Int
    val numTrain          : Int
    val numTest           : Int
    val avgSamplesPerLabel: Float
    val avgLabelsPerSample: Float
  }

  sealed trait SmallDatasetType extends DatasetType {
    def dataFilename: String = s"${name}_data.txt"
    def trainSplitsFilename: String = s"${name.toLowerCase()}_trSplit.txt"
    def testSplitsFilename: String = s"${name.toLowerCase()}_tstSplit.txt"

    def loadDense(path: Path, bufferSize: Int = 8192): SmallDataset[Tensor] = {
      loadSmallDense(path, this, bufferSize)
    }

    def loadSparse(path: Path, bufferSize: Int = 8192): SmallDataset[SparseTensor] = {
      loadSmallSparse(path, this, bufferSize)
    }

    // TODO: [DATA] Wikipedia-LSHTC: A=0.5,  B=0.4
    // TODO: [DATA] Amazon:          A=0.6,  B=2.6

    private[data] val labelsPropensityA: Float = 0.55f
    private[data] val labelsPropensityB: Float = 1.5f
  }

  sealed trait LargeDatasetType extends DatasetType {
    def trainDataFilename: String = s"${name.toLowerCase()}_train.txt"
    def testDataFilename: String = s"${name.toLowerCase()}_test.txt"

    def loadDense(path: Path, bufferSize: Int = 8192): LargeDataset[Tensor] = {
      loadLargeDense(path, this, bufferSize)
    }

    def loadSparse(path: Path, bufferSize: Int = 8192): LargeDataset[SparseTensor] = {
      loadLargeSparse(path, this, bufferSize)
    }

    // TODO: [DATA] Add support for computing the label propensity scores.
  }

  case object BIBTEX extends SmallDatasetType {
    override val name: String = "Bibtex"
    override val url : String = "https://drive.google.com/uc?id=0B3lPMIHmG6vGcy1xM2pJZ09MMGM&export=download"

    override val numFeatures       : Int   = 1836
    override val numLabels         : Int   = 159
    override val numTrain          : Int   = 4880
    override val numTest           : Int   = 2515
    override val avgSamplesPerLabel: Float = 111.71f
    override val avgLabelsPerSample: Float = 2.40f
  }

  case object DELICIOUS extends SmallDatasetType {
    override val name: String = "Delicious"
    override val url : String = "https://drive.google.com/uc?id=0B3lPMIHmG6vGdG1jZ19VS2NWRVU&export=download"

    override val numFeatures       : Int   = 500
    override val numLabels         : Int   = 983
    override val numTrain          : Int   = 12920
    override val numTest           : Int   = 3185
    override val avgSamplesPerLabel: Float = 311.61f
    override val avgLabelsPerSample: Float = 19.03f
  }

  case object MEDIAMILL extends SmallDatasetType {
    override val name: String = "Mediamill"
    override val url : String = "https://drive.google.com/uc?id=0B3lPMIHmG6vGY3B4TXRmZnZBTkk&export=download"

    override val numFeatures       : Int   = 120
    override val numLabels         : Int   = 101
    override val numTrain          : Int   = 30993
    override val numTest           : Int   = 12914
    override val avgSamplesPerLabel: Float = 1902.15f
    override val avgLabelsPerSample: Float = 4.38f
  }

  case object EURLEX extends LargeDatasetType {
    override val name: String = "Eurlex"
    override val url : String = "https://drive.google.com/uc?id=0B3lPMIHmG6vGU0VTR1pCejFpWjg&export=download"

    override val numFeatures       : Int   = 5000
    override val numLabels         : Int   = 3993
    override val numTrain          : Int   = 15539
    override val numTest           : Int   = 3809
    override val avgSamplesPerLabel: Float = 25.73f
    override val avgLabelsPerSample: Float = 5.31f
  }

  // TODO: [DATA] RCV1-x (an issue related to Google Drive will need to be resolved first).

  case class Data[TL[A] <: TensorLike[A]](features: TL[Float], labels: TL[Boolean])

  case class SplitData[TL[A] <: TensorLike[A]](trainData: Data[TL], testData: Data[TL]) {
    def splitRandomly(trainPortion: Float, seed: Option[Long] = None): SplitData[Tensor] = {
      if (trainPortion == 1.0f) {
        SplitData(
          trainData = Data(features = trainData.features.toTensor, labels = trainData.labels.toTensor),
          testData = Data(features = testData.features.toTensor, labels = testData.labels.toTensor))
      } else {
        val allFeatures = tfi.concatenate(Seq(trainData.features.toTensor, testData.features.toTensor), axis = 0)
        val allLabels = tfi.concatenate(Seq(trainData.labels.toTensor, testData.labels.toTensor), axis = 0)
        val split = UniformSplit(allLabels.shape(0), seed)
        val (trainIndices, testIndices) = split(trainPortion)
        SplitData(
          trainData = Data(
            features = allFeatures.gather[Int](trainIndices),
            labels = allLabels.gather[Int](trainIndices)),
          testData = Data(
            features = allFeatures.gather[Int](testIndices),
            labels = allLabels.gather[Int](testIndices)))
      }
    }
  }

  case class Split(trainIndices: Seq[Int], testIndices: Seq[Int])

  case class SmallDataset[TL[A] <: TensorLike[A]](
      datasetType: SmallDatasetType,
      data: Data[TL],
      splits: Seq[Split]
  ) {
    /**
      * '''NOTE:''' This method will convert the data to dense format, even if it was loaded in sparse format.
      *
      * @param  split Split index.
      * @return Input data
      */
    def split(split: Int = 0): XCLoader.SplitData[Tensor] = {
      val trainData = XCLoader.Data(
        features = data.features.toTensor.gather[Int](splits(split).trainIndices),
        labels = data.labels.toTensor.gather[Int](splits(split).trainIndices))
      val testData = XCLoader.Data(
        features = data.features.toTensor.gather[Int](splits(split).testIndices),
        labels = data.labels.toTensor.gather[Int](splits(split).testIndices))
      XCLoader.SplitData(trainData, testData)
    }
  }

  case class LargeDataset[TL[A] <: TensorLike[A]](datasetType: LargeDatasetType, data: SplitData[TL])

  override protected val logger = Logger(LoggerFactory.getLogger("XC Data Loader"))

  def loadSmallDense(
      path: Path,
      datasetType: SmallDatasetType,
      bufferSize: Int = 8192
  ): SmallDataset[Tensor] = {
    val compressedFile = loadCommon(path, datasetType, bufferSize)
    val dataset = extractSmallScaleDataset(compressedFile, datasetType, sparseDataToDense, bufferSize)
    logger.info(s"Finished loading the XC ${datasetType.name} dataset (in dense format).")
    dataset
  }

  def loadSmallSparse(
      path: Path,
      datasetType: SmallDatasetType,
      bufferSize: Int = 8192
  ): SmallDataset[SparseTensor] = {
    val compressedFile = loadCommon(path, datasetType, bufferSize)
    val dataset = extractSmallScaleDataset(compressedFile, datasetType, identity[Data[SparseTensor]], bufferSize)
    logger.info(s"Finished loading the XC ${datasetType.name} dataset (in sparse format).")
    dataset
  }

  def loadLargeDense(
      path: Path,
      datasetType: LargeDatasetType,
      bufferSize: Int = 8192
  ): LargeDataset[Tensor] = {
    val compressedFile = loadCommon(path, datasetType, bufferSize)
    val dataset = extractLargeScaleDataset(compressedFile, datasetType, sparseDataToDense, bufferSize)
    logger.info(s"Finished loading the XC ${datasetType.name} dataset (in dense format).")
    dataset
  }

  def loadLargeSparse(
      path: Path,
      datasetType: LargeDatasetType,
      bufferSize: Int = 8192
  ): LargeDataset[SparseTensor] = {
    val compressedFile = loadCommon(path, datasetType, bufferSize)
    val dataset = extractLargeScaleDataset(compressedFile, datasetType, identity[Data[SparseTensor]], bufferSize)
    logger.info(s"Finished loading the XC ${datasetType.name} dataset (in sparse format).")
    dataset
  }

  def labelPropensityScores(dataset: SmallDataset[Tensor]): Tensor[Float] = {
    val numSamples = dataset.data.labels.shape(0)
    val labelCounts = dataset.data.labels.castTo[Float].sum[Int](axes = Seq(0))
    val a = dataset.datasetType.labelsPropensityA
    val b = dataset.datasetType.labelsPropensityB
    val c = (math.log(numSamples) - 1) * math.pow(b + 1, a)
    ((labelCounts + b).pow(-a) * c.toFloat) + 1.0f
  }

  // TODO: [SPARSE] Add support for computing the propensity scores from sparse datasets.

  private def loadCommon(path: Path, datasetType: DatasetType, bufferSize: Int = 8192): Path = {
    val url = datasetType.url
    val compressedFilename = datasetType.compressedFilename
    val workingDir = path.resolve(datasetType.name.toLowerCase())
    maybeDownload(workingDir.resolve(compressedFilename), url, bufferSize)
    workingDir.resolve(datasetType.compressedFilename)
  }

  private def sparseDataToDense(data: Data[SparseTensor]): Data[Tensor] = {
    Data(data.features.toTensor, data.labels.toTensor)
  }

  private def extractSmallScaleDataset[TL[A] <: TensorLike[A]](
      path: Path,
      datasetType: SmallDatasetType,
      dataConverter: Data[SparseTensor] => Data[TL],
      bufferSize: Int = 8192
  ): SmallDataset[TL] = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new ZipInputStream(Files.newInputStream(path))
    var dataset = SmallDataset[TL](datasetType, null, null)
    var entry = inputStream.getNextEntry
    while (entry != null) {
      if (entry.getName.endsWith(datasetType.dataFilename)) {
        val data = readData(inputStream, datasetType, bufferSize)
        dataset = dataset.copy(data = dataConverter(data))
      } else if (entry.getName.endsWith(datasetType.trainSplitsFilename)) {
        val splits = readSplits(inputStream, bufferSize)
        val datasetSplits = {
          if (dataset.splits == null)
            splits.toSeq.map(s => Split(trainIndices = s, testIndices = null))
          else
            dataset.splits.zip(splits).map(p => p._1.copy(trainIndices = p._2))
        }
        dataset = dataset.copy(splits = datasetSplits)
      } else if (entry.getName.endsWith(datasetType.testSplitsFilename)) {
        val splits = readSplits(inputStream, bufferSize)
        val datasetSplits = {
          if (dataset.splits == null)
            splits.toSeq.map(s => Split(trainIndices = null, testIndices = s))
          else
            dataset.splits.zip(splits).map(p => p._1.copy(testIndices = p._2))
        }
        dataset = dataset.copy(splits = datasetSplits)
      }
      entry = inputStream.getNextEntry
    }
    inputStream.close()
    dataset
  }

  private def extractLargeScaleDataset[TL[A] <: TensorLike[A]](
      path: Path,
      datasetType: LargeDatasetType,
      dataConverter: Data[SparseTensor] => Data[TL],
      bufferSize: Int = 8192
  ): LargeDataset[TL] = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new ZipInputStream(Files.newInputStream(path))
    var dataset = LargeDataset[TL](datasetType, null)
    var entry = inputStream.getNextEntry
    while (entry != null) {
      if (entry.getName.endsWith(datasetType.trainDataFilename)) {
        val data = dataConverter(readData(inputStream, datasetType, bufferSize))
        val existingData = dataset.data
        val updatedData = {
          if (existingData == null)
            SplitData(trainData = data, testData = null)
          else
            existingData.copy(trainData = data)
        }
        dataset = dataset.copy(data = updatedData)
      } else if (entry.getName.endsWith(datasetType.testDataFilename)) {
        val data = dataConverter(readData(inputStream, datasetType, bufferSize))
        val existingData = dataset.data
        val updatedData = {
          if (existingData == null)
            SplitData(trainData = null, testData = data)
          else
            existingData.copy(testData = data)
        }
        dataset = dataset.copy(data = updatedData)
      }
      entry = inputStream.getNextEntry
    }
    inputStream.close()
    dataset
  }

  @throws[IllegalArgumentException]
  private def readData(
      inputStream: ZipInputStream,
      datasetType: DatasetType,
      bufferSize: Int = 8192
  ): Data[SparseTensor] = {
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    val lines = outputStream.toString(StandardCharsets.UTF_8.name()).split("\\r?\\n")
    val firstLineParts = lines.head.split(' ')
    val numSamples = firstLineParts(0).toInt
    val numFeatures = firstLineParts(1).toInt
    val numLabels = firstLineParts(2).toInt
    assert(numFeatures == datasetType.numFeatures, "Invalid dataset number of features read.")
    assert(numLabels == datasetType.numLabels, "Invalid dataset number of labels read.")
    var labelsCount = 0
    var featuresCount = 0
    val labelIndicesStream = new ByteArrayOutputStream()
    val labelValuesStream = new ByteArrayOutputStream()
    val featureIndicesStream = new ByteArrayOutputStream()
    val featureValuesStream = new ByteArrayOutputStream()
    lines.tail.filter(_.nonEmpty).zipWithIndex.foreach(l => {
      val sampleIndex = longToBytes(l._2.toLong)
      val lineParts = l._1.split(' ')
      lineParts(0).split(',').filter(_.nonEmpty).foreach(i => {
        labelIndicesStream.write(sampleIndex)
        labelIndicesStream.write(longToBytes(i.toLong))
        labelValuesStream.write(1)
        labelsCount += 1
      })
      lineParts.tail.foreach(feature => {
        val featureParts = feature.split(':')
        featureIndicesStream.write(sampleIndex)
        featureIndicesStream.write(longToBytes(featureParts(0).toLong))
        featureValuesStream.write(floatToBytes(featureParts(1).toFloat))
        featuresCount += 1
      })
    })

    val labelIndicesBuffer = ByteBuffer.wrap(labelIndicesStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    labelIndicesStream.close()
    val labelIndices = Tensor.fromBuffer[Long](Shape(labelsCount, 2), labelsCount * 2 * 8, labelIndicesBuffer)

    val labelValuesBuffer = ByteBuffer.wrap(labelValuesStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    labelValuesStream.close()
    val labelValues = Tensor.fromBuffer[Boolean](Shape(labelsCount), labelsCount, labelValuesBuffer)

    val featureIndicesBuffer = ByteBuffer.wrap(featureIndicesStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    featureIndicesStream.close()
    val featureIndices = Tensor.fromBuffer[Long](Shape(featuresCount, 2), featuresCount * 2 * 8, featureIndicesBuffer)

    val featureValuesBuffer = ByteBuffer.wrap(featureValuesStream.toByteArray).order(ByteOrder.BIG_ENDIAN)
    featureValuesStream.close()
    val featureValues = Tensor.fromBuffer[Float](Shape(featuresCount), featuresCount * 4, featureValuesBuffer)

    val features = SparseTensor(featureIndices, featureValues, Shape(numSamples, numFeatures))
    val labels = SparseTensor(labelIndices, labelValues, Shape(numSamples, numLabels))

    Data(features, labels)
  }

  private def readSplits(
      inputStream: ZipInputStream,
      bufferSize: Int = 8192
  ): Array[Array[Int]] = {
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Array[Byte](bufferSize)
    Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
    outputStream.toString(StandardCharsets.UTF_8.name())
        .split("\\r?\\n")
        .filter(_.nonEmpty)
        .map(_.split(' ').map(_.toInt - 1))
        .transpose
  }

  private[this] def longToBytes(value: Long): Array[Byte] = {
    ByteBuffer.allocate(java.lang.Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array()
  }

  private[this] def floatToBytes(value: Float): Array[Byte] = {
    ByteBuffer.allocate(java.lang.Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array()
  }
}
