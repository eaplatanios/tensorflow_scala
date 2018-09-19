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

package org.platanios.tensorflow.data.text

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.Loader

import com.typesafe.scalalogging.Logger
import org.apache.commons.compress.archivers.tar._
import org.slf4j.LoggerFactory

import java.io.{BufferedReader, InputStream, InputStreamReader}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

import scala.collection.mutable

/** Loader for the PTB raw data from
  * [Tomas Mikolov's website](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).
  *
  * @author Emmanouil Antonios Platanios
  */
object PTBLoader extends Loader {
  override protected val logger = Logger(LoggerFactory.getLogger("PTB Data Loader"))

  val url               : String = "http://www.fit.vutbr.cz/~imikolov/rnnlm/"
  val compressedFilename: String = "simple-examples.tgz"
  val dataPath          : String = "./simple-examples/data/"
  val trainFilename     : String = "ptb.train.txt"
  val validFilename     : String = "ptb.valid.txt"
  val testFilename      : String = "ptb.test.txt"

  def load(path: Path, bufferSize: Int = 8192): PTBDataset = {
    // Download the data, if necessary.
    maybeDownload(path.resolve(compressedFilename), url + compressedFilename, bufferSize)

    // Load the data.
    val dataset = extractData(path.resolve(compressedFilename), bufferSize)
    logger.info(s"Finished loading the PTB dataset.")
    dataset
  }

  def tokensToBatchedTFDataset(
      tokens: Seq[Int],
      batchSize: Int,
      numSteps: Int,
      name: String
  ): tf.data.Dataset[(Tensor[Int], Tensor[Int]), (Output, Output), (INT32, INT32), (Shape, Shape)] = {
    tf.createWithNameScope(name) {
      tf.data.fromGenerator[(Tensor[Int], Tensor[Int]), (Output, Output), (INT32, INT32), (Shape, Shape)](
        () => tokensToBatchIterable(tokens, batchSize, numSteps),
        (INT32, INT32),
        (Shape(batchSize, numSteps), Shape(batchSize, numSteps)))
    }
  }

  def tokensToBatchIterable(
      tokens: Seq[Int],
      batchSize: Int,
      numSteps: Int
  ): Iterable[(Tensor[Int], Tensor[Int])] = {
    new Iterable[(Tensor[Int], Tensor[Int])] {
      override def iterator: Iterator[(Tensor[Int], Tensor[Int])] = new Iterator[(Tensor[Int], Tensor[Int])] {
        private val tokensTensor = Tensor(tokens.head, tokens.tail: _*)
        private val numTokens    = tokens.size
        private val batchLength  = numTokens / batchSize
        private val data         = tokensTensor(0 :: batchSize * batchLength).reshape(Shape(batchSize, batchLength))
        private val numEpochs    = (batchLength - 1) / numSteps

        if (numEpochs <= 0)
          throw tf.InvalidArgumentException("The epoch size is 0. Decrease 'batchSize' or 'numSteps'.")

        private var currentEpoch: Int = 0

        override def hasNext: Boolean = currentEpoch < numEpochs

        override def next(): (Tensor[Int], Tensor[Int]) = {
          val batch = (
              data(::, (currentEpoch * numSteps) :: ((currentEpoch + 1) * numSteps)),
              data(::, (currentEpoch * numSteps + 1) :: ((currentEpoch + 1) * numSteps + 1)))
          currentEpoch += 1
          batch
        }
      }
    }
  }

  private[this] def extractData(path: Path, bufferSize: Int = 8192): PTBDataset = {
    logger.info(s"Extracting data from file '$path'.")
    val inputStream = new TarArchiveInputStream(new GZIPInputStream(Files.newInputStream(path)))
    var trainTokens: Seq[String] = null
    var validTokens: Seq[String] = null
    var testTokens: Seq[String] = null
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      entry.getName match {
        case f if f == dataPath + trainFilename => trainTokens = readTokens(inputStream)
        case f if f == dataPath + validFilename => validTokens = readTokens(inputStream)
        case f if f == dataPath + testFilename => testTokens = readTokens(inputStream)
        case _ => ()
      }
      entry = inputStream.getNextTarEntry
    }
    inputStream.close()
    val vocabulary = buildVocabulary(trainTokens ++ validTokens ++ testTokens)
    val trainIds = trainTokens.filter(vocabulary.contains).map(vocabulary)
    val validIds = validTokens.filter(vocabulary.contains).map(vocabulary)
    val testIds = testTokens.filter(vocabulary.contains).map(vocabulary)
    PTBDataset(trainIds, validIds, testIds, for ((k, v) <- vocabulary) yield (v, k))
  }

  private[this] def readTokens(inputStream: InputStream): Seq[String] = {
    val reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
    val tokens = mutable.ListBuffer.empty[String]
    var line = reader.readLine()
    while (line != null) {
      tokens ++= line.replace("\n", "<eos>").split("\\s+").toSeq
      line = reader.readLine()
    }
    tokens
  }

  private[this] def buildVocabulary(tokens: Seq[String]): Map[String, Int] = {
    val counts = tokens.groupBy(identity).mapValues(_.size)
    val sorted = counts.toSeq.sortBy(-_._2)
    sorted.map(_._1).zipWithIndex.toMap
  }

  def main(args: Array[String]): Unit = {
    val dataSet = PTBLoader.load(Paths.get(args(0)))
    println(s"Number of tokens: ${dataSet.vocabulary.size}")
    println(s"Number of train tokens: ${dataSet.train.size}")
    println(s"Number of validation tokens: ${dataSet.validation.size}")
    println(s"Number of test tokens: ${dataSet.test.size}")
  }
}

case class PTBDataset(train: Seq[Int], validation: Seq[Int], test: Seq[Int], vocabulary: Map[Int, String])
