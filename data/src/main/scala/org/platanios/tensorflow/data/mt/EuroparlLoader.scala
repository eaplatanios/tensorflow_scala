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

package org.platanios.tensorflow.data.mt

import org.platanios.tensorflow.data.Loader
import org.platanios.tensorflow.data.utilities.{CompressedFiles, MosesDecoder}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}

import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
object EuroparlLoader extends Loader {
  val url: String = "http://www.statmt.org/europarl/v7"

  override protected val logger = Logger(LoggerFactory.getLogger("Europarl Data Loader"))

  sealed trait DatasetType {
    val srcLanguage: String
    val tgtLanguage: String

    def name: String = s"$srcLanguage-$tgtLanguage"

    def srcVocab: String = s"$srcLanguage.vocab"
    def tgtVocab: String = s"$tgtLanguage.vocab"

    def srcCorpus: String = s"europarl-v7.$srcLanguage-$tgtLanguage.$srcLanguage"
    def tgtCorpus: String = s"europarl-v7.$srcLanguage-$tgtLanguage.$tgtLanguage"

    def srcTokenizedCorpus: String = s"europarl-v7.$srcLanguage-$tgtLanguage.tok.$srcLanguage"
    def tgtTokenizedCorpus: String = s"europarl-v7.$srcLanguage-$tgtLanguage.tok.$tgtLanguage"
  }

  case object BulgarianEnglish extends DatasetType {
    override val srcLanguage: String = "bg"
    override val tgtLanguage: String = "en"
  }

  case object CzechEnglish extends DatasetType {
    override val srcLanguage: String = "cs"
    override val tgtLanguage: String = "en"
  }

  case object DanishEnglish extends DatasetType {
    override val srcLanguage: String = "da"
    override val tgtLanguage: String = "en"
  }

  case object GermanEnglish extends DatasetType {
    override val srcLanguage: String = "de"
    override val tgtLanguage: String = "en"
  }

  case object GreekEnglish extends DatasetType {
    override val srcLanguage: String = "el"
    override val tgtLanguage: String = "en"
  }

  case object EstonianEnglish extends DatasetType {
    override val srcLanguage: String = "et"
    override val tgtLanguage: String = "en"
  }

  case object FinnishEnglish extends DatasetType {
    override val srcLanguage: String = "fi"
    override val tgtLanguage: String = "en"
  }

  case object FrenchEnglish extends DatasetType {
    override val srcLanguage: String = "fr"
    override val tgtLanguage: String = "en"
  }

  case object HungarianEnglish extends DatasetType {
    override val srcLanguage: String = "hu"
    override val tgtLanguage: String = "en"
  }

  case object ItalianEnglish extends DatasetType {
    override val srcLanguage: String = "it"
    override val tgtLanguage: String = "en"
  }

  case object LithuanianEnglish extends DatasetType {
    override val srcLanguage: String = "lt"
    override val tgtLanguage: String = "en"
  }

  case object LatvianEnglish extends DatasetType {
    override val srcLanguage: String = "lv"
    override val tgtLanguage: String = "en"
  }

  case object DutchEnglish extends DatasetType {
    override val srcLanguage: String = "nl"
    override val tgtLanguage: String = "en"
  }

  case object PolishEnglish extends DatasetType {
    override val srcLanguage: String = "pl"
    override val tgtLanguage: String = "en"
  }

  case object PortugueseEnglish extends DatasetType {
    override val srcLanguage: String = "pt"
    override val tgtLanguage: String = "en"
  }

  case object RomanianEnglish extends DatasetType {
    override val srcLanguage: String = "ro"
    override val tgtLanguage: String = "en"
  }

  case object SlovakEnglish extends DatasetType {
    override val srcLanguage: String = "sk"
    override val tgtLanguage: String = "en"
  }

  case object SlovenianEnglish extends DatasetType {
    override val srcLanguage: String = "sl"
    override val tgtLanguage: String = "en"
  }

  case object SpanishEnglish extends DatasetType {
    override val srcLanguage: String = "es"
    override val tgtLanguage: String = "en"
  }

  case object SwedishEnglish extends DatasetType {
    override val srcLanguage: String = "sv"
    override val tgtLanguage: String = "en"
  }

  def download(path: Path, datasetType: DatasetType, bufferSize: Int = 8192): Unit = {
    // Download the data, if necessary.
    val toolsPath = path.resolve("tools.tgz")
    if (!Files.exists(toolsPath))
      maybeDownload(toolsPath, s"$url/tools.tgz", bufferSize)
    val archivePath = path.resolve(datasetType.name + ".tgz")
    if (!Files.exists(archivePath))
      maybeDownload(archivePath, s"$url/${datasetType.name}.tgz", bufferSize)
    CompressedFiles.decompressTGZ(toolsPath, path.resolve("tools"))
    CompressedFiles.decompressTGZ(archivePath, path.resolve(datasetType.name))
  }

  def tokenize(path: Path, datasetType: DatasetType, useMoses: Boolean = true, numThreads: Int = 8): Unit = {
    // Make the tokenizer script executable, if necessary.
    if (!useMoses)
      Seq("chmod", "+x", path.resolve("tools").resolve("tools").resolve("tokenizer.perl").toAbsolutePath.toString).!

    // Clone the Moses repository, if necessary.
    val mosesDecoder = MosesDecoder(path.resolve("moses"))
    if (useMoses && !mosesDecoder.exists)
      mosesDecoder.cloneRepository()

    // Determine the appropriate tokenizer command
    val tokenizeCommand = {
      if (useMoses)
        Seq(mosesDecoder.tokenizerScript.toAbsolutePath.toString, "-q", "-threads", numThreads.toString)
      else
        Seq(path.resolve("tools").resolve("tools").resolve("tokenizer.perl").toAbsolutePath.toString, "-q")
    }

    // Resolve paths
    val dataPath = path.resolve(datasetType.name)
    val srcTokenized = dataPath.resolve(datasetType.srcTokenizedCorpus)
    val tgtTokenized = dataPath.resolve(datasetType.tgtTokenizedCorpus)
    val srcTextFile = dataPath.resolve(datasetType.srcCorpus)
    val tgtTextFile = dataPath.resolve(datasetType.tgtCorpus)

    ((tokenizeCommand ++ Seq("-l", datasetType.srcLanguage)) #< srcTextFile.toFile #> srcTokenized.toFile).!
    ((tokenizeCommand ++ Seq("-l", datasetType.tgtLanguage)) #< tgtTextFile.toFile #> tgtTokenized.toFile).!
  }
}
