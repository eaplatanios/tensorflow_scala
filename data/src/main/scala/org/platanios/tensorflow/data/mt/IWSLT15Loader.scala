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

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.data.Loader
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15Loader extends Loader {
  val url: String = "https://nlp.stanford.edu/projects/nmt/data"

  override protected val logger = Logger(LoggerFactory.getLogger("IWSLT-15 Data Loader"))

  sealed trait DatasetType {
    val directoryName        : String
    val trainDatasetFilenames: (String, String)
    val devDatasetFilenames  : (String, String)
    val testDatasetFilenames : (String, String)
    val vocabularyFilenames  : (String, String)

    def files: Seq[(String, String)] = {
      val filePaths = Seq(
        directoryName + trainDatasetFilenames._1, directoryName + trainDatasetFilenames._2,
        directoryName + devDatasetFilenames._1, directoryName + devDatasetFilenames._2,
        directoryName + testDatasetFilenames._1, directoryName + testDatasetFilenames._2,
        directoryName + vocabularyFilenames._1, directoryName + vocabularyFilenames._2)
      val urls = filePaths.map(f => s"$url/f")
      filePaths.zip(urls)
    }
  }

  case object EnglishVietnamese extends DatasetType {
    override val directoryName        : String           = "iwslt15.en-vi/"
    override val trainDatasetFilenames: (String, String) = ("train.en", "train.vi")
    override val devDatasetFilenames  : (String, String) = ("tst2012.en", "tst2012.vi")
    override val testDatasetFilenames : (String, String) = ("tst2013.en", "tst2013.vi")
    override val vocabularyFilenames  : (String, String) = ("vocab.en", "vocab.vi")
  }

  def download(path: Path, datasetType: DatasetType = EnglishVietnamese, bufferSize: Int = 8192): TranslationDataset = {
    // Download the data, if necessary.
    val files = datasetType.files
        .filter(f => !Files.exists(path.resolve(f._1)))
        .map(f => {
          val file = path.resolve(f._1)
          maybeDownload(file, f._2, bufferSize)
          file
        })
    TranslationDataset(files(6), files(7), files(0), files(1), files(2), files(3), files(4), files(5))
  }
}
