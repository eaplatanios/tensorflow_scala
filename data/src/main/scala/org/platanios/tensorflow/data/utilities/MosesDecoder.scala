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

package org.platanios.tensorflow.data.utilities

import org.eclipse.jgit.api.Git

import java.io.File
import java.nio.file.{Files, Path}

import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
case class MosesDecoder(path: Path) {
  val gitUrl: String = "https://github.com/moses-smt/mosesdecoder.git"

  def exists: Boolean = Files.exists(path)

  def cloneRepository(): Unit = {
    Git.cloneRepository()
        .setURI(gitUrl)
        .setDirectory(new File(path.toAbsolutePath.toString))
        .call()
  }

  def tokenizerScript: Path = {
    path.resolve("scripts").resolve("tokenizer").resolve("tokenizer.perl")
  }

  def cleanCorpusScript: Path = {
    path.resolve("scripts").resolve("training").resolve("clean-corpus-n.perl")
  }

  def inputFromSGMScript: Path = {
    path.resolve("scripts").resolve("ems").resolve("support").resolve("input-from-sgm.perl")
  }

  def tokenize(textFile: Path, vocabFile: Path, language: String, numThreads: Int = 8): Unit = {
    Seq(tokenizerScript.toAbsolutePath.toString, "-q", "-l", language, "-threads", numThreads.toString) #<
        textFile.toFile #>
        vocabFile.toFile
  }
}
