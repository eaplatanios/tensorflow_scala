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

package org.platanios.tensorflow.data.nmt.utilities

import org.platanios.tensorflow.api._

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.collection.mutable

/** Contains utilities for dealing with vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
object Vocabulary {
  private[this] val logger: Logger = Logger(LoggerFactory.getLogger("Vocabulary Utilities"))

  val BEGIN_SEQUENCE_TOKEN: String = "<s>"
  val END_SEQUENCE_TOKEN  : String = "</s>"
  val UNKNOWN_TOKEN       : String = "<unk>"
  val UNKNOWN_TOKEN_ID    : Int    = 0

  /** Checks if the specified vocabulary file exists and if it does, checks that special tokens are being used
    * correctly. If not, this method can optionally create a new file by prepending the appropriate tokens to the
    * existing one.
    *
    * The special tokens check simply involves checking whether the first three tokens in the vocabulary file match the
    * specified `unknownToken`, `beginSequenceToken`, and `endSequenceToken` values.
    *
    * @param  file               Vocabulary file to check.
    * @param  checkSpecialTokens Boolean value indicating whether or not to check for the use of special tokens, and
    *                            prepend them while creating a new vocabulary file, if the check fails.
    * @param  directory          Directory to use when creating the new vocabulary file, in case the special tokens
    *                            check fails. Defaults to the current directory in which `file` is located, meaning that
    *                            if the special tokens check fails, `file` will be replaced with the appended vocabulary
    *                            file.
    * @param  beginSequenceToken Special token for the beginning of a sequence. Defaults to `<s>`.
    * @param  endSequenceToken   Special token for the end of a sequence. Defaults to `</s>`.
    * @param  unknownToken       Special token for unknown tokens. Defaults to `<unk>`.
    */
  def check(
      file: Path, checkSpecialTokens: Boolean = true, directory: Path = null,
      beginSequenceToken: String = BEGIN_SEQUENCE_TOKEN, endSequenceToken: String = END_SEQUENCE_TOKEN,
      unknownToken: String = UNKNOWN_TOKEN): Option[(Int, Path)] = {
    if (!Files.exists(file)) {
      None
    } else {
      logger.info(s"Vocabulary file '$file' exists.")
      val reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)
      val tokens = mutable.ListBuffer.empty[String]
      var line = reader.readLine()
      while (line != null) {
        tokens ++= line.split("\\s+").toSeq
        line = reader.readLine()
      }
      reader.close()
      if (!checkSpecialTokens) {
        Some((tokens.size, file))
      } else {
        // Verify that the loaded vocabulary using the right special tokens.
        // If it does not, use those tokens and generate a new vocabulary file.
        assert(tokens.size >= 3, "The loaded vocabulary must contain at least three tokens.")
        if (tokens(0) != unknownToken || tokens(1) != beginSequenceToken || tokens(2) != endSequenceToken) {
          logger.info(
            s"The first 3 vocabulary tokens [${tokens(0)}, ${tokens(1)}, ${tokens(2)}] " +
                s"are not equal to [$unknownToken, $beginSequenceToken, $endSequenceToken].")
          tokens.prepend(unknownToken, beginSequenceToken, endSequenceToken)
          val newFile = if (directory != null) directory.resolve(file.getFileName) else file
          val writer = Files.newBufferedWriter(newFile, StandardCharsets.UTF_8)
          tokens.foreach(token => writer.write(s"$token\n"))
          writer.close()
          Some((tokens.size, newFile))
        } else {
          Some((tokens.size, file))
        }
      }
    }
  }

  /** Creates vocabulary lookup tables (from word string to word ID), from the provided vocabulary files.
    *
    * @param  sourceFile Source vocabulary file.
    * @param  targetFile Target vocabulary file.
    * @return Tuple contain the source vocabulary lookup table and the target one.
    */
  def createTables(sourceFile: Path, targetFile: Path): (tf.LookupTable, tf.LookupTable) = {
    val sourceTable = tf.indexTableFromFile(sourceFile.toAbsolutePath.toString, defaultValue = UNKNOWN_TOKEN_ID)
    val targetTable = {
      if (sourceFile == targetFile)
        sourceTable
      else
        tf.indexTableFromFile(targetFile.toAbsolutePath.toString, defaultValue = UNKNOWN_TOKEN_ID)
    }
    (sourceTable, targetTable)
  }
}
