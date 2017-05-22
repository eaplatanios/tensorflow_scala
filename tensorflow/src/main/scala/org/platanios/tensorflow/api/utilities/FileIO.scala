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

package org.platanios.tensorflow.api.utilities

import org.apache.commons.lang3.StringUtils

import java.nio.file._
import java.util.UUID
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/** Contains helper functions for dealing with file IO.
  *
  * @author Emmanouil Antonios Platanios
  */
object FileIO {
  /** Returns a file's last modified time.
    *
    * @param  path                Path to the file.
    * @param  unit                Time unit in which to return the last modified time. Defaults to [[TimeUnit.SECONDS]].
    * @param  followSymbolicLinks Boolean value indicating whether or not to follow symbolic links. By default, symbolic
    *                             links are followed and the file attribute of the final target of the link is read. If
    *                             `followSymbolicLinks` is set to `false`, then symbolic links are not followed.
    * @return
    */
  def getLastModifiedTime(path: Path, unit: TimeUnit = TimeUnit.SECONDS, followSymbolicLinks: Boolean = true): Long = {
    if (!followSymbolicLinks)
      Files.getLastModifiedTime(path, LinkOption.NOFOLLOW_LINKS).to(unit)
    else
      Files.getLastModifiedTime(path).to(unit)
  }

  /** Gets all the matching paths to the path pattern provided.
    *
    * The pattern must follow the TensorFlow path pattern rules and it pattern must match all of a name (i.e., not just
    * a substring). A pattern definition has the following form:
    * {{{
    *   pattern: { term }
    *   term:
    *     '*': matches any sequence of non-'/' characters
    *     '?': matches a single non-'/' character
    *     '[' [ '^' ] { match-list } ']': matches any single character (not) on the list
    *     c: matches character c (c != '*', '?', '\\', '[')
    *     '\\' c: matches character c
    *   character-range:
    *     c: matches character c (c != '\\', '-', ']')
    *     '\\' c: matches character c
    *     lo '-' hi: matches character c for lo <= c <= hi
    * }}}
    *
    * @param  path Path pattern.
    * @return Set of all paths matching the provided path pattern.
    */
  def getMatchingPaths(path: Path): Set[Path] = {
    val separator = FileSystems.getDefault.getSeparator

    // Find the fixed prefix by looking for the first wildcard.
    val pathAsString = path.toString
    val glob = pathAsString.replaceAll("([^\\[]*)\\[\\^", "$1\\[!")
    val directory = {
      val prefix = pathAsString.substring(0, StringUtils.indexOfAny(pathAsString, "*?[\\"))
      path.getFileSystem.getPath(prefix.substring(0, prefix.lastIndexOf(separator)))
    }

    // Get all the matching paths.
    Files.newDirectoryStream(directory, glob).asScala.toSet[Path]
  }

  /** Deletes all the matching paths to the path pattern provided.
    *
    * The pattern must follow the TensorFlow path pattern rules and it pattern must match all of a name (i.e., not just
    * a substring). A pattern definition has the following form:
    * {{{
    *   pattern: { term }
    *   term:
    *     '*': matches any sequence of non-'/' characters
    *     '?': matches a single non-'/' character
    *     '[' [ '^' ] { match-list } ']': matches any single character (not) on the list
    *     c: matches character c (c != '*', '?', '\\', '[')
    *     '\\' c: matches character c
    *   character-range:
    *     c: matches character c (c != '\\', '-', ']')
    *     '\\' c: matches character c
    *     lo '-' hi: matches character c for lo <= c <= hi
    * }}}
    *
    * @param  path Path pattern.
    */
  def deleteMatchingPaths(path: Path): Unit = {
    getMatchingPaths(path).foreach(Files.delete)
  }

  /** Writes the provided string to the file located at `filePath`. */
  def writeStringToFile(filePath: Path, string: String): Unit = {
    Files.write(filePath, Seq(string).asJava)
  }

  /** Writes the provided string to the file located at `filePath` as an atomic operation. This is achieved by first
    * writing to a temporary file and then renaming that file. */
  def writeStringToFileAtomic(filePath: Path, string: String): Unit = {
    val temporaryFilePath = filePath.resolveSibling(filePath + s".tmp${UUID.randomUUID().toString}")
    writeStringToFile(temporaryFilePath, string)
    try {
      Files.move(temporaryFilePath, filePath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    } catch {
      case _: AtomicMoveNotSupportedException =>
        Files.move(temporaryFilePath, filePath, StandardCopyOption.REPLACE_EXISTING)
    }
  }
}
