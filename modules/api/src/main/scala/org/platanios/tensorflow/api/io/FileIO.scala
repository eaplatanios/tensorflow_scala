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

package org.platanios.tensorflow.api.io

import org.platanios.tensorflow.api.core.exception.NotFoundException

import java.nio.charset.Charset
import java.nio.file._
import java.util.UUID
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/** Contains helper functions for dealing with file IO.
  *
  * @author Emmanouil Antonios Platanios
  */
object FileIO {
  // TODO: Replace this with `Files.walkFileTree`.
  /** Walks a directory tree rooted at `dirPath` recursively.
    *
    * Note that exceptions that are thrown while listing directories are ignored.
    *
    * @param  dirPath Path to a directory from which to start traversing.
    * @param  inOrder If `true`, the top-level directories are returned first, otherwise they are returned last.
    * @return Stream over tuples containing: (i) the path for the directory, (ii) a sequence containing all of its
    *         subdirectories, and (iii) a sequence containing all files in that directory.
    */
  def walk(dirPath: Path, inOrder: Boolean = true): Stream[(Path, Seq[Path], Seq[Path])] = {
    val children: Seq[Path] = {
      try {
        Files.walk(dirPath, 1).iterator().asScala.toSeq
      } catch {
        case _: NotFoundException => Seq.empty[Path]
      }
    }
    var files: Seq[Path] = Seq.empty[Path]
    var subDirs: Seq[Path] = Seq.empty[Path]
    children.foreach(child => {
      val fullPath = dirPath.resolve(child)
      if (!Files.isDirectory(fullPath))
        files :+= child
      else
        subDirs :+= child
    })
    val hereStream = Stream((dirPath, subDirs, files))
    val subDirsStream = subDirs.toStream.map(s => walk(dirPath.resolve(s), inOrder))
        .foldLeft(Stream.empty[(Path, Seq[Path], Seq[Path])])(_ ++ _)
    if (inOrder) {
      hereStream ++ subDirsStream
    } else {
      subDirsStream ++ hereStream
    }
  }

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
    val pathAsString = path.toAbsolutePath.toString
    val separator = FileSystems.getDefault.getSeparator

    // Find the fixed prefix by looking for the first wildcard.
    val (directory, glob) = {
      val wildcard = pathAsString.indexWhere("*?[\\".contains(_))
      val prefix = if (wildcard != -1) pathAsString.substring(0, wildcard) else pathAsString
      val suffix = if (wildcard != -1) pathAsString.substring(wildcard) else ""
      val separatorIndex = prefix.lastIndexOf(separator)
      val directory = path.getFileSystem.getPath(prefix.substring(0, separatorIndex))
      val glob = (prefix.substring(separatorIndex + 1) + suffix).replaceAll("([^\\[]*)\\[\\^", "$1\\[!")
      (directory, glob)
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

  /** Writes the provided string to the file located at `filePath` as an atomic operation. This means that when
    * `filePath` appears in the filesystem, it will contain all of `content`. With `writeStringToFile()`, it is possible
    * for the file to appear in the filesystem with `content` only partially written.
    *
    * The atomic write is achieved by first writing to a temporary file and then renaming that file.
    */
  def writeStringToFileAtomic(filePath: Path, content: String, overwrite: Boolean = true): Unit = {
    val temporaryFilePath = filePath.resolveSibling(filePath + s".tmp${UUID.randomUUID().toString}")
    Files.write(
      temporaryFilePath,
      content.getBytes(Charset.forName("UTF-8")),
      StandardOpenOption.WRITE,
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING)
    try {
      if (overwrite)
        Files.move(temporaryFilePath, filePath, StandardCopyOption.REPLACE_EXISTING)
      else
        Files.move(temporaryFilePath, filePath)
    } finally {
      if (Files.exists(temporaryFilePath))
        Files.deleteIfExists(temporaryFilePath)
    }
  }
}
