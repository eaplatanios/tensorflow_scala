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

package org.platanios.tensorflow.api.io

import org.platanios.tensorflow.api.core.exception.NotFoundException
import org.platanios.tensorflow.jni
import org.platanios.tensorflow.jni.{FileIO => NativeFileIO}

import java.nio.file._
import java.util.UUID
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/** Contains helper functions for dealing with file IO.
  *
  * @author Emmanouil Antonios Platanios
  */
object FileIO {
  type FileStatistics = jni.FileStatistics
  val FileStatistics: jni.FileStatistics.type = jni.FileStatistics

  /** Determines whether a file path exists or not.
    *
    * @param  filePath File path.
    * @return `true` if the path exists, whether its a file or a directory, and `false` if the path does not exist and
    *         there are no filesystem errors.
    */
  def fileExists(filePath: Path): Boolean = {
    try {
      NativeFileIO.fileExists(filePath.toAbsolutePath.toString)
      true
    } catch {
      case _: NotFoundException => false
    }
  }

  /** Deletes the file located at `filePath`.
    *
    * @param  filePath File path.
    */
  def deleteFile(filePath: Path): Unit = {
    NativeFileIO.deleteFile(filePath.toAbsolutePath.toString)
  }

  /** Creates a directory at `path`, but requires that all the parent directories exist. If they may not exist,
    * `mkDirs()` should be used instead. The call is also successful if a directory already exists at `path`.
    *
    * @param  path Directory path.
    */
  def mkDir(path: Path): Unit = {
    NativeFileIO.mkDir(path.toAbsolutePath.toString)
  }

  /** Creates a directory at `path`, along with all necessary parent/intermediate directories. The call is also
    * successful if a directory already exists at `path`.
    *
    * @param  path Directory path.
    */
  def mkDirs(path: Path): Unit = {
    NativeFileIO.mkDirs(path.toAbsolutePath.toString)
  }

  /** Deletes everything under the provided path, recursively.
    *
    * @param  path File/directory path to delete.
    */
  def deleteRecursively(path: Path): Unit = {
    NativeFileIO.deleteRecursively(path.toAbsolutePath.toString)
  }

  /** Returns `true` is `path` points to a directory, and `false` otherwise.
    *
    * @param  path Path to a file or directory.
    * @return `true` is `path` points to a directory, and `false` otherwise.
    */
  def isDirectory(path: Path): Boolean = {
    NativeFileIO.isDirectory(path.toAbsolutePath.toString)
  }

  /** Returns file statistics for the provided path.
    *
    * @param  path Path to a file or directory.
    * @return File statistics for the file/directory pointed to by `path`.
    */
  def fileStatistics(path: Path): FileStatistics = {
    NativeFileIO.statistics(path.toAbsolutePath.toString)
  }

  /** Copies data from the file at `oldPath` to a new file at `newPath`.
    *
    * @param  oldPath   Old file path.
    * @param  newPath   New file path.
    * @param  overwrite Boolean value indicating whether it is allowed to overwrite the file at `newPath`, if one
    *                   already exists.
    */
  def copyFile(oldPath: Path, newPath: Path, overwrite: Boolean = false): Unit = {
    NativeFileIO.copyFile(oldPath.toAbsolutePath.toString, newPath.toAbsolutePath.toString, overwrite)
  }

  /** Rename/move data from the file/directory at `oldPath` to a new file/directory at `newPath`.
    *
    * @param  oldPath   Old path.
    * @param  newPath   New path.
    * @param  overwrite Boolean value indicating whether it is allowed to overwrite the file/directory at `newPath`, if
    *                   one already exists.
    */
  def rename(oldPath: Path, newPath: Path, overwrite: Boolean = false): Unit = {
    NativeFileIO.renameFile(oldPath.toAbsolutePath.toString, newPath.toAbsolutePath.toString, overwrite)
  }

  /** Returns all entries contained within the directory at `path`. The returned sequence is in arbitrary order and it
    * does not contain the special entries `"."` and `".."`.
    *
    * @param  dirPath Path to a directory.
    * @return Sequence of entries contained in the directory at `path`.
    * @throws NotFoundException If the provided directory does not exist.
    */
  @throws[NotFoundException]
  def listDirectories(dirPath: Path): Seq[Path] = {
    if (!isDirectory(dirPath))
      throw NotFoundException("Could not find the specified directory.")
    NativeFileIO.getChildren(dirPath.toAbsolutePath.toString).map(Paths.get(_))
  }

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
        listDirectories(dirPath)
      } catch {
        case _: NotFoundException => Seq.empty[Path]
      }
    }
    var files: Seq[Path] = Seq.empty[Path]
    var subDirs: Seq[Path] = Seq.empty[Path]
    children.foreach(child => {
      val fullPath = dirPath.resolve(child)
      if (!isDirectory(fullPath))
        files :+= child
      else
        subDirs :+= child
    })
    val hereStream = Stream((dirPath, subDirs, files))
    val subDirsStream = subDirs.toStream.map(walk(_, inOrder)).reduceLeft(_ ++ _)
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
  def getMatchingPaths(path: Path, useNativeFileIO: Boolean = true): Set[Path] = {
    val pathAsString = path.toAbsolutePath.toString
    if (useNativeFileIO) {
      NativeFileIO.getMatchingFiles(pathAsString).toSet[String].map(Paths.get(_))
    } else {
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
