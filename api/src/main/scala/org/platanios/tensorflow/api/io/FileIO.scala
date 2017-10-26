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
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni
import org.platanios.tensorflow.jni.{PermissionDeniedException, FileIO => NativeFileIO}

import java.nio.file._
import java.util.UUID
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/** Helper class used for reading from and writing to a file.
  *
  * @param  filePath       Path to a file.
  * @param  mode           Mode in which to open the file.
  * @param  readBufferSize Buffer size used when reading from the file.
  */
case class FileIO(filePath: Path, mode: FileIO.Mode, readBufferSize: Long = 1024 * 512) extends Closeable {
  private[this] var readBufferNativeHandle  : Long = 0
  private[this] var writableFileNativeHandle: Long = 0

  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the file IO object is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  private[this] def preReadCheck(): Unit = NativeHandleLock.synchronized {
    if (readBufferNativeHandle == 0) {
      if (!mode.supportsRead)
        throw PermissionDeniedException("The specified file mode does not support reading.")
      readBufferNativeHandle = NativeFileIO.newBufferedInputStream(filePath.toAbsolutePath.toString, readBufferSize)
    }
  }

  private[this] def preWriteCheck(): Unit = NativeHandleLock.synchronized {
    if (writableFileNativeHandle == 0) {
      if (!mode.supportsWrite)
        throw PermissionDeniedException("The specified file mode does not support writing.")
      writableFileNativeHandle = NativeFileIO.newWritableFile(filePath.toAbsolutePath.toString, mode.cValue)
    }
  }

  /** Returns the size of the file. */
  def size: Long = FileIO.fileStatistics(filePath).length

  /** Returns the current position in the file. */
  def tell: Long = {
    preReadCheck()
    NativeFileIO.tellBufferedInputStream(readBufferNativeHandle)
  }

  /** Seeks to the provided offset in the file.
    *
    * @param  offset Position offset.
    * @param  whence Position reference, relative to which `offset` is defined.
    */
  def seek(offset: Long, whence: FileIO.Whence = FileIO.START_OF_FILE): FileIO = {
    preReadCheck()
    val position = whence match {
      case FileIO.START_OF_FILE => offset
      case FileIO.CURRENT_POSITION => offset + tell
      case FileIO.END_OF_FILE => offset + size
    }
    NativeFileIO.seekBufferedInputStream(readBufferNativeHandle, position)
    this
  }

  /** Appends `content` to the end of the file. */
  def write(content: String): Unit = {
    preWriteCheck()
    NativeFileIO.appendToWritableFile(writableFileNativeHandle, content)
  }

  /** Returns the contents of the file as a string, starting from current position in the file.
    *
    * @param  numBytes Number of bytes to read from the file. If equal to `-1` (the default) then the file is read up to
    *                  its end.
    * @return Read contents of the file as a string.
    */
  def read(numBytes: Long = -1L): String = {
    preReadCheck()
    NativeFileIO.readFromBufferedInputStream(readBufferNativeHandle, if (numBytes == -1L) size - tell else numBytes)
  }

  /** Reads the next line from the file and returns it (including the new-line character at the end). */
  def readLine(): String = {
    preReadCheck()
    NativeFileIO.readLineAsStringFromBufferedInputStream(readBufferNativeHandle)
  }

  /** Reads all the lines from the file and returns them (including the new-line character at the end of each line). */
  def readLines(): Seq[String] = {
    preReadCheck()
    var lines = Seq.empty[String]
    var continue = true
    while (continue) {
      val line = NativeFileIO.readLineAsStringFromBufferedInputStream(readBufferNativeHandle)
      if (line == null)
        continue = false
      else
        lines :+= line
    }
    lines
  }

  /** Returns an iterator over the lines in this file (including the new-line character at the end of each line). */
  def linesIterator: Iterator[String] = new Iterator[String] {
    private[this] var nextLine: String = readLine()

    override def hasNext: Boolean = nextLine != null
    override def next(): String = {
      val line = nextLine
      nextLine = readLine()
      line
    }
  }

  /** Flushes the file. This only ensures that the data has made its way out of the process without any guarantees on
    * whether it is written to disk. This means that the data would survive an application crash but not necessarily an
    * OS crash. */
  def flush(): Unit = {
    if (writableFileNativeHandle != 0)
      NativeFileIO.flushWritableFile(writableFileNativeHandle)
  }

  /** Closes this file IO object and releases any resources associated with it. Note that an events file reader is not
    * usable after it has been closed. */
  def close(): Unit = {
    NativeHandleLock.synchronized {
      if (readBufferNativeHandle != 0) {
        NativeFileIO.deleteBufferedInputStream(readBufferNativeHandle)
        readBufferNativeHandle = 0
      }
      if (writableFileNativeHandle != 0) {
        NativeFileIO.deleteWritableFile(writableFileNativeHandle)
        writableFileNativeHandle = 0
      }
    }
  }
}

/** Contains helper functions for dealing with file IO.
  *
  * @author Emmanouil Antonios Platanios
  */
object FileIO {
  type FileStatistics = jni.FileStatistics

  val FileStatistics: jni.FileStatistics.type = jni.FileStatistics

  /** Mode in which to open a file. */
  sealed trait Mode {
    val cValue       : String
    val supportsRead : Boolean
    val supportsWrite: Boolean

    override def toString: String = cValue
  }

  /** Open file for reading. The file pointer will be at the beginning of the file. */
  case object READ extends Mode {
    override val cValue       : String  = "r"
    override val supportsRead : Boolean = true
    override val supportsWrite: Boolean = false
  }

  /** Open file for writing only. Overwrite the file if the file exists. If the file does not exist, creates a new file
    * for writing. */
  case object WRITE extends Mode {
    override val cValue       : String  = "w"
    override val supportsRead : Boolean = false
    override val supportsWrite: Boolean = true
  }

  /** Open file for appending. The file pointer will be at the end of the file if the file exists. That is, the file is
    * in the append mode. If the file does not exist, a new file is created for writing. */
  case object APPEND extends Mode {
    override val cValue       : String  = "a"
    override val supportsRead : Boolean = false
    override val supportsWrite: Boolean = true
  }

  /** Open file for both reading and writing. The file pointer will be at the beginning of the file. */
  case object READ_WRITE extends Mode {
    override val cValue       : String  = "r+"
    override val supportsRead : Boolean = true
    override val supportsWrite: Boolean = true
  }

  /** Open file for both reading and writing. Overwrite the file if the file exists. If the file does not exist, creates
    * a new file for writing. */
  case object READ_WRITE_TRUNCATE extends Mode {
    override val cValue       : String  = "w+"
    override val supportsRead : Boolean = true
    override val supportsWrite: Boolean = true
  }

  /** Open file for both reading and appending. The file pointer will be at the end of the file if the file exists. That
    * is, the file is in the append mode. If the file does not exist, a new file is created for writing. */
  case object READ_APPEND extends Mode {
    override val cValue       : String  = "a+"
    override val supportsRead : Boolean = true
    override val supportsWrite: Boolean = true
  }

  /** Used to represent the position reference for calls to `seek`. */
  sealed trait Whence
  case object START_OF_FILE extends Whence
  case object CURRENT_POSITION extends Whence
  case object END_OF_FILE extends Whence

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

  /** Returns `true` if `path` points to a Google Cloud Service (GCS) path. */
  def isGCSPath(path: Path): Boolean = {
    path.startsWith("gs://")
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

  /** Reads the entire contents of the file located at `filePath` to a string and returns it. */
  def readFileToString(filePath: Path): String = {
    FileIO(filePath, READ).read()
  }

  /** Writes the provided string to the file located at `filePath`. */
  def writeStringToFile(filePath: Path, content: String): Unit = {
    FileIO(filePath, WRITE).write(content)
  }

  /** Writes the provided string to the file located at `filePath` as an atomic operation. This means that when
    * `filePath` appears in the filesystem, it will contain all of `content`. With `writeStringToFile()`, it is possible
    * for the file to appear in the filesystem with `content` only partially written.
    *
    * The atomic write is achieved by first writing to a temporary file and then renaming that file.
    */
  def writeStringToFileAtomic(filePath: Path, content: String, overwrite: Boolean = true): Unit = {
    val temporaryFilePath = filePath.resolveSibling(filePath + s".tmp${UUID.randomUUID().toString}")
    writeStringToFile(temporaryFilePath, content)
    try {
      rename(temporaryFilePath, filePath, overwrite)
    } finally {
      deleteFile(temporaryFilePath)
    }
  }
}
