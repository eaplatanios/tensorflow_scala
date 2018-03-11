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

package org.platanios.tensorflow.api.io

import org.platanios.tensorflow.api.core.exception.UnavailableException
import org.platanios.tensorflow.api.io.DirectoryLoader.DirectoryDeletedException

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/** A directory loader that wraps a loader factory to load entries from a sequence of paths.
  *
  * A [[Loader]] reads a path and produces an iterator over entries. An [[DirectoryLoader]] takes a directory, a factory
  * for entry loaders, and optionally a path filter and watches all the paths inside that directory.
  *
  * This class is only valid under the assumption that only one path will be written to by the data source at a time and
  * that once the source stops writing to a path, it will start writing to a new path that is lexicographically greater,
  * and never come back. It uses some heuristics to check whether this is true based on tracking changes to the files'
  * sizes, but the checks can yield false negatives. However, they should never yield false positives.
  *
  * @param  directory     Directory to watch.
  * @param  loaderFactory Factory for loaders.
  * @param  pathFilter    Optional path filter to use.
  *
  * @author Emmanouil Antonios Platanios
  */
case class DirectoryLoader[T](
    private val directory: Path,
    private val loaderFactory: (Path) => Loader[T],
    private val pathFilter: (Path) => Boolean = _ => true) {
  private[this] var _path: Path = _
  private[this] var _loader                  : Loader[T] = _
  private[this] var _outOfOrderWritesDetected: Boolean         = false
  private[this] var _finalizedSizes          : Map[Path, Long] = Map.empty[Path, Long]

  /** Number of paths before the current one to check for out of order writes. */
  private[this] val OUT_OF_ORDER_WRITE_CHECK_COUNT: Int = 20

  /** Returns a boolean value indicating whether any out-of-order writes have been detected.
    *
    * Out-of-order writes are only checked as part of the `load()` iterator. After an out-of-order write is detected,
    * this method will always return `true`.
    *
    * Note that out-of-order write detection is not performed on GCS paths, and so, in that case, this function will
    * always return `false`. */
  def outOfOrderWritesDetected: Boolean = _outOfOrderWritesDetected

  /** Loads new entries and returns an iterator over them. The loader will load from one path at a time; as soon as
    * that path stops yielding entries, it will move on to the next path. We assume that old paths are never modified
    * after a newer path has been written. As a result, `load()` can be called multiple times in a row without losing
    * entries that have not been yielded yet. In other words, we guarantee that every entry will be yielded exactly
    * once.
    *
    * @throws DirectoryDeletedException If the directory being watched has been permanently deleted (as opposed to being
    *                                   temporarily unavailable).
    */
  @throws[DirectoryDeletedException]
  def load(): Iterator[T] = new Iterator[T] {
    private[this] var currentIterator: Iterator[T] = {
      if (_loader == null) {
        _path = nextPath()
        if (_path != null)
          setPath(_path)
      }
      if (_loader == null)
        null
      else
        _loader.load()
    }

    private[this] var secondPass: Boolean = false

    private[this] def maybeNextPath(): Unit = {
      if (!currentIterator.hasNext) {
        val next = nextPath()
        if (next == null) {
          DirectoryLoader.logger.info(s"No path found after '${_path}'.")
          // The current path is empty and there are no new paths, and so we are done.
          currentIterator = null
        } else {
          if (!secondPass) {
            // There is a new path and so we check to make sure there were not any entries written between when we
            // finished reading the current path and when we checked for the new one. The sequence of entries might look
            // something like this:
            //   1. Entry #1 written to path #1.
            //   2. We check for entries and yield entry #1 from path #1.
            //   3. We check for entries and see that there are no more entries in path #1.
            //   4. Entry #2 is written to path #1.
            //   5. Entry #3 is written to path #2.
            //   6. We check for a new path and see that path #2 exists.
            // Without this loop, we would miss entry #2. We are also guaranteed by the reader contract that no more
            // entries will be written to path #1 after entries start being written to path #2, and so we do not have to
            // worry about that.
            currentIterator = _loader.load()
          }
          if (currentIterator.hasNext) {
            secondPass = true
          } else {
            DirectoryLoader.logger.info(s"Directory loader advancing from '${_path}' to '$next'.")
            secondPass = false
            // Advance to the next path and start over.
            setPath(next)
            currentIterator = _loader.load()
          }
        }
      }
    }

    override def hasNext: Boolean = {
      try {
        if (currentIterator == null) {
          false
        } else if (!currentIterator.hasNext) {
          maybeNextPath()
          currentIterator.hasNext
        } else {
          true
        }
      } catch {
        case _: Throwable =>
          if (!FileIO.exists(directory))
            throw DirectoryDeletedException(s"Directory '$directory' has been permanently deleted.")
          false
      }
    }

    override def next(): T = {
      val event = currentIterator.next()
      try {
        maybeNextPath()
      } catch {
        case _: Throwable =>
          if (!FileIO.exists(directory))
            throw DirectoryDeletedException(s"Directory '$directory' has been permanently deleted.")
      }
      event
    }
  }

  /** Gets the next path to load entries from. This method also does the checking for out-of-order writes as it iterates
    * through the paths. */
  private[this] def nextPath(): Path = {
    val sortedPaths = FileIO.listDirectories(directory).map(directory.resolve).filter(pathFilter).sortBy(_.toString)
    if (sortedPaths.isEmpty) {
      null
    } else if (_path == null) {
      sortedPaths.head
    } else {
      val currentPathIndex = sortedPaths.indexOf(_path)
      // Do not bother checking if the paths are in GCS (which we cannot check) or if we have already detected an
      // out-of-order write.
      if (!FileIO.isGCSPath(sortedPaths.head) && !outOfOrderWritesDetected) {
        // Check the previous `OUT_OF_ORDER_WRITE_CHECK_COUNT` paths for out of order writes.
        val outOfOrderCheckStart = math.max(0, currentPathIndex - OUT_OF_ORDER_WRITE_CHECK_COUNT)
        _outOfOrderWritesDetected = sortedPaths.slice(outOfOrderCheckStart, currentPathIndex).exists(hasOutOfOrderWrite)
      }
      sortedPaths.drop(currentPathIndex + 1).headOption.orNull
    }
  }

  /** Sets the current path to watch for new entries to `path`. This method also records the size of the old path, if
    * any. If the size cannot be determined, an error is logged. */
  private[this] def setPath(path: Path): Unit = {
    val oldPath = this._path
    if (oldPath != null && !FileIO.isGCSPath(oldPath)) {
      try {
        // We are done with the path, and so we store its size.
        val size = FileIO.fileStatistics(oldPath).length
        DirectoryLoader.logger.debug(s"Setting latest size of '$oldPath' to $size.")
        _finalizedSizes = _finalizedSizes.updated(oldPath, size)
      } catch {
        case t: Throwable => DirectoryLoader.logger.error(s"Unable to get the size of '$oldPath'.", t)
      }
    }
    this._path = path
    this._loader = loaderFactory(path)
  }

  /** Returns a boolean value indicating whether `path` has had an out-of-order write. */
  private[this] def hasOutOfOrderWrite(path: Path): Boolean = {
    // Check the sizes of each path before the current one.
    val size = FileIO.fileStatistics(path).length
    val oldSize = _finalizedSizes.getOrElse(path, -1L)
    if (size != oldSize) {
      if (oldSize == -1L)
        DirectoryLoader.logger.error(
          s"File '$path' created after file '${this._path}' " +
              s"even though its name lexicographical order indicates otherwise.")
      else
        DirectoryLoader.logger.error(s"File '$path' updated even though the current file is '${this._path}'.")
      true
    } else {
      false
    }
  }
}

object DirectoryLoader {
  private[DirectoryLoader] val logger: Logger = Logger(LoggerFactory.getLogger("Directory Loader"))

  /** Exception thrown by `DirectoryLoader.load()` when the directory being watched is *permanently* gone (i.e.,
    * deleted). We distinguish this from temporary errors so that other code can decide to drop all of our data only
    * when a directory has been intentionally deleted, as opposed to due to transient filesystem errors. */
  case class DirectoryDeletedException(message: String = null, cause: Throwable = null)
      extends UnavailableException(message, cause)
}
