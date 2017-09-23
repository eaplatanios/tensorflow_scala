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

import org.platanios.tensorflow.api.ops.Op

import java.nio.file.Path

import scala.collection.mutable

/** Cache for summary file writers, which caches one writer per directory.
  *
  * @author Emmanouil Antonios Platanios
  */
object SummaryFileWriterCache {
  private[this] val cache: mutable.Map[Path, SummaryFileWriter] = mutable.HashMap.empty[Path, SummaryFileWriter]

  /** Returns the summary file writer responsible for the specified directory. */
  def get(directory: Path): SummaryFileWriter = cache synchronized {
    cache.getOrElseUpdate(directory, SummaryFileWriter(directory, Op.currentGraph))
  }

  /** Clears the cached summary writers. Currently only used for testing. */
  private[io] def clear(): Unit = cache synchronized {
    // Make sure all the writers are closed.
    // Otherwise, open file handles may hang around, blocking deletions on Windows.
    cache.values.foreach(_.close())
    cache.clear()
  }
}
