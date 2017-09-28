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

import org.platanios.tensorflow.api.core.exception.UnavailableException
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{CheckpointReader => NativeCheckpointReader}

import java.nio.file.Path

/** Helper class for reading checkpoint files.
  *
  * This class currently only interacts with single-slice (i.e., non-partitioned) variables.
  *
  * @param  nativeHandle Handle to a native checkpoint reader object.
  *
  * @author Emmanouil Antonios Platanios
  */
class CheckpointReader private[CheckpointReader] (private[CheckpointReader] var nativeHandle: Long)
    extends Closeable {
  /** Lock for the native handle. */
  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the reader is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Checks if the checkpoint file contains a tensor named `name`.
    *
    * @param  name Tensor name.
    * @return `true` if the tensor exists in the checkpoint file and `false` otherwise.
    * @throws UnavailableException If this checkpoint reader object has already been disposed.
    */
  @throws[UnavailableException]
  def hasTensor(name: String): Boolean = {
    if (nativeHandle == 0)
      throw UnavailableException("This checkpoint reader has already been disposed.")
    NativeCheckpointReader.hasTensor(nativeHandle, name)
  }

  /** Attempts to look up the tensor named `name` in the checkpoint file. If found, the tensor is returned, otherwise
    * `None` is returned.
    *
    * @param  name Tensor name.
    * @return Found (or not) tensor.
    * @throws UnavailableException If this checkpoint reader object has already been disposed.
    */
  @throws[UnavailableException]
  def getTensor(name: String): Option[Tensor] = {
    if (nativeHandle == 0)
      throw UnavailableException("This checkpoint reader has already been disposed.")
    Option(NativeCheckpointReader.getTensor(nativeHandle, name)).map(Tensor.fromNativeHandle)
  }

  /** Closes this [[CheckpointReader]] and releases any resources associated with it. Note that a [[CheckpointReader]]
    * is not usable after it has been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        NativeCheckpointReader.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

object CheckpointReader {
  /** Creates a new [[CheckpointReader]] for the checkpoint file pointed to by `checkpointPath`.
    *
    * @param  checkpointPath Path to a checkpoint file.
    * @return Constructed checkpoint reader.
    */
  def apply(checkpointPath: Path): CheckpointReader = {
    new CheckpointReader(NativeCheckpointReader.newCheckpointReader(checkpointPath.toAbsolutePath.toString))
  }
}
