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

package org.platanios.tensorflow.api.utilities

import org.platanios.tensorflow.api.io.FileIO

import com.google.protobuf.{GeneratedMessageV3, TextFormat}

import java.nio.file.{Files, Path}

/** Contains helper functions for working with ProtoBuf.
  *
  * @author Emmanouil Antonios Platanios
  */
object Proto {
  /** Writes `message` to the specified file.
    *
    * @param  directory Directory in which to write the file.
    * @param  filename  Name of the file.
    * @param  message   ProtoBuf message to write.
    * @param  asText    Boolean value indicating whether to serialize the ProtoBuf message in the human-friendly text
    *                   format, or in the more efficient binary format.
    * @return Path of the written file.
    */
  def write(directory: Path, filename: String, message: GeneratedMessageV3, asText: Boolean = false): Path = {
    // GCS does not have the concept of a directory at the moment.
    if (!Files.exists(directory) && !directory.startsWith("gs:")) {
      Files.createDirectories(directory)
    }
    val filePath = directory.resolve(filename)
    if (asText)
      FileIO.writeStringToFileAtomic(filePath, TextFormat.printToString(message))
    else
      message.writeTo(Files.newOutputStream(filePath))
    filePath
  }

  /** Trait that all ProtoBuf-serializable objects should extend. */
  trait Serializable {
    /** Converts this object to its corresponding ProtoBuf object.
      *
      * @return ProtoBuf object corresponding to this object.
      */
    def toProto: GeneratedMessageV3
  }
}
