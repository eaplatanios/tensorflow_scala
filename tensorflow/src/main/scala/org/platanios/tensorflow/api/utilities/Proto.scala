package org.platanios.tensorflow.api.utilities

import java.nio.file.{Files, Path}

import com.google.protobuf.{GeneratedMessageV3, TextFormat}

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
