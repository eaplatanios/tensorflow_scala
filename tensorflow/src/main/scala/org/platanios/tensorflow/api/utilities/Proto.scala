package org.platanios.tensorflow.api.utilities

/**
  * @author Emmanouil Antonios Platanios
  */
object Proto {
  trait Serializable {
    /** Converts this object to its corresponding ProtoBuf object.
      *
      * @return ProtoBuf object corresponding to this object.
      */
    def toProto: com.google.protobuf.GeneratedMessageV3
  }
}
