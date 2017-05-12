package org.platanios.tensorflow.api.utilities

/**
  * @author Emmanouil Antonios Platanios
  */
object Proto {
  trait Serializable {
    /** Convert this object to its corresponding ProtoBuf object.
      *
      * @param  exportScope Optional string specifying the name scope to remove. All ops within that name scope will not
      *                     be included in the resulting ProtoBuf object.
      * @return ProtoBuf object corresponding to this object.
      */
    def toProto(exportScope: String = null): com.google.protobuf.GeneratedMessageV3
  }

  private[api] trait Implicits {
    type ProtoSerializable = Serializable
  }
}
