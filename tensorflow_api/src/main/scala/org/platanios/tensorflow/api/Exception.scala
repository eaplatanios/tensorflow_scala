package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
object Exception {
  case class IllegalNameException(message: String = null, cause: Throwable = null)
      extends IllegalArgumentException(message, cause)

  case class InvalidDeviceSpecificationException(message: String = null, cause: Throwable = null)
      extends IllegalArgumentException(message, cause)

  case class InvalidGraphElementException(message: String = null, cause: Throwable = null)
      extends IllegalArgumentException(message, cause)

  case class InvalidShapeException(message: String = null, cause: Throwable = null)
      extends IllegalArgumentException(message, cause)

  case class OpBuilderUsedException(message: String = null, cause: Throwable = null)
      extends IllegalStateException(message, cause)
}
