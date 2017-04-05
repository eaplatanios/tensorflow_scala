package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
object Exception {
  case class IllegalNameException(message: String = null, cause: Throwable = null)
      extends IllegalArgumentException(message, cause)

  case class OpBuilderUsedException(message: String = null, cause: Throwable = null)
      extends IllegalStateException(message, cause)
}
