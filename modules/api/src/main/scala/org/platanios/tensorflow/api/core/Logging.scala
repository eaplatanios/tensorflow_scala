package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.jni.TensorFlow

object Logging {

  /** Represents the TensorFlow logging level. */
  sealed trait Level {
    private[Logging] def value: Int
  }

  case object DEBUG extends Level {
    override private[Logging] def value = 0
  }

  case object INFO extends Level {
    override private[Logging] def value = 1
  }

  case object WARNING extends Level {
    override private[Logging] def value = 2
  }

  case object ERROR extends Level {
    override private[Logging] def value = 3
  }

  /** Sets the current TensorFlow logging [[Level]]. */
  def setLoggingLevel(level: Level, overwrite: Boolean = true): Unit = {
    TensorFlow.setLogLevel(level.value.toString, overwrite)
  }

  /** Returns the current TensorFlow logging [[Level]]. */
  def currentLoggingLevel: Level = TensorFlow.getLogLevel match {
    case null | "0" => DEBUG
    case "1" => INFO
    case "2" => WARNING
    case "3" => ERROR
    case _ => throw new AssertionError("This should be unreachable.")
  }
}
