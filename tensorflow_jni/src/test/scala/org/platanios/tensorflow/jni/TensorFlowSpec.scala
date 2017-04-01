package org.platanios.tensorflow.jni

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorFlowSpec extends FlatSpec {
  "The TensorFlow library version" should "have non-zero length" in {
    assert(TensorFlow.version.length > 0)
  }
}
