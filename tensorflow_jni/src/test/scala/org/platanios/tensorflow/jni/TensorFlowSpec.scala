package org.platanios.tensorflow.jni

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorFlowSpec extends FlatSpec {
  "The TensorFlow library version" must "have non-zero length" in {
    assert(TensorFlow.version.length > 0)
  }

  "The TensorFlow library data type sizes" must "be correct" in {
    assert(TensorFlow.dataTypeSize(1) === 4)
    assert(TensorFlow.dataTypeSize(2) === 8)
    assert(TensorFlow.dataTypeSize(3) === 4)
    assert(TensorFlow.dataTypeSize(4) === 1)
    assert(TensorFlow.dataTypeSize(5) === 2)
    assert(TensorFlow.dataTypeSize(6) === 1)
    assert(TensorFlow.dataTypeSize(7) === 0)
    assert(TensorFlow.dataTypeSize(8) === 8)
    assert(TensorFlow.dataTypeSize(9) === 8)
    assert(TensorFlow.dataTypeSize(10) === 1)
    assert(TensorFlow.dataTypeSize(11) === 1)
    assert(TensorFlow.dataTypeSize(12) === 1)
    assert(TensorFlow.dataTypeSize(13) === 4)
    assert(TensorFlow.dataTypeSize(14) === 0)
    assert(TensorFlow.dataTypeSize(15) === 0)
    assert(TensorFlow.dataTypeSize(16) === 0)
    assert(TensorFlow.dataTypeSize(17) === 2)
    assert(TensorFlow.dataTypeSize(18) === 16)
    assert(TensorFlow.dataTypeSize(19) === 2)
    assert(TensorFlow.dataTypeSize(20) === 0)
  }
}
