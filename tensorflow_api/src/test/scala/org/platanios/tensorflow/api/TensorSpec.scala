package org.platanios.tensorflow.api

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorSpec extends FlatSpec with Matchers {
  "'Tensor.apply'" must "work for all data types except 'DataType.String'" in {
    val array = Array(Array(2, 3), Array(0, 0), Array(5, 7))
    val tensor = Tensor.create(array)
    assert(tensor(0, 0) == array(0)(0))
    assert(tensor(0, 1) == array(0)(1))
    assert(tensor(1, 0) == array(1)(0))
    assert(tensor(1, 1) == array(1)(1))
    assert(tensor(2, 0) == array(2)(0))
    assert(tensor(2, 1) == array(2)(1))
  }
}
