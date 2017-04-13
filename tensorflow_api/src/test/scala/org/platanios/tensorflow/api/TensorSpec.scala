package org.platanios.tensorflow.api

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorSpec extends FlatSpec with Matchers {
  "'Tensor.create'" must "create a valid Tensor when provided no data type or shape" in {
    val array1 = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
                       Array(Array(1, 23), Array(4, -5), Array(7, 9)),
                       Array(Array(56, 1), Array(-2, -4), Array(-7, -9)))
    val tensor1 = Tensor.create(array1)
    assert(tensor1.dataType === DataType.Int32)
    assert(tensor1.shape === Shape(3, 3, 2))
    assert(tensor1(1, 1, 1) === -5)
    val array2 = Array(Array(Array(2.0, 3.0), Array(0.0, 0.0), Array(5.0, 7.0)),
                       Array(Array(1.0, 23.0), Array(4.0, -5.0), Array(7.0, 9.0)),
                       Array(Array(56.0, 1.0), Array(-2.0, -4.0), Array(-7.0, -9.0)))
    val tensor2 = Tensor.create(array2)
    assert(tensor2.dataType === DataType.Float64)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1) === -5.0)
  }

  it must "create a valid Tensor when a data type is provided but no shape" in {
    val array1 = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
                       Array(Array(1, 23), Array(4, -5), Array(7, 9)),
                       Array(Array(56, 1), Array(-2, -4), Array(-7, -9)))
    val tensor1 = Tensor.create(array1, dataType = DataType.Int32)
    assert(tensor1.dataType === DataType.Int32)
    assert(tensor1.shape === Shape(3, 3, 2))
    assert(tensor1(1, 1, 1) === -5)
    val tensor2 = Tensor.create(array1, dataType = DataType.Float64)
    assert(tensor2.dataType === DataType.Float64)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1) === -5.0)
  }

  "'Tensor.apply'" must "work for obtaining individual elements for all data types except 'DataType.String'" in {
    val array = Array(Array(2, 3), Array(0, 0), Array(5, 7))
    val tensor = Tensor.create(array)
    assert(tensor(0, 0) == array(0)(0))
    assert(tensor(0, 1) == array(0)(1))
    assert(tensor(1, 0) == array(1)(0))
    assert(tensor(1, 1) == array(1)(1))
    assert(tensor(2, 0) == array(2)(0))
    assert(tensor(2, 1) == array(2)(1))
    assert(tensor(-2, -1) == array(0)(0))
    assert(tensor(0, -1) == array(0)(0))
    assert(tensor(-1, 0) == array(1)(0))
    assert(tensor(-1, -1) == array(1)(0))
    assert(tensor(2, -1) == array(2)(0))
  }

  "'TensorSlice'" must "always have the correct shape" in {
    val array = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
                      Array(Array(1, 23), Array(4, -5), Array(7, 9)),
                      Array(Array(56, 1), Array(-2, -4), Array(-7, -9)))
    // TODO: Move this to the shape specification.
    val tensor = Tensor.create(array)
    assert(tensor.shape === Shape(3, 3, 2))
    assert(tensor.slice(::, NewAxis, 1 :: 3, 0 :: -1).shape === Shape(3, 1, 2, 1))
    assert(tensor.slice(---).shape === Shape(3, 3, 2))
    assert(tensor.slice(::, ---).shape === Shape(3, 3, 2))
    assert(tensor.slice(---, -1).shape === Shape(3, 3, 1))
    assert(tensor.slice(---, 0 ::).shape === Shape(3, 3, 2))
    assert(tensor.slice(---, NewAxis).shape === Shape(3, 3, 2, 1))
    assert(tensor.slice(::, ---, NewAxis).shape === Shape(3, 3, 2, 1))
    assert(tensor.slice(---, -1, NewAxis).shape === Shape(3, 3, 1, 1))
    assert(tensor.slice(---, 0 ::, NewAxis).shape === Shape(3, 3, 2, 1))
  }

  "'Tensor.apply'" must "work for obtaining slices for all data types except 'DataType.String'" in {
    val array = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
                      Array(Array(1, 23), Array(4, -5), Array(7, 9)),
                      Array(Array(56, 1), Array(-2, -4), Array(-7, -9))) // [3, 3, 2] array
    val tensor = Tensor.create(array)
    val tensorSlice1 = tensor.slice(::, NewAxis, 1 :: 3, 0 :: -1)
    assert(tensorSlice1(0, 0, 0, 0) == array(0)(1)(0))
    assert(tensorSlice1(0, 0, 1, 0) == array(0)(2)(0))
    assert(tensorSlice1(1, 0, 0, 0) == array(1)(1)(0))
    assert(tensorSlice1(1, 0, 1, 0) == array(1)(2)(0))
    val tensorSlice2 = tensor.slice(-1, ::, NewAxis, NewAxis, 0)
    assert(tensorSlice2(0, 0, 0, 0, 0) == array(1)(0)(0))
    assert(tensorSlice2(0, 1, 0, 0, 0) == array(1)(1)(0))
    assert(tensorSlice2(0, 2, 0, 0, 0) == array(1)(2)(0))
    val tensorSlice3 = tensor.slice(---, NewAxis, NewAxis, 0)
    assert(tensorSlice3(0, 0, 0, 0, 0) == array(0)(0)(0))
    assert(tensorSlice3(0, 1, 0, 0, 0) == array(0)(1)(0))
    assert(tensorSlice3(0, 2, 0, 0, 0) == array(0)(2)(0))
    assert(tensorSlice3(1, 0, 0, 0, 0) == array(1)(0)(0))
    assert(tensorSlice3(1, 1, 0, 0, 0) == array(1)(1)(0))
    assert(tensorSlice3(1, 2, 0, 0, 0) == array(1)(2)(0))
  }
}
