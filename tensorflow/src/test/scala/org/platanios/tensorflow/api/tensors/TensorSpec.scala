package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api._

import org.scalatest._

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorSpec extends FlatSpec with Matchers {
  "'Tensor.create'" must "create a valid numeric tensors when provided no data type or shape" in {
    val tensor1: Tensor = -2
    assert(tensor1.dataType === TFInt32)
    assert(tensor1.shape === Shape())
    assert(tensor1(0).scalar === -2)
    val tensor2 = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor2.dataType === TFInt32)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1).scalar === -5)
    val tensor3 = Tensor(Tensor(Tensor(2.0, 3.0), Tensor(0.0, 0.0), Tensor(5.0, 7.0)),
                         Tensor(Tensor(1.0, 23.0), Tensor(4.0, -5.0), Tensor(7.0, 9.0)),
                         Tensor(Tensor(56.0, 1.0), Tensor(-2.0, -4.0), Tensor(-7.0, -9.0)))
    assert(tensor3.dataType === TFFloat64)
    assert(tensor3.shape === Shape(3, 3, 2))
    assert(tensor3(1, 1, 1).scalar === -5.0)
    val tensor4: Tensor = Tensor(5, 6.0)
    assert(tensor4.dataType === TFFloat64)
    assert(tensor4.shape === Shape(2))
    assert(tensor4(0).scalar === 5.0)
    assert(tensor4(1).scalar === 6.0)
    val tensor5: Tensor = "foo"
    assert(tensor5.dataType === TFString)
    assert(tensor5.shape === Shape())
    assert(tensor5(0).scalar === "foo")
    val tensor6: Tensor = Tensor("foo", "bar")
    assert(tensor6.dataType === TFString)
    assert(tensor6.shape === Shape(2))
    assert(tensor6(0).scalar === "foo")
    assert(tensor6(1).scalar === "bar")
    val tensor7 = Tensor(Tensor(Tensor("0,0,0", "0,0,1"), Tensor("0,1,0", "0,1,1"), Tensor("0,2,0", "0,2,1")),
                         Tensor(Tensor("1,0,0", "1,0,1"), Tensor("1,1,0", "1,1,1"), Tensor("1,2,0", "1,2,1")),
                         Tensor(Tensor("2,0,0", "2,0,1"), Tensor("2,1,0", "2,1,1"), Tensor("2,2,0", "2,2,1")))
    assert(tensor7.dataType === TFString)
    assert(tensor7.shape === Shape(3, 3, 2))
    assert(tensor7(0, 0, 0).scalar === "0,0,0")
    assert(tensor7(0, 0, 1).scalar === "0,0,1")
    assert(tensor7(0, 1, 0).scalar === "0,1,0")
    assert(tensor7(0, 1, 1).scalar === "0,1,1")
    assert(tensor7(0, 2, 0).scalar === "0,2,0")
    assert(tensor7(0, 2, 1).scalar === "0,2,1")
    assert(tensor7(2, 0, 0).scalar === "2,0,0")
    assert(tensor7(2, 0, 1).scalar === "2,0,1")
    assert(tensor7(2, 1, 0).scalar === "2,1,0")
    assert(tensor7(2, 1, 1).scalar === "2,1,1")
    assert(tensor7(2, 2, 0).scalar === "2,2,0")
    assert(tensor7(2, 2, 1).scalar === "2,2,1")
  }

  it must "create a valid string tensors when provided no data type or shape" in {
    val tensor1: Tensor = "foo"
    assert(tensor1.dataType === TFString)
    assert(tensor1.shape === Shape())
    assert(tensor1(0).scalar === "foo")
    val tensor2: Tensor = Tensor("foo", "bar")
    assert(tensor2.dataType === TFString)
    assert(tensor2.shape === Shape(2))
    assert(tensor2(0).scalar === "foo")
    assert(tensor2(1).scalar === "bar")
    val tensor3 = Tensor(Tensor(Tensor("0,0,0", "0,0,1"), Tensor("0,1,0", "0,1,1"), Tensor("0,2,0", "0,2,1")),
                         Tensor(Tensor("1,0,0", "1,0,1"), Tensor("1,1,0", "1,1,1"), Tensor("1,2,0", "1,2,1")),
                         Tensor(Tensor("2,0,0", "2,0,1"), Tensor("2,1,0", "2,1,1"), Tensor("2,2,0", "2,2,1")))
    assert(tensor3.dataType === TFString)
    assert(tensor3.shape === Shape(3, 3, 2))
    assert(tensor3(0, 0, 0).scalar === "0,0,0")
    assert(tensor3(0, 0, 1).scalar === "0,0,1")
    assert(tensor3(0, 1, 0).scalar === "0,1,0")
    assert(tensor3(0, 1, 1).scalar === "0,1,1")
    assert(tensor3(0, 2, 0).scalar === "0,2,0")
    assert(tensor3(0, 2, 1).scalar === "0,2,1")
    assert(tensor3(2, 0, 0).scalar === "2,0,0")
    assert(tensor3(2, 0, 1).scalar === "2,0,1")
    assert(tensor3(2, 1, 0).scalar === "2,1,0")
    assert(tensor3(2, 1, 1).scalar === "2,1,1")
    assert(tensor3(2, 2, 0).scalar === "2,2,0")
    assert(tensor3(2, 2, 1).scalar === "2,2,1")
  }

  it must "not compile when invalid Scala data types are used for its arguments" in {
    assertDoesNotCompile("val tensor: Tensor = Tensor(5.asInstanceOf[Any])")
  }

  "'Tensor.update'" must "work for arbitrary indexing sequences" in {
    val tensor1: Tensor = -2
    assert(tensor1.dataType === TFInt32)
    assert(tensor1.shape === Shape())
    assert(tensor1(0).scalar === -2)
    tensor1(0).set(-5)
    assert(tensor1(0).scalar === -5)
    val tensor2 = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    assert(tensor2.dataType === TFInt32)
    assert(tensor2.shape === Shape(3, 3, 2))
    assert(tensor2(1, 1, 1).scalar === -5)
    tensor2(0, 0, ::).set(Tensor(-4, -9)(NewAxis, NewAxis, ::))
    assert(tensor2(0, 0, 0).scalar === -4)
    assert(tensor2(0, 0, 1).scalar === -9)
    tensor2(1 ::, 1, ::).set(Tensor(Tensor(-4, 5), Tensor(2, 4))(::, NewAxis, ::))
    assert(tensor2(1, 1, 0).scalar === -4)
    assert(tensor2(1, 1, 1).scalar === 5)
    assert(tensor2(2, 1, 0).scalar === 2)
    assert(tensor2(2, 1, 1).scalar === 4)
  }

  //  it must "create a valid Tensor when a data type is provided but no shape" in {
  //    val array1 = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
  //                       Array(Array(1, 23), Array(4, -5), Array(7, 9)),
  //                       Array(Array(56, 1), Array(-2, -4), Array(-7, -9)))
  //    val tensor1 = Tensor.create(array1, dataType = DataType.Int32)
  //    assert(tensor1.dataType === DataType.Int32)
  //    assert(tensor1.shape === Shape(3, 3, 2))
  //    assert(tensor1(1, 1, 1) === -5)
  //    val tensor2 = Tensor.create(array1, dataType = DataType.Float64)
  //    assert(tensor2.dataType === DataType.Float64)
  //    assert(tensor2.shape === Shape(3, 3, 2))
  //    assert(tensor2(1, 1, 1) === -5.0)
  //  }
  //
  //  "'Tensor.apply'" must "work for obtaining individual elements for all data types except 'DataType.String'" in {
  //    val array = Array(Array(2, 3), Array(0, 0), Array(5, 7))
  //    val tensor = Tensor.create(array)
  //    assert(tensor(0, 0) == array(0)(0))
  //    assert(tensor(0, 1) == array(0)(1))
  //    assert(tensor(1, 0) == array(1)(0))
  //    assert(tensor(1, 1) == array(1)(1))
  //    assert(tensor(2, 0) == array(2)(0))
  //    assert(tensor(2, 1) == array(2)(1))
  //    assert(tensor(-2, -1) == array(0)(0))
  //    assert(tensor(0, -1) == array(0)(0))
  //    assert(tensor(-1, 0) == array(1)(0))
  //    assert(tensor(-1, -1) == array(1)(0))
  //    assert(tensor(2, -1) == array(2)(0))
  //  }
  //
  //  "'TensorSlice'" must "always have the correct shape" in {
  //    val array = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
  //                      Array(Array(1, 23), Array(4, -5), Array(7, 9)),
  //                      Array(Array(56, 1), Array(-2, -4), Array(-7, -9)))
  //    // TODO: Move this to the shape specification.
  //    val tensor = Tensor.create(array)
  //    assert(tensor.shape === Shape(3, 3, 2))
  //    assert(tensor.slice(::, NewAxis, 1 :: 3, 0 :: -1).shape === Shape(3, 1, 2, 1))
  //    assert(tensor.slice(---).shape === Shape(3, 3, 2))
  //    assert(tensor.slice(::, ---).shape === Shape(3, 3, 2))
  //    assert(tensor.slice(---, -1).shape === Shape(3, 3, 1))
  //    assert(tensor.slice(---, 0 ::).shape === Shape(3, 3, 2))
  //    assert(tensor.slice(---, NewAxis).shape === Shape(3, 3, 2, 1))
  //    assert(tensor.slice(::, ---, NewAxis).shape === Shape(3, 3, 2, 1))
  //    assert(tensor.slice(---, -1, NewAxis).shape === Shape(3, 3, 1, 1))
  //    assert(tensor.slice(---, 0 ::, NewAxis).shape === Shape(3, 3, 2, 1))
  //  }
  //
  //  "'Tensor.apply'" must "work for obtaining slices for all data types except 'DataType.String'" in {
  //    val array = Array(Array(Array(2, 3), Array(0, 0), Array(5, 7)),
  //                      Array(Array(1, 23), Array(4, -5), Array(7, 9)),
  //                      Array(Array(56, 1), Array(-2, -4), Array(-7, -9))) // [3, 3, 2] array
  //    val tensor = Tensor.create(array)
  //    val tensorSlice1 = tensor.slice(::, NewAxis, 1 :: 3, 0 :: -1)
  //    assert(tensorSlice1(0, 0, 0, 0) == array(0)(1)(0))
  //    assert(tensorSlice1(0, 0, 1, 0) == array(0)(2)(0))
  //    assert(tensorSlice1(1, 0, 0, 0) == array(1)(1)(0))
  //    assert(tensorSlice1(1, 0, 1, 0) == array(1)(2)(0))
  //    val tensorSlice2 = tensor.slice(-1, ::, NewAxis, NewAxis, 0)
  //    assert(tensorSlice2(0, 0, 0, 0, 0) == array(1)(0)(0))
  //    assert(tensorSlice2(0, 1, 0, 0, 0) == array(1)(1)(0))
  //    assert(tensorSlice2(0, 2, 0, 0, 0) == array(1)(2)(0))
  //    val tensorSlice3 = tensor.slice(---, NewAxis, NewAxis, 0)
  //    assert(tensorSlice3(0, 0, 0, 0, 0) == array(0)(0)(0))
  //    assert(tensorSlice3(0, 1, 0, 0, 0) == array(0)(1)(0))
  //    assert(tensorSlice3(0, 2, 0, 0, 0) == array(0)(2)(0))
  //    assert(tensorSlice3(1, 0, 0, 0, 0) == array(1)(0)(0))
  //    assert(tensorSlice3(1, 1, 0, 0, 0) == array(1)(1)(0))
  //    assert(tensorSlice3(1, 2, 0, 0, 0) == array(1)(2)(0))
  //  }
}
