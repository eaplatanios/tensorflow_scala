package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.{DataType, Shape, Tensor}

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class ArrayOpsSpec extends FlatSpec with Matchers {
//  "'ArrayOps.constant'" must "create a constant op when provided a Tensor of the same data type and shape" in {
//    // DataType.Int32 Tensor
//    val tensor1 = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
//                         Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
//                         Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
//    val constant1 = ArrayOps.constant(tensor1)
//    val constantValue1 = constant1.value()
//    assert(tensor1.get(1, 1, 1) === -5)
//    assert(constant1.shape === Shape(3, 3, 2))
//    assert(constant1.dataType === DataType.Int32)
//    assert(constantValue1.shape === Shape(3, 3, 2))
//    assert(constantValue1.dataType === DataType.Int32)
//    assert(constantValue1(1, 1, 1) === -5)
//    tensor1.close()
//
//    // DataType.Float64 Tensor
//    val tensor2 = Tensor.create(array, dataType = DataType.Float64)
//    val constant2 = ArrayOps.constant(tensor2)
//    val constantValue2 = constant2.value()
//    assert(tensor2(1, 1, 1) === -5.0)
//    assert(constant2.shape === Shape(3, 3, 2))
//    assert(constant2.dataType === DataType.Float64)
//    assert(constantValue2.shape === Shape(3, 3, 2))
//    assert(constantValue2.dataType === DataType.Float64)
//    assert(constantValue2(1, 1, 1) === -5.0)
//    tensor2.close()
//  }
}
