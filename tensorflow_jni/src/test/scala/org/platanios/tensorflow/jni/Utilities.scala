//package org.platanios.tensorflow
//
//import org.platanios.tensorflow.jni.{DataType, Graph, Output, Tensor}
//import org.platanios.utilities.ARM.using
//
//import scala.reflect.ClassTag
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//object Utilities {
//  def constant(graph: Graph, name: String, value: Any): Output = {
//    using(Tensor.create(value = value)) { tensor =>
//      graph.operationBuilder(opType = "Const", name = name)
//          .setAttr(name = "dtype", value = tensor.dataType)
//          .setAttr(name = "value", tensor)
//          .build()
//          .output(index = 0)
//    }
//  }
//
//  def placeholder(graph: Graph, name: String, dataType: DataType[_]): Output = {
//    graph.operationBuilder(opType = "Placeholder", name = name)
//        .setAttr(name = "dtype", value = dataType)
//        .build()
//        .output(index = 0)
//  }
//
//  def addN(graph: Graph, inputs: Output*): Output = {
//    graph.operationBuilder(opType = "AddN", name = "AddN")
//        .addInputList(inputs = inputs.toArray)
//        .build()
//        .output(index = 0)
//  }
//
//  def matMul(graph: Graph, name: String, a: Output, b: Output, transposeA: Boolean, transposeB: Boolean): Output = {
//    graph.operationBuilder(opType = "MatMul", name = name)
//        .addInput(input = a)
//        .addInput(input = b)
//        .setAttr(name = "transpose_a", value = transposeA)
//        .setAttr(name = "transpose_b", value = transposeB)
//        .build()
//        .output(index = 0)
//  }
//
//  def transposeATimesX(graph: Graph, a: Array[Array[Int]]): Output = {
//    matMul(graph = graph,
//           name = "Y",
//           a = constant(graph = graph, name = "A", value = a),
//           b = placeholder(graph = graph, name = "X", dataType = DataType.int32),
//           transposeA = true,
//           transposeB = false)
//  }
//
//  def flattenedNumElements(array: Array[_]): Int = {
//    var count: Int = 0
//    var i: Int = 0
//    while (i < array.length) {
//      if (!array(i).isInstanceOf[Array[_]])
//        count += 1
//      else
//        count += flattenedNumElements(array(i).asInstanceOf[Array[_]])
//      i += 1
//    }
//    count
//  }
//
//  def flatten[T: ClassTag](array: Array[_]): Array[T] = {
//    val outputArray: Array[T] = Array.ofDim[T](flattenedNumElements(array = array))
//    flatten[T](inputArray = array, outputArray = outputArray, next = 0)
//    outputArray
//  }
//
//  private def flatten[T](inputArray: Array[_], outputArray: Array[T], next: Int): Int = {
//    var mutableNext: Int = next
//    var i: Int = 0
//    while (i < inputArray.length) {
//      if (!inputArray(i).isInstanceOf[Array[_]]) {
//        outputArray(mutableNext) = inputArray(i).asInstanceOf[T]
//        mutableNext += 1
//      } else {
//        mutableNext = flatten[T](
//          inputArray = inputArray(i).asInstanceOf[Array[_]], outputArray = outputArray, next = mutableNext)
//      }
//      i += 1
//    }
//    mutableNext
//  }
//
//  def boolToByte(array: Array[Boolean]): Array[Byte] = {
//    val outputArray: Array[Byte] = Array.ofDim[Byte](array.length)
//    var i: Int = 0
//    while (i < array.length) {
//      if (array(i))
//        outputArray(i) = 1
//      else
//        outputArray(i) = 0
//      i += 1
//    }
//    outputArray
//  }
//
//  def byteToBool(array: Array[Byte]): Array[Boolean] = {
//    val outputArray: Array[Boolean] = Array.ofDim[Boolean](array.length)
//    var i: Int = 0
//    while (i < array.length) {
//      outputArray(i) = array(i) != 0
//      i += 1
//    }
//    outputArray
//  }
//}
