package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.ArrayOps._
import org.platanios.tensorflow.api.ops.MathOps._
import org.platanios.tensorflow.api.ops.Op.usingGraph
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SessionSpec extends FlatSpec with Matchers {
  "Session run fetch by name" should "return the correct result" in {
    val graph = Graph()
    usingGraph(graph) {
      val a = constant(Array[Array[Int]](Array[Int](2), Array[Int](3)))
      val x = placeholder(dataType = DataType.int32, name = "X")
      subtract(constant(1), matMul(a = a, b = x, transposeA = true), name = "Y")
    }
    val session = Session(graph = graph)
    using(Tensor.create(Array[Array[Int]](Array[Int](5), Array[Int](7)))) { x =>
      val outputs = session.Runner().feed(opName = "X", tensor = x).fetch(opName = "Y").run()._1
      assert(outputs.size == 1)
      val expectedResult = Array[Array[Int]](Array[Int](-30))
      outputs.head.copyTo(Array.ofDim[Int](1, 1)) should equal(expectedResult)
    }
    graph.close()
  }
}
