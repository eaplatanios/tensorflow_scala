package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.ArrayOps._
import org.platanios.tensorflow.api.ops.MathOps._
import org.platanios.tensorflow.api.ops.Op.createWith

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SessionSpec extends FlatSpec with Matchers {
  "Session run fetch by name" should "return the correct result" in {
    val graph = Graph()
    createWith(graph = graph) {
      val a = constant(Tensor(Tensor(2, 3)), name = "A")
      val x = placeholder(dataType = DataType.Int32, name = "X")
      subtract(constant(1), matMul(a = a, b = x, transposeB = true), name = "Y")
    }
    val session = Session(graph = graph)
    val feeds = Map(graph.opOutputByName("X:0") -> Tensor(Tensor(5, 7)))
    val fetches = Array(graph.opOutputByName("Y:0"))
    val outputs = session.run(feeds, fetches)
    assert(outputs.size == 1)
    val expectedResult = Tensor(Tensor(-30))
    assert(outputs.head.scalar === expectedResult.scalar)
    graph.close()
  }
}
