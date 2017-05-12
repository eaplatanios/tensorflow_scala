package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.Basic._
import org.platanios.tensorflow.api.ops.Math._
import org.platanios.tensorflow.api.ops.Op.createWith

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SessionSpec extends FlatSpec with Matchers {
  "Session run fetch by name" should "return the correct result" in {
    val graph = Graph()
    createWith(graph = graph) {
      val a = constant(tf.Tensor(tf.Tensor(2, 3)), name = "A")
      val x = placeholder(dataType = tf.INT32, name = "X")
      subtract(constant(1), matMul(a = a, b = x, transposeB = true), name = "Y")
    }
    val session = Session(graph = graph)
    val feeds = Map(graph.getOpOutputByName("X:0") -> tf.Tensor(tf.Tensor(5, 7)))
    val fetches = Array(graph.getOpOutputByName("Y:0"))
    val outputs = session.run(feeds, fetches)
    assert(outputs.size == 1)
    val expectedResult = tf.Tensor(tf.Tensor(-30))
    assert(outputs.head.scalar === expectedResult.scalar)
    graph.close()
  }
}
