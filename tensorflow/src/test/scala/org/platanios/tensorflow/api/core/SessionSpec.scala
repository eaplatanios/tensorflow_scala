package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.ops.Basic._
import org.platanios.tensorflow.api.ops.Math._
import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.tf.createWith

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
    val fetches = graph.getOpOutputByName("Y:0")
    val output = session.run(feeds, fetches)
    val expectedResult = tf.Tensor(tf.Tensor(-30))
    assert(output.scalar === expectedResult.scalar)
    graph.close()
  }
}
