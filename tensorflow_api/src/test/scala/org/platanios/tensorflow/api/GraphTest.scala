package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.ArrayOps.constant
import org.platanios.tensorflow.api.ops.Op.createWith
import org.scalatest.{FlatSpec, Matchers}

/**
  * @author Emmanouil Antonios Platanios
  */
class GraphTest extends FlatSpec with Matchers {
  "'allOps'" must "return all the ops in a graph" in {
    val graph = Graph()
    val allOps = createWith(graph = graph) {
      val c1 = constant(1.0, name = "C_1")
      val c2 = constant(2.0, name = "C_2")
      val c3 = createWith(nameScope = "Nested") {
        constant(3.0, name = "C_3")
      }
      val c4 = constant(4.0, name = "C_4")
      Array(c1.op, c2.op, c3.op, c4.op)
    }
    assert(graph.allOps === allOps)
  }
}
