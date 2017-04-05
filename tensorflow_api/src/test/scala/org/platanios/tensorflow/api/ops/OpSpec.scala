package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Exception.IllegalNameException
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.ops.ArrayOps.constant
import org.platanios.tensorflow.api.ops.Op._
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class OpSpec extends FlatSpec with Matchers {
  "The 'usingGraph' function" must "change the default graph (only) for its code block" in {
    val graph1 = Graph()
    val graph2 = Graph()
    usingGraph(graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    usingGraph(graph2) {
      val graph2Constant1 = constant(2.0)
      assert(graph2Constant1.graph === graph2)
      usingGraph(graph1) {
        val graph1NestedConstant = constant(3.0)
        assert(graph1NestedConstant.graph === graph1)
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
    }
  }

  "The 'usingNameScope' function" must "change the name scope for newly created ops (only) for its code block" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      val c1 = constant(2.0, name = "C_1")
      assert(c1.op.name === "C_1")
      usingNameScope("Nested") {
        val nestedC = constant(3.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
      }
      val c2 = constant(4.0, name = "C_2")
      assert(c2.op.name === "C_2")
    }
  }

  it must "allow for creation of nested name scopes" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      usingNameScope("Nested") {
        val nestedC = constant(2.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
        usingNameScope("Inner") {
          val nestedInnerC = constant(3.0, name = "C")
          assert(nestedInnerC.op.name === "Nested/Inner/C")
        }
        usingNameScope("Inner_1") {
          val nestedInner1C = constant(4.0, name = "C")
          assert(nestedInner1C.op.name === "Nested/Inner_1/C")
        }
      }
    }
  }

  it must "reset the name scope when provided an empty string or 'null'" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      usingNameScope("Nested") {
        val nestedC = constant(2.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
        usingNameScope("Inner") {
          val nestedInnerC1 = constant(3.0, name = "C_1")
          assert(nestedInnerC1.op.name === "Nested/Inner/C_1")
          usingNameScope("") {
            val c1 = constant(4.0, name = "C_1")
            assert(c1.op.name === "C_1")
          }
          val nestedInnerC2 = constant(5.0, name = "C_2")
          assert(nestedInnerC2.op.name == "Nested/Inner/C_2")
          usingNameScope(null) {
            val c2 = constant(6.0, name = "C_2")
            assert(c2.op.name === "C_2")
          }
        }
      }
    }
  }

  "Ops created using the same name" must "have their name made unique by appending an index to it" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      val c1 = constant(2.0, name = "C")
      assert(c1.op.name === "C_1")
      val c2 = constant(3.0, name = "C")
      assert(c2.op.name === "C_2")
      val c3 = constant(4.0, name = "C_3")
      assert(c3.op.name === "C_3")
      val c4 = constant(5.0, name = "C")
      assert(c4.op.name === "C_4")
    }
  }

  "An 'IllegalNameException'" must "be thrown when invalid characters are used in an op's name" in {
    usingGraph(Graph()) {
      assertThrows[IllegalNameException](constant(1.0, name = "C!"))
      assertThrows[IllegalNameException](constant(1.0, name = "_C"))
      assertThrows[IllegalNameException](constant(1.0, name = "\\C"))
      assertThrows[IllegalNameException](constant(1.0, name = "-C"))
      assertThrows[IllegalNameException](constant(1.0, name = "/C"))
    }
  }

  // TODO: Add name scope exceptions spec.
}
