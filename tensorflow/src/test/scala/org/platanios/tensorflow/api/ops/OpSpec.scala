package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Math.matMul
import org.platanios.tensorflow.api.tf._

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class OpSpec extends FlatSpec with Matchers {
  "'Op.Output.setShape'" must "always work" in {
    createWith(graph = Graph()) {
      val a = placeholder(INT32, Shape(-1, -1, 3))
      assert(!a.shape.isFullyDefined)
      assert(a.shape === Shape(-1, -1, 3))
      a.setShape(Shape(2, 4, 3))
      assert(a.shape.isFullyDefined)
      assert(a.shape === Shape(2, 4, 3))
    }
  }

  //region createWith(...) Specification

  "The 'createWith' function" must "change the default graph (only) for its code block" in {
    val graph1 = Graph()
    val graph2 = Graph()
    createWith(graph = graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    createWith(graph = graph2) {
      val graph2Constant1 = constant(2.0)
      assert(graph2Constant1.graph === graph2)
      createWith(graph = graph1) {
        val graph1NestedConstant = constant(3.0)
        assert(graph1NestedConstant.graph === graph1)
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
    }
  }

  //region createWith(nameScope = ...) Specification

  it must "change the name scope for newly created ops (only) for its code block" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      val c1 = constant(2.0, name = "C_1")
      assert(c1.op.name === "C_1")
      createWith(nameScope = "Nested") {
        val nestedC = constant(3.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
      }
      val c2 = constant(4.0, name = "C_2")
      assert(c2.op.name === "C_2")
    }
  }

  it must "allow for creation of nested name scopes" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      createWith(nameScope = "Nested") {
        val nestedC = constant(2.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
        createWith(nameScope = "Inner") {
          val nestedInnerC = constant(3.0, name = "C")
          assert(nestedInnerC.op.name === "Nested/Inner/C")
        }
        createWith(nameScope = "Inner_1") {
          val nestedInner1C = constant(4.0, name = "C")
          assert(nestedInner1C.op.name === "Nested/Inner_1/C")
        }
      }
    }
  }

  it must "reset the name scope when provided an empty string" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      createWith(nameScope = "Nested") {
        val nameScope = Op.currentNameScope
        val nestedC1 = constant(2.0, name = "C_1")
        assert(nestedC1.op.name === "Nested/C_1")
        createWith(nameScope = "Inner") {
          val nestedInnerC = constant(3.0, name = "C_1")
          assert(nestedInnerC.op.name === "Nested/Inner/C_1")
          createWith(nameScope = nameScope) {
            val nestedC2 = constant(5.0, name = "C_2")
            assert(nestedC2.op.name === "Nested/C_2")
            createWith(nameScope = "") {
              val c1 = constant(4.0, name = "C_1")
              assert(c1.op.name === "C_1")
            }
          }
        }
      }
    }
  }

  //endregion createWith(nameScope = ...) Specification

  //region createWith(device = ...) Specification

  it must "change the device in which newly created ops from its code block (only) are placed" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/CPU:0") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:CPU:0")
        createWith(device = "/GPU:0") {
          val c3 = constant(3.0)
          assert(c3.device === "/device:GPU:0")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "allow for nesting of device scopes" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/job:worker/replica:2") {
        val c2 = constant(2.0)
        assert(c2.device === "/job:worker/replica:2")
        createWith(device = "/job:worker/replica:3/task:0") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/replica:3/task:0")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "nest device scopes by appropriately overriding device specifications" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/job:worker/replica:2/device:CPU:1") {
        val c2 = constant(2.0)
        assert(c2.device === "/job:worker/replica:2/device:CPU:1")
        createWith(device = "/job:worker/replica:2/device:GPU:2") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/replica:2/device:GPU:2")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "nest device scopes by appropriately merging device specifications" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/device:GPU:*") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:GPU:*")
        createWith(device = "/job:worker") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/device:GPU:*")
          createWith(device = "/device:CPU:0") {
            val c4 = constant(4.0)
            assert(c4.device === "/job:worker/device:CPU:0")
            createWith(device = "/job:ps") {
              val c5 = constant(5.0)
              assert(c5.device === "/job:ps/device:CPU:0")
            }
          }
        }
        createWith(device = "/device:GPU:5") {
          val c6 = constant(6.0)
          assert(c6.device === "/device:GPU:5")
        }
      }
      val c7 = constant(7.0)
      assert(c7.device === "")
    }
  }

  it must "reset the device whenever an empty string is provided for its device argument" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0, name = "C_1")
      assert(c1.device === "")
      createWith(device = "/CPU:0") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:CPU:0")
        createWith(device = null) {
          val c3 = constant(3.0)
          assert(c3.device === "")
        }
      }
    }
  }

  it must "be able to use device functions for setting op devices individually" in {
    def matMulOnGPU(opSpecification: OpSpecification): String = {
      if (opSpecification.opType == "MatMul")
        "/GPU:0"
      else
        "/CPU:0"
    }
    createWith(graph = Graph()) {
      createWith(device = matMulOnGPU) {
        val c = constant(1.0)
        assert(c.device === "/device:CPU:0")
        val m = matMul(c, constant(2.0))
        assert(m.device === "/device:GPU:0")
      }
    }
  }

  //endregion createWith(device = ...) Specification

  //region createWith(colocationOps = ...) Specification

  it must "be able to colocate ops" in {
    createWith(graph = Graph()) {
      val a = createWith(device = "/CPU:0")(constant(1.0))
      val b = createWith(device = "/GPU:0")(constant(1.0))
      assert(a.colocationOps === Set.empty[Op])
      assert(b.colocationOps === Set.empty[Op])
      val c = createWith(colocationOps = Set(a))(constant(1.0))
      assert(c.colocationOps === Set[Op](a))
      createWith(colocationOps = Set[Op](b)) {
        val d = constant(1.0)
        assert(d.colocationOps === Set[Op](b))
        createWith(colocationOps = Set[Op](a, d)) {
          val e = constant(1.0)
          assert(e.colocationOps === Set[Op](a, b, d))
          createWith(colocationOps = Set.empty[Op]) {
            val f = constant(1.0)
            assert(f.colocationOps === Set.empty[Op])
          }
        }
      }
    }
  }

  //endregion createWith(colocationOps = ...) Specification

  //region createWith(controlDependencies = ...) Specification

  it must "change the control dependencies for newly created ops (only) for its code block" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs === Set.empty[Op])
      assert(a.controlOutputs === Set.empty[Op])
      assert(b.controlInputs === Set.empty[Op])
      assert(b.controlOutputs === Set.empty[Op])
      val c = createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs === Set[Op](a))
        assert(c.controlOutputs === Set.empty[Op])
        assert(a.controlOutputs === Set[Op](c))
        assert(b.controlOutputs === Set.empty[Op])
        c
      }
      createWith(controlDependencies = Set[Op](a, b)) {
        val d = constant(1.0)
        assert(d.controlInputs === Set[Op](a, b))
        assert(d.controlOutputs === Set.empty[Op])
        assert(a.controlOutputs === Set[Op](c, d))
        assert(b.controlOutputs === Set[Op](d))
      }
    }
  }

  it must "allow for nesting of control dependencies specifications" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs === Set.empty[Op])
      assert(a.controlOutputs === Set.empty[Op])
      assert(b.controlInputs === Set.empty[Op])
      assert(b.controlOutputs === Set.empty[Op])
      createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs === Set[Op](a))
        assert(c.controlOutputs === Set.empty[Op])
        assert(a.controlOutputs === Set[Op](c))
        assert(b.controlOutputs === Set.empty[Op])
        createWith(controlDependencies = Set[Op](b)) {
          val d = constant(1.0)
          assert(d.controlInputs === Set[Op](a, b))
          assert(d.controlOutputs === Set.empty[Op])
          assert(a.controlOutputs === Set[Op](c, d))
          assert(b.controlOutputs === Set[Op](d))
        }
      }
    }
  }

  it must "reset the control dependencies (only) for its code block when provided an empty set" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs === Set.empty[Op])
      assert(a.controlOutputs === Set.empty[Op])
      assert(b.controlInputs === Set.empty[Op])
      assert(b.controlOutputs === Set.empty[Op])
      createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs === Set[Op](a))
        assert(c.controlOutputs === Set.empty[Op])
        assert(a.controlOutputs === Set[Op](c))
        assert(b.controlOutputs === Set.empty[Op])
        createWith(controlDependencies = Set.empty[Op]) {
          val d = constant(1.0)
          assert(d.controlInputs === Set.empty[Op])
          assert(d.controlOutputs === Set.empty[Op])
          assert(a.controlOutputs === Set[Op](c))
          assert(b.controlOutputs === Set.empty[Op])
          createWith(controlDependencies = Set[Op](b)) {
            val e = constant(1.0)
            assert(e.controlInputs === Set[Op](b))
            assert(e.controlOutputs === Set.empty[Op])
            assert(a.controlOutputs === Set[Op](c))
            assert(b.controlOutputs === Set[Op](e))
          }
        }
      }
    }
  }

  //endregion createWith(controlDependencies = ...) Specification

  //region createWith(attributes = ...) Specification

  it must "change the attributes for newly created ops (only) for its code block" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      assert(intercept[IllegalArgumentException](a.stringAttribute("_a")).getMessage ===
                 "Op has no attribute named '_a'. " +
                     "TensorFlow native library error message: Operation has no attr named '_a'.")
      createWith(attributes = Map("_a" -> "foo")) {
        val b = constant(1.0)
        assert(b.stringAttribute("_a") === "foo")
        createWith(attributes = Map("_a" -> "bar")) {
          val c = constant(1.0)
          assert(c.stringAttribute("_a") === "bar")
          createWith(attributes = Map("_a" -> null)) {
            val d = constant(1.0)
            assert(intercept[IllegalArgumentException](d.stringAttribute("_a")).getMessage ===
                       "Op has no attribute named '_a'. " +
                           "TensorFlow native library error message: Operation has no attr named '_a'.")
          }
        }
      }
    }
  }

  //endregion createWith(attributes = ...) Specification

  // TODO: [VARIABLE] Add "createWtih(container = ...)" specification.

  it must "allow changing, the graph, the name scope, and the device used for its code block simultaneously" in {
    val graph1 = Graph()
    val graph2 = Graph()
    createWith(graph = graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    createWith(graph = graph2, nameScope = "Nested", device = "/GPU:0") {
      val graph2Constant1 = constant(2.0, name = "C")
      assert(graph2Constant1.graph === graph2)
      assert(graph2Constant1.op.name === "Nested/C")
      assert(graph2Constant1.device === "/device:GPU:0")
      createWith(graph = graph1, nameScope = "Inner") {
        val graph1NestedConstant = constant(3.0, name = "C")
        assert(graph1NestedConstant.graph === graph1)
        assert(graph1NestedConstant.op.name === "Nested/Inner/C")
        assert(graph1NestedConstant.device === "/device:GPU:0")
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
      assert(graph2Constant2.device === "/device:GPU:0")
    }
  }

  //endregion createWith(...) Specification

  "Ops created using the same name" must "have their name made unique by appending an index to it" in {
    createWith(graph = Graph()) {
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
    createWith(graph = Graph()) {
      assertThrows[IllegalNameException](constant(1.0, name = "C!"))
      assertThrows[IllegalNameException](constant(1.0, name = "_C"))
      assertThrows[IllegalNameException](constant(1.0, name = "\\C"))
      assertThrows[IllegalNameException](constant(1.0, name = "-C"))
      assertThrows[IllegalNameException](constant(1.0, name = "/C"))
    }
  }

  // TODO: Add name scope exceptions spec.

  "'stripNameScope' and 'prependNameScope'" must "work correctly whether or not 'loc:@' is used" in {
    val names = Array[String](
      "hidden1/hidden1/weights", // Same prefix. Should strip.
      "hidden1///hidden1/weights", // Extra '/'. Should strip.
      "^hidden1/hidden1/weights", // Same prefix. Should strip.
      "loc:@hidden1/hidden1/weights", // Same prefix. Should strip.
      "hhidden1/hidden1/weights", // Different prefix. Should keep.
      "hidden1" // Not a prefix. Should keep.
    )
    val expectedStripedNames = Array[String](
      "hidden1/weights", "hidden1/weights", "^hidden1/weights", "loc:@hidden1/weights", "hhidden1/hidden1/weights",
      "hidden1")
    val expectedPrependedNames = Array[String](
      "hidden2/hidden1/weights", "hidden2/hidden1/weights", "^hidden2/hidden1/weights", "loc:@hidden2/hidden1/weights",
      "hidden2/hhidden1/hidden1/weights", "hidden2/hidden1")
    val nameScopeToStrip = "hidden1"
    val nameScopeToPrepend = "hidden2"
    (names, expectedStripedNames, expectedPrependedNames).zipped
        .foreach((name, expectedStripedName, expectedPrependedName) => {
          val strippedName = Op.stripNameScope(nameScope = nameScopeToStrip, name = name)
          val prependedName = Op.prependNameScope(nameScope = nameScopeToPrepend, name = strippedName)
          assert(strippedName === expectedStripedName)
          assert(prependedName === expectedPrependedName)
        })
  }

  //  "'Op.OutputIndexedSlices'" must "be convertible to 'Op.Output'" in {
  //    createWith(graph = Graph()) {
  //      val values = ArrayOps.constant(Tensor(Tensor(2, 3), Tensor(5, 7)))
  //      val indices = ArrayOps.constant(Tensor(0, 2))
  //      val denseShape = ArrayOps.constant(Tensor(3, 2))
  //      val indexedSlices = Op.OutputIndexedSlices(values, indices, denseShape)
  //      // TODO: Simplify this after we standardize our tensor interface.
  //      val resultTensor = indexedSlices.toOpOutput().value()
  //      val resultArray = Array.ofDim[Int](resultTensor.shape(0).asInstanceOf[Int],
  //                                         resultTensor.shape(1).asInstanceOf[Int])
  //      resultTensor.copyTo(resultArray)
  //      assert(resultArray === Array(Array(2, 3), Array(0, 0), Array(5, 7)))
  //    }
  //  }

  //    def testToTensor(self):
  //    with self.test_session():
  //        values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
  //    indices = constant_op.constant([0, 2])
  //    dense_shape = constant_op.constant([3, 2])
  //    x = ops.IndexedSlices(values, indices, dense_shape)
  //    tensor = ops.convert_to_tensor(x, name="tensor")
  //    self.assertAllEqual(tensor.eval(), [[2, 3], [0, 0], [5, 7]])
  //
  //    def testNegation(self):
  //    with self.test_session():
  //        values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
  //    indices = constant_op.constant([0, 2])
  //    x = -ops.IndexedSlices(values, indices)
  //    self.assertAllEqual(x.values.eval(), [[-2, -3], [-5, -7]])
  //    self.assertAllEqual(x.indices.eval(), [0, 2])
  //
  //    def testScalarMul(self):
  //    with self.test_session():
  //        values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
  //    indices = constant_op.constant([0, 2])
  //    x = math_ops.scalar_mul(-2, ops.IndexedSlices(values, indices))
  //    self.assertAllEqual(x.values.eval(), [[-4, -6], [-10, -14]])
  //    self.assertAllEqual(x.indices.eval(), [0, 2])
}
