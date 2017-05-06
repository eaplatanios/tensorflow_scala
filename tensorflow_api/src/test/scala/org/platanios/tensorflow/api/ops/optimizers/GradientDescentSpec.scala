package org.platanios.tensorflow.api.ops.optimizers

import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor}
import org.platanios.tensorflow.api.ops.{Basic, Op, Variable}
import org.platanios.tensorflow.api.types.{DataType, Float64}

import org.scalactic.Equality
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientDescentSpec extends FlatSpec with Matchers {
  implicit val doubleEquality: Equality[Float64] = new Equality[Float64] {
    val tolerance: Double = 1e-6

    def areEqual(a: Float64, b: Any): Boolean = {
      b match {
        case bDouble: Float64 => (a <= bDouble + tolerance) && (a >= bDouble - tolerance)
        case _ => false
      }
    }
    override def toString: String = s"TolerantDoubleEquality($tolerance)"
  }

  "Gradient descent" must "work for dense updates to resource-based variables" in {
    for (dataType <- Set[DataType](DataType.Float32, DataType.Float64)) {
      val graph = Graph()
      val (variable0, variable1, gdOp) = Op.createWith(graph) {
        val variable0 = Variable(Variable.ConstantInitializer(Tensor(1, 2)), shape = Shape(2), dataType = dataType)
        val variable1 = Variable(Variable.ConstantInitializer(Tensor(3, 4)), shape = Shape(2), dataType = dataType)
        val gradient0 = Basic.constant(Tensor(0.1, 0.1), dataType = dataType)
        val gradient1 = Basic.constant(Tensor(0.01, 0.01), dataType = dataType)
        val gdOp = GradientDescent(3.0).applyGradients(Seq((gradient0, variable0), (gradient1, variable1)))
        (variable0, variable1, gdOp)
      }
      val session = Session(graph)
      session.run(targets = Array(Variable.initializer(graph.trainableVariables)))
      var variable0Value = session.run(fetches = Array(variable0.value)).head
      var variable1Value = session.run(fetches = Array(variable1.value)).head
      assert(variable0Value(0).scalar === 1.0)
      assert(variable0Value(1).scalar === 2.0)
      assert(variable1Value(0).scalar === 3.0)
      assert(variable1Value(1).scalar === 4.0)
      session.run(targets = Array(gdOp))
      variable0Value = session.run(fetches = Array(variable0.value)).head
      variable1Value = session.run(fetches = Array(variable1.value)).head
      assert(variable0Value(0).scalar === 1.0 - 3.0 * 0.1)
      assert(variable0Value(1).scalar === 2.0 - 3.0 * 0.1)
      assert(variable1Value(0).scalar === 3.0 - 3.0 * 0.01)
      assert(variable1Value(1).scalar === 4.0 - 3.0 * 0.01)
    }
  }
}
