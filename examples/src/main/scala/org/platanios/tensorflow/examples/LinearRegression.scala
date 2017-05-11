package org.platanios.tensorflow.examples

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Basic.placeholder
import org.platanios.tensorflow.api.ops.Math.{matMul, reduceSum, square}
import org.platanios.tensorflow.api.ops.optimizers.GradientDescent
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
object LinearRegression {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))
  private[this] val random = new Random()

  private[this] val weight = random.nextFloat()

  def main(args: Array[String]): Unit = {
    logger.info("Building linear regression model.")
    val graph = Graph()
    val (inputs, outputs, weights, loss, trainOp) = Op.createWith(graph) {
      val optimizer = GradientDescent(0.0001)
      val inputs = placeholder(TFFloat32, Shape(-1, 1))
      val outputs = placeholder(TFFloat32, Shape(-1, 1))
      val initialWeights = Tensor.fill(TFFloat32, Shape(1, 1))(0.0f)
      val weights = Variable(Variable.ConstantInitializer(initialWeights), Shape(1, 1), TFFloat32)
      val predictions = matMul(inputs, weights)
      val loss = reduceSum(square(predictions - outputs))
      val trainOp = optimizer.minimize(loss)
      (inputs, outputs, weights, loss, trainOp)
    }

    logger.info("Training the linear regression model.")
    val session = Session(graph)
    session.run(targets = Array(Variable.initializer(graph.trainableVariables)))
    for (i <- 0 to 50) {
      val trainBatch = batch(10000)
      val feeds = Map[Op.Output, Tensor](
        inputs -> trainBatch._1,
        outputs -> trainBatch._2)
      val fetches = Array[Op.Output](loss)
      val targets = Array[Op](trainOp)
      val trainLoss = session.run(feeds, fetches, targets)(0)
      if (i % 1 == 0)
        logger.info(s"Train loss at iteration $i = ${trainLoss.scalar} " +
                        s"(weight = ${session.run(fetches = Array(weights.value))(0).scalar})")
    }

    logger.info(s"Trained weight value: ${session.run(fetches = Array(weights.value))(0).scalar}")
    logger.info(s"True weight value: $weight")
  }

  def batch(batchSize: Int): (Tensor, Tensor) = {
    val inputs = ArrayBuffer.empty[Float]
    val outputs = ArrayBuffer.empty[Float]
    var i = 0
    while (i < batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
      i += 1
    }
    (Tensor(inputs.map(Tensor(_)): _*), Tensor(outputs.map(Tensor(_)): _*))
  }
}
