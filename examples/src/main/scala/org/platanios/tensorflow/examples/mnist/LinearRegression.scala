package org.platanios.tensorflow.examples.mnist

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Basic.{placeholder, reshape}
import org.platanios.tensorflow.api.ops.Math.{cast, matMul, reduceSum, square}
import org.platanios.tensorflow.api.ops.optimizers.GradientDescent
import org.platanios.tensorflow.data.loaders.{MNISTDataSet, MNISTLoader}

import java.nio.file.{Files, Path, Paths}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
object LinearRegression {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / MNIST Linear Regression"))

  def main(args: Array[String]): Unit = {
    logger.info("Loading MNIST data set.")
    val dataDirectory: String       = "/Users/Anthony/Development/GitHub/tensorflow_scala/temp/data/mnist"
    val dataPath     : Path         = Files.createDirectories(Paths.get(dataDirectory))
    val dataSet      : MNISTDataSet = MNISTLoader.load(dataPath)

    logger.info("Building linear regression model.")
    val graph = Graph()
    val (imagesPlaceholder, labelsPlaceholder, loss, trainOp) = Op.createWith(graph) {
      val optimizer = GradientDescent(0.01)
      val imagesPlaceholder = placeholder(TFUInt8, Shape(-1, 28, 28))
      val labelsPlaceholder = placeholder(TFUInt8, Shape(-1))
      val images = cast(imagesPlaceholder, TFFloat32)
      val labels = cast(labelsPlaceholder, TFFloat32)
      val vectorizedImages = reshape(images, Shape(-1, 784))
      val initialWeights = Tensor.fill(TFFloat32, Shape(784))(0.0f)
      val weights = Variable(Variable.ConstantInitializer(initialWeights), Shape(784), TFFloat32)
      val predictions = matMul(vectorizedImages, weights)
      val loss = reduceSum(square(predictions - labels))
      val trainOp = optimizer.minimize(loss)
      (imagesPlaceholder, labelsPlaceholder, loss, trainOp)
    }

    logger.info("Training the linear regression model.")
    val session = Session(graph)
    for (i <- 0 to 1000) {
      val feeds = Map[Op.Output, Tensor](
        imagesPlaceholder -> dataSet.trainImages,
        labelsPlaceholder -> dataSet.trainLabels)
      val fetches = Array[Op.Output](loss)
      val targets = Array[Op](trainOp)
      val trainLoss = session.run(feeds, fetches, targets)(0)
      if (i % 10 == 0)
        logger.info(s"Train loss at iteration $i = $trainLoss")
    }

    logger.info("Evaluating linear regression model.")
    val feeds    = Map[Op.Output, Tensor](
      imagesPlaceholder -> dataSet.testImages,
      labelsPlaceholder -> dataSet.testLabels)
    val fetches  = Array[Op.Output](loss)
    val testLoss = session.run(feeds, fetches)
    logger.info(s"Test loss = $testLoss")
  }
}
