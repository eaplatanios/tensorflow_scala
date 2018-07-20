package org.platanios.tensorflow.examples.python2scala

import java.io._
import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.{FLOAT64, Op, Session, Shape, Tensor, tf}
import org.slf4j.LoggerFactory
import org.tensorflow.framework.MetaGraphDef

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Purpose: take a model trained with the Python API of Tensor Flow and restore
  * it inside a Scala environment in order to continue the training process. For
  * this example was used a virgin model but can be used a trained one without any
  * problem.
  *
  * Information, Python API side: The model was saved using tf.train.Saver().
  *
  * @author Luca Tagliabue
  */
object LinearRegressionFromRestoredPythonModel {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))
  private[this] val random = new Random(22)
  private[this] val weight = random.nextFloat()

  private[this] val checkpoint = "examples/src/main/resources/python2scala/virgin-linear-regression-pull-request"
  private[this] val meta = new File(getClass.getClassLoader.getResource("python2scala/virgin-linear-regression-pull-request.meta").getFile)
  private[this] val metaGraphDefFile = new File(getClass.getClassLoader.getResource("python2scala/MetaGraphDef.txt").getFile)

  def main(args: Array[String]): Unit = {
    val metaGraphDefInputStream = new BufferedInputStream(new FileInputStream(meta))
    val mgf = MetaGraphDef.parseFrom(metaGraphDefInputStream)
    val checkpointPath = Paths.get(checkpoint)

    // WRITE META GRAPH DEF ON TEXT FILE
    val fileWriter = new BufferedWriter(new FileWriter(metaGraphDefFile))
    fileWriter.write(mgf.toString)
    fileWriter.close()

    tf.createWith(graph = Graph()) {
      val session = Session()
      val saver = tf.Saver.fromMetaGraphDef(metaGraphDef = mgf)
      saver.restore(session, checkpointPath)

      // ----- PRINT OPERATIONS IN THE GRAPH ---------
      println(" -" * 40)
      for (op <- session.graph.ops)
        myPrintln("OPERATION name:    " + op, 76)
      println(" -" * 40)

      // RESTORE NODES and OPERATIONS
      val input = session.graph.getOutputByName("p2s_input:0")
      val output = session.graph.getOutputByName("p2s_output:0")
      val weight = session.graph.getOutputByName("p2s_weights_w/Read/ReadVariableOp:0")
      val bias = session.graph.getOutputByName("p2s_weights_b/Read/ReadVariableOp:0")
      val loss = session.graph.getOutputByName("p2s_loss:0")
      val placeholderW = session.graph.getOutputByName("placeholder_p2s_weights_w:0")
      val placeholderB = session.graph.getOutputByName("placeholder_p2s_weights_b:0")
      val gradientW = session.graph.getOutputByName("exported_gradient_p2s_weights_w:0")
      val gradientB = session.graph.getOutputByName("exported_gradient_p2s_weights_b:0")

      val trainOp = session.graph.getOpByName("p2s_train_op")
      val assignVariableW = session.graph.getOpByName("AssignOp_p2s_weights_w")
      val assignVariableB = session.graph.getOpByName("AssignOp_p2s_weights_b")

      printRestoredNodesAndOperations(session, input, output, weight, bias, loss, placeholderW, placeholderB, gradientW, gradientB, trainOp, assignVariableW, assignVariableB)

      // TRAINING LOOP
      for (i <- 0 to 50) {
        val (one, two) = batch(10000)
        val feedsMap = FeedMap(Map(input -> one, output -> two))
        val fetchesSeq = Seq(loss, weight, bias)
        val trainFetches = session.run(feeds = feedsMap, fetches = fetchesSeq, targets = trainOp)
        val trainLoss = trainFetches(0)
        val trainWeight = trainFetches(1)
        val trainBias = trainFetches(2)
        logger.info(s"Train loss at iteration $i = ${trainLoss.scalar}\nTrain weight at iteration $i = ${trainWeight.scalar}\nTrain bias at iteration $i = ${trainBias.scalar}")
      }
    }
  }


  // UTILIY METHODS
  def batch(batchSize: Int): (Tensor[FLOAT64], Tensor[FLOAT64]) = {
    val inputs = ArrayBuffer.empty[Double]
    val outputs = ArrayBuffer.empty[Double]
    for (_ <- 0 until batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
    }
    (Tensor(inputs).reshape(Shape(-1, 1)), Tensor(outputs).reshape(Shape(-1, 1)))
  }

  def printRestoredNodesAndOperations(session: Session, input: Output, output: Output, weight: Output, bias: Output, loss: Output, placeholderW: Output, placeholderB: Output, gradientW: Output, gradientB: Output, trainOp: Op, assignVariableW: Op, assignVariableB: Op): Unit = {
    // ---------- PRINT RESTORED OUTPUT AND OPERATIONS ------------
    println(" *" * 60)
    myPrintln("Trained weight value: " + session.run(fetches = weight).scalar, 90)
    myPrintln("Trained bias value: " + session.run(fetches = bias).scalar, 90)
    myPrintln("" + input, 116)
    myPrintln("" + output, 116)
    myPrintln("" + weight, 116)
    myPrintln("" + bias, 116)
    myPrintln("" + loss, 116)
    myPrintln("" + placeholderW, 116)
    myPrintln("" + placeholderB, 116)
    myPrintln("" + gradientW, 116)
    myPrintln("" + gradientB, 116)

    myPrintln("" + trainOp, 116)
    myPrintln("" + assignVariableW, 116)
    myPrintln("" + assignVariableB, 116)
    println(" *" * 60)
    // ---------------------------------------------

    println("")
  }

  def myPrintln(str: String, size: Int): Unit = {
    println(String.format("| %1$-" + size + "s |", str))
  }
}
