package org.platanios.tensorflow.examples.python2scala

import java.io._
import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.ops.{Output, UntypedOp}
import org.slf4j.LoggerFactory
import org.tensorflow.framework.MetaGraphDef

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Purpose: take a model trained with the Python API of Tensor Flow and restore
  * it inside a Scala environment in order to continue the training process. For
  * this example was used a trained model but can be used a virgin one without any
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

  private[this] val checkpoint = "examples/src/main/resources/python2scala/linear-regression"
  private[this] val meta = new File(getClass.getClassLoader.getResource("python2scala/linear-regression.meta").getFile)
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
      val prediction = session.graph.getOutputByName("p2s_prediction:0")
      val loss = session.graph.getOutputByName("p2s_loss:0")

      val trainOp = session.graph.getOpByName("p2s_train_op")

      printRestoredNodesAndOperations(session, input, output, weight, bias, prediction, loss, trainOp)

      // TRAINING LOOP
      for (i <- 0 to 50) {
        val (one, two) = batch(10000)
        val feedsMap = FeedMap(Map(input -> one, output -> two))
        val fetchesSeq = Seq(loss, weight, bias)
        val trainFetches = session.run(feeds = feedsMap, fetches = fetchesSeq, targets = trainOp)
        val trainLoss = trainFetches(0)
        val trainWeight = trainFetches(1)
        val trainBias = trainFetches(2)
        logger.info(s"\nTrain loss at iteration ${i+1} = ${trainLoss.scalar}\nTrain weight at iteration ${i+1} = ${trainWeight.scalar}\nTrain bias at iteration ${i+1} = ${trainBias.scalar}\n")
      }
    }
  }

  // UTILITY METHODS
  def batch(batchSize: Int): (Tensor[Any], Tensor[Any]) = {
    val inputs = ArrayBuffer.empty[Double]
    val outputs = ArrayBuffer.empty[Double]
    for (_ <- 0 until batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
    }
    (Tensor(inputs).reshape(Shape(-1, 1)), Tensor(outputs).reshape(Shape(-1, 1)))
  }

  def printRestoredNodesAndOperations(
      session: Session,
      input: Output[Any],
      output: Output[Any],
      weight: Output[Any],
      bias: Output[Any],
      prediction: Output[Any],
      loss: Output[Any],
      trainOp: UntypedOp
  ): Unit = {
    // ---------- PRINT RESTORED OUTPUT AND OPERATIONS ------------
    println(" *" * 60)
    myPrintln("Trained weight value: " + session.run(fetches = weight).scalar, 90)
    myPrintln("Trained bias value: " + session.run(fetches = bias).scalar, 90)
    myPrintln("" + input, 116)
    myPrintln("" + output, 116)
    myPrintln("" + weight, 116)
    myPrintln("" + bias, 116)
    myPrintln("" + prediction, 116)
    myPrintln("" + loss, 116)
    println(" *" * 60)
    // ---------------------------------------------

    println("")
  }

  def myPrintln(str: String, size: Int): Unit = {
    println(String.format("| %1$-" + size + "s |", str))
  }
}
