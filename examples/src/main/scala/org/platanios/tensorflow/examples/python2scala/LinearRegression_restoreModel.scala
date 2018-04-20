package python2scala

import java.io._
import java.nio.file.Paths

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.{Session, Shape, Tensor, tf}
import org.slf4j.LoggerFactory
import org.tensorflow.framework.MetaGraphDef

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


object LinearRegression_restoreModel {
  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))
  private[this] val random = new Random(22)
  private[this] val weight = random.nextFloat()

  def main(args: Array[String]): Unit = {

    logger.info("\n- - - - - PYTHON 2 SCALA - - - - - ")
    val stringPath: String = "/Users/databiz/Desktop/tensorflowHome/PythonTensorFlow/model-store-python/my-model-resource-backed"
    val path = Paths.get(stringPath + ".meta")
    val checkpointPath = Paths.get(stringPath)
    val mgf = MetaGraphDef.parseFrom(new BufferedInputStream(new FileInputStream(path.toFile)))


    val session = Session()
    val saver = tf.Saver.fromMetaGraphDef(metaGraphDef = mgf)
    saver.restore(session, checkpointPath)



    // ----- PRINT OPERATIONS IN THE GRAPH ---------
    println(" -" * 40)
    for (op <- session.graph.ops) {
      myPrintln("OPERATION name:    " + op, 76)
    }
    println(" -" * 40)
    // ---------------------------------------------


    //tf.Tensor in Python is the equivalent of Output here in scala
    val weights = session.graph.getOutputByName("p2s_weights/Read/ReadVariableOp:0")
    val inputs = session.graph.getOutputByName("p2s_inputs:0")
    val outputs = session.graph.getOutputByName("p2s_output:0")
    val loss = session.graph.getOutputByName("p2s_loss:0")
    val trainOp = session.graph.getOpByName("p2s_train_op")

    // ---------- PRINT RESTORED OUTPUT ------------
    println(" *" * 60)
    myPrintln("Trained weight value: " + session.run(fetches = weights).scalar, 90)
    myPrintln("" + weights, 116)
    myPrintln("" + inputs, 116)
    myPrintln("" + outputs, 116)
    myPrintln("" + loss, 116)
    myPrintln("" + trainOp, 116)
    println(" *" * 60)
    // ---------------------------------------------

    println("")

    for (i <- 0 to 50) {
      val trainBatch: (Tensor, Tensor) = batch(10000)
      val feedsMap: Map[Output, Tensor] = Map(inputs -> trainBatch._1, outputs -> trainBatch._2)
      val fetchesSeq = Seq(loss, weights)
      val trainFetches = session.run(feeds = feedsMap, fetches = fetchesSeq, targets = trainOp)
      val trainLoss = trainFetches.head
      val trainWeight = trainFetches(1)
      logger.info(s"Train loss at iteration $i = ${trainLoss.scalar} ")
      logger.info(s"Train weight at iteration $i = ${trainWeight.scalar}\n")
    }
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
    (Tensor(inputs).reshape(Shape(-1, 1)), Tensor(outputs).reshape(Shape(-1, 1)))
  }

  def myPrintln(str: String, size: Int): Unit = {
    println(String.format("| %1$-" + size + "s |", str))
  }
}
