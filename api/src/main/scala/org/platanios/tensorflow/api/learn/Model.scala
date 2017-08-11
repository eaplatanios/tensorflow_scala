/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.core.client.FeedMap
import org.platanios.tensorflow.api.ops.io.{Data, Dataset}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object Model {
  // TODO: Make private.
  val logger = Logger(LoggerFactory.getLogger("Learn / Model"))

  trait API {
    def model[IT, ID, IS, I, TT, TD, TS, ST, T](
        input: SupportedInput[IT, ID, IS],
        layer: Layer[IT, I],
        trainingInput: SupportedInput[TT, TD, TS],
        trainingInputLayer: Layer[TT, T],
        loss: Layer[(I, T), tf.Output],
        optimizer: tf.train.Optimizer)(implicit
        inputEv: Data.Aux[IT, ID, IS],
        trainInputEv: Data.Aux[TT, TD, TS]): TrainableModel[IT, ID, IS, I, TT, TD, TS, ST, T] = {
      new TrainableModel(input, layer, trainingInput, trainingInputLayer, loss, optimizer)
    }
  }

  object API extends API

  //  def apply[I](input: Input, inferenceLayer: Layer[OutputLike, I]): InferenceModel[I] = new InferenceModel[I](inferenceLayer)
  //
  //  def apply[I](inferenceLayer: Layer[Unit, I], loss: Loss[(I, I)], optimizer: Optimizer): TrainableModel[I, I] = {
  //    new TrainableModel[I, I](
  //      inferenceLayer = inferenceLayer,
  //      trainingLayer = inferenceLayer,
  //      loss = loss,
  //      optimizer = optimizer)
  //  }
  //
  //  def apply[O, I, S](
  //      bodyLayer: Layer[Unit, O], inferenceLayer: Layer[O, I], loss: Loss[(O, S)],
  //      optimizer: Optimizer): TrainableModel[O, I, O, S] = {
  //    new TrainableModel[O, I, O, S](
  //      bodyLayer = bodyLayer,
  //      inferenceLayer = inferenceLayer,
  //      trainingLayer = IdentityLayer[O](),
  //      loss = loss,
  //      optimizer = optimizer)
  //  }
  //
  //  def apply[I, T, S](inferenceLayer: Layer[Unit, I], trainingLayer: Layer[Unit, T], loss: Loss[(T, S)],
  //      optimizer: Optimizer): TrainableModel[I, T, S] = {
  //    new TrainableModel[I, T, S](inferenceLayer, trainingLayer, loss, optimizer)
  //  }

  private[Model] def save(session: tf.Session, saver: tf.Saver, directory: Path, filename: String): Unit = {
    saver.save(session, directory.resolve(filename))
  }

  private[Model] def load(session: tf.Session, saver: tf.Saver, directory: Path, filename: String): Unit = {
    saver.restore(session, directory.resolve(filename))
  }

  private[Model] def saveCheckpoint(
      session: tf.Session, saver: tf.Saver, directory: Path, filePrefix: String,
      step: Option[Int] = None): Option[Path] = {
    saver.save(session, if (step.isEmpty) directory.resolve(filePrefix) else directory, step, filePrefix)
  }

  private[Model] def loadCheckpoint(
      session: tf.Session, saver: tf.Saver, directory: Path, filePrefix: String, step: Option[Int] = None,
      feedMap: FeedMap = FeedMap.empty): Unit = {
    step match {
      case Some(s) =>
        val path = directory.resolve(s"$filePrefix$s")
        if (Files.isRegularFile(path)) {
          saver.restore(session, path)
        } else {
          logger.warn(s"Could not load checkpoint file from the provided path: $path.")
          logger.warn("Using default initialization instead.")
          session.run(feeds = feedMap, targets = tf.createWith(session.graph)(tf.globalVariablesInitializer()))
        }
      case None =>
        // TODO: [LEARN] Check if filename/working_dir contains illegal characters.
        val path = tf.Saver.latestCheckpoint(directory, filePrefix)
        if (path.isDefined) {
          saver.restore(session, path.get)
        } else {
          logger.warn(s"No checkpoint file was found in '$directory' with prefix '$filePrefix'.")
          logger.warn("Using default initialization instead.")
          session.run(feeds = feedMap, targets = tf.createWith(session.graph)(tf.globalVariablesInitializer()))
        }
    }
  }

  private[Model] def loadLatestCheckpoint(
      session: tf.Session, saver: tf.Saver, directory: Path, filePrefix: String,
      feedMap: FeedMap = FeedMap.empty): Unit = {
    loadCheckpoint(session, saver, directory, filePrefix, None, feedMap)
  }

  sealed trait Initialization {
    private[Model] def initialize(
        graph: tf.Graph, session: tf.Session, saver: Option[tf.Saver] = None,
        checkpointDirectory: Path = Paths.get("."), checkpointFilePrefix: String = "checkpoint",
        step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): tf.Session
  }

  case object NoInitialization extends Initialization {
    override private[Model] def initialize(
        graph: tf.Graph, session: tf.Session, saver: Option[tf.Saver] = None,
        checkpointDirectory: Path = Paths.get("."), checkpointFilePrefix: String = "checkpoint",
        step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): tf.Session = session
  }

  case object DefaultInitialization extends Initialization {
    override private[Model] def initialize(
        graph: tf.Graph, session: tf.Session, saver: Option[tf.Saver] = None,
        checkpointDirectory: Path = Paths.get("."), checkpointFilePrefix: String = "checkpoint",
        step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): tf.Session = {
      session.run(feeds = feedMap, targets = tf.createWith(graph)(tf.globalVariablesInitializer()))
      session
    }
  }

  case object LatestCheckpointInitialization extends Initialization {
    override private[Model] def initialize(
        graph: tf.Graph, session: tf.Session, saver: Option[tf.Saver] = None,
        checkpointDirectory: Path = Paths.get("."), checkpointFilePrefix: String = "checkpoint",
        step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): tf.Session = {
      saver.foreach(loadCheckpoint(session, _, checkpointDirectory, checkpointFilePrefix, step, feedMap))
      session
    }
  }

  case class CheckpointInitialization(checkpointNumber: Int) extends Initialization {
    override private[Model] def initialize(
        graph: tf.Graph, session: tf.Session, saver: Option[tf.Saver] = None,
        checkpointDirectory: Path = Paths.get("."), checkpointFilePrefix: String = "checkpoint",
        step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): tf.Session = {
      saver.foreach(loadCheckpoint(session, _, checkpointDirectory, checkpointFilePrefix, step, feedMap))
      session
    }
  }
}

trait Model {
  private[learn] val graph  : tf.Graph   = tf.Graph()
  private[learn] var session: tf.Session = tf.session(graph)

  // TODO: Add "restoreSequentially" support, and other options.
  private[learn] val saver: tf.Saver

  def initialize(
      method: Model.Initialization, checkpointDirectory: Path = Paths.get("."),
      checkpointFilePrefix: String = "checkpoint", step: Option[Int] = None, feedMap: FeedMap = FeedMap.empty): Unit = {
    session = method.initialize(
      graph = graph, session = session, saver = Some(saver), checkpointDirectory = checkpointDirectory,
      checkpointFilePrefix = checkpointFilePrefix, step = step, feedMap = feedMap)
  }

  def save(directory: Path, filename: String): Unit = Model.save(session, saver, directory, filename)
  def load(directory: Path, filename: String): Unit = Model.load(session, saver, directory, filename)
}

class InferenceModel[IT, ID, IS, I] private[learn](
    val input: SupportedInput[IT, ID, IS],
    val layer: Layer[IT, I])(implicit
    inputEv: Data.Aux[IT, ID, IS]) extends Model {
  private[this] val tfBuiltInferenceOps               = addInferenceOpsToGraph(graph)
  protected     val tfInput : tf.Iterator[IT, ID, IS] = tfBuiltInferenceOps._1
  protected     val tfOutput: I                       = tfBuiltInferenceOps._2

  override private[learn] val saver: tf.Saver = tf.createWith(graph)(tf.saver())

  protected def addInferenceOpsToGraph(graph: tf.Graph): (tf.Iterator[IT, ID, IS], I) = {
    tf.createWith(graph) {
      val tfInput = input()
      val tfOutput = layer(tfInput.next())
      (tfInput, tfOutput)
    }
  }
}

class TrainableModel[IT, ID, IS, I, TT, TD, TS, ST, T] private[learn](
    override val input: SupportedInput[IT, ID, IS],
    override val layer: Layer[IT, I],
    val trainingInput: SupportedInput[TT, TD, TS],
    val trainingInputLayer: Layer[TT, T],
    val loss: Layer[(I, T), tf.Output],
    val optimizer: tf.train.Optimizer)(implicit
    inputEv: Data.Aux[IT, ID, IS],
    trainInputEv: Data.Aux[TT, TD, TS])
    extends InferenceModel[IT, ID, IS, I](input, layer) {
  private[this] val tfBuiltTrainingOps                        = addTrainingOpsToGraph(graph)
  protected     val tfTrainingInput : tf.Iterator[TT, TD, TS] = tfBuiltTrainingOps._1
  protected     val tfTrainingOutput: T                       = tfBuiltTrainingOps._2
  protected     val tfLoss          : tf.Output               = tfBuiltTrainingOps._3
  protected     val tfIteration     : tf.Variable             = tfBuiltTrainingOps._4
  protected     val tfTrainOp       : tf.Op                   = tfBuiltTrainingOps._5

  override private[learn] val saver: tf.Saver = tf.createWith(graph)(tf.saver())

  protected def addTrainingOpsToGraph(graph: tf.Graph): (tf.Iterator[TT, TD, TS], T, tf.Output, tf.Variable, tf.Op) = {
    tf.createWith(graph = graph) {
      val tfTrainingInput = trainingInput()
      val tfTrainingOutput = trainingInputLayer(tfTrainingInput.next())
      // TODO: [LEARN] !!! Remove this cast.
      val tfLoss = tf.cast(loss(tfOutput, tfTrainingOutput), tf.FLOAT32, name = "LearnLossCast")
      val tfIteration = tf.variable("TrainingIteration", tf.INT32, tf.shape(), tf.zerosInitializer)
      val tfTrainOp = optimizer.minimize(tfLoss, iteration = Some(tfIteration))
      (tfTrainingInput, tfTrainingOutput, tfLoss, tfIteration, tfTrainOp)
    }
  }

  // TODO: [DATASETS] !!! Switch to using a single dataset.
  def train(
      dataI: Dataset[IT, ID, IS], dataT: Dataset[TT, TD, TS], maxEpochs: Int = 100, maxIterations: Int = 10000,
      absLossChangeTol: Float = 1e-3f, relLossChangeTol: Float = 1e-3f, maxIterBelowTol: Int = 10): Unit = {
    // Initialize the dataset iterators.
    session.run(targets = (tfInput.createInitializer(dataI), tfTrainingInput.createInitializer(dataT)))
    // Run the training loop.
    var previousLoss: Float = Float.MaxValue
    var iterBelowTol: Int = 0
    var iteration: Int = 0
    var converged: Boolean = false
    while (!converged) {
      try {
        // TODO: [LEARN] !!! Remove the cast after the TODO in "addTrainingOpsToGraph" is resolved.
        val loss = session.run(fetches = tfLoss, targets = tfTrainOp).scalar.asInstanceOf[Float]
        Model.logger.info(f"Loss value: $loss%13.4e")
        val lossDifference = Math.abs(previousLoss - loss)
        if (lossDifference < absLossChangeTol || Math.abs(lossDifference / previousLoss) < relLossChangeTol)
          iterBelowTol += 1
        else
          iterBelowTol = 0
        if (iterBelowTol > maxIterBelowTol) {
          Model.logger.info("Loss value converged.")
          converged = true
        } else if (iteration > maxIterations - 1) {
          Model.logger.info("Maximum number of iterations reached.")
          converged = true
        }
        iteration += 1
        previousLoss = loss
      } catch {
        case _: Exception => converged = true
      }
    }
  }
}
