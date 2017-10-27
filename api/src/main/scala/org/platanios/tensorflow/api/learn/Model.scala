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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops.{Math, Op, Output}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.io.Iterator
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.types.FLOAT32

/**
  * @author Emmanouil Antonios Platanios
  */
trait Model

trait InferenceModel[IT, IO, ID, IS, I] extends Model {
  def buildInferenceOps(graph: Graph = Op.currentGraph): Model.InferenceOps[IT, IO, ID, IS, I]
}

trait TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] extends InferenceModel[IT, IO, ID, IS, I] {
  def buildTrainingOps(graph: Graph = Op.currentGraph): Model.TrainingOps[IT, IO, ID, IS, I, TT, TO, TD, TS]
  def buildEvaluationOps(
      metrics: Seq[Metric[EI, Output]], graph: Graph = Op.currentGraph
  ): Model.EvaluationOps[TT, TO, TD, TS, I]
}

trait SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    extends TrainableModel[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, T)] {
  def buildTrainingOps(
      graph: Graph = Op.currentGraph
  ): Model.SupervisedTrainingOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T]

  def buildEvaluationOps(
      metrics: Seq[Metric[(I, T), Output]], graph: Graph = Op.currentGraph
  ): Model.EvaluationOps[(IT, TT), (IO, TO), (ID, TD), (IS, TS), I]
}

trait UnsupervisedTrainableModel[IT, IO, ID, IS, I]
    extends TrainableModel[IT, IO, ID, IS, I, IT, IO, ID, IS, I] {
  def buildTrainingOps(graph: Graph = Op.currentGraph): Model.UnsupervisedTrainingOps[IT, IO, ID, IS, I]
  def buildEvaluationOps(
      metrics: Seq[Metric[I, Output]], graph: Graph = Op.currentGraph
  ): Model.EvaluationOps[IT, IO, ID, IS, I]
}

object Model {
  case class InferenceOps[IT, IO, ID, IS, I](inputIterator: Iterator[IT, IO, ID, IS], input: IO, output: I)

  private[learn] class TrainingOps[IT, IO, ID, IS, I, TT, TO, TD, TS](
      val inputIterator: Iterator[TT, TO, TD, TS],
      val input: TO,
      val output: I,
      val loss: Output,
      val trainOp: Op)

  case class UnsupervisedTrainingOps[IT, IO, ID, IS, I](
      override val inputIterator: Iterator[IT, IO, ID, IS],
      override val input: IO,
      override val output: I,
      override val loss: Output,
      override val trainOp: Op
  ) extends TrainingOps[IT, IO, ID, IS, I, IT, IO, ID, IS](inputIterator, input, output, loss, trainOp)

  case class SupervisedTrainingOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      override val inputIterator: Iterator[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
      override val input: (IO, TO),
      override val output: I,
      trainOutput: T,
      override val loss: Output,
      override val trainOp: Op
  ) extends TrainingOps[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS)](
    inputIterator, input, output, loss, trainOp)

  object SupervisedTrainingOps {
    def apply[IT, IO, ID, IS, I](
        inputIterator: Iterator[(IT, IT), (IO, IO), (ID, ID), (IS, IS)],
        input: (IO, IO),
        output: I,
        loss: Output,
        trainOp: Op): SupervisedTrainingOps[IT, IO, ID, IS, I, IT, IO, ID, IS, I] = {
      SupervisedTrainingOps(inputIterator, input, output, output, loss, trainOp)
    }
  }

  case class EvaluationOps[IT, IO, ID, IS, I](
      inputIterator: Iterator[IT, IO, ID, IS],
      input: IO,
      output: I,
      metricValues: Seq[Output],
      metricUpdates: Seq[Output],
      metricResets: Seq[Op])

  trait API {
    def Model[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
        input: Input[IT, IO, ID, IS],
        layer: Layer[IO, I],
        trainInput: Input[TT, TO, TD, TS],
        trainInputLayer: Layer[TO, T],
        loss: Layer[(I, T), Output],
        optimizer: Optimizer): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      new SimpleSupervisedTrainableModel(input, layer, trainInput, trainInputLayer, loss, optimizer)
    }

    def Model[IT, IO, ID, IS, I, TT, TO, TD, TS](
        input: Input[IT, IO, ID, IS],
        layer: Layer[IO, I],
        trainInput: Input[TT, TO, TD, TS],
        loss: Layer[(I, TO), Output],
        optimizer: Optimizer): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
      new SimpleSupervisedTrainableModel(
        input, layer, trainInput, Layer.identity[TO]("TrainInputLayer"), loss, optimizer)
    }
  }

  object API extends API
}

private[learn] class SimpleInferenceModel[IT, IO, ID, IS, I] private[learn](
    val input: Input[IT, IO, ID, IS],
    val layer: Layer[IO, I]
) extends InferenceModel[IT, IO, ID, IS, I] {
  def buildInferenceOps(graph: Graph = Op.currentGraph): Model.InferenceOps[IT, IO, ID, IS, I] = {
    Op.createWith(graph) {
      val tfInputIterator = input()
      val tfInput = tfInputIterator.next()
      val tfOutput = layer(tfInput)
      Model.InferenceOps(tfInputIterator, tfInput, tfOutput)
    }
  }
}

private[learn] class SimpleUnsupervisedTrainableModel[IT, IO, ID, IS, I] private[learn](
    override val input: Input[IT, IO, ID, IS],
    override val layer: Layer[IO, I],
    val loss: Layer[I, Output],
    val optimizer: Optimizer
) extends SimpleInferenceModel[IT, IO, ID, IS, I](input, layer)
    with UnsupervisedTrainableModel[IT, IO, ID, IS, I] {
  // TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

  def buildTrainingOps(graph: Graph = Op.currentGraph): Model.UnsupervisedTrainingOps[IT, IO, ID, IS, I] = {
    Op.createWith(graph = graph) {
      val tfInputIterator = input()
      val tfInput = tfInputIterator.next()
      val tfOutput = layer(tfInput)
      // TODO: [LEARN] Remove this cast.
      val tfLoss = Math.cast(loss(tfOutput), FLOAT32, name = "LossCast")
      val tfIteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      val tfTrainOp = optimizer.minimize(tfLoss, iteration = Some(tfIteration))
      Model.UnsupervisedTrainingOps(tfInputIterator, tfInput, tfOutput, tfLoss, tfTrainOp)
    }
  }

  def buildEvaluationOps(
      metrics: Seq[Metric[I, Output]], graph: Graph = Op.currentGraph
  ): Model.EvaluationOps[IT, IO, ID, IS, I] = {
    Op.createWith(graph = graph) {
      val tfInputIterator = input.apply()
      val tfInput = tfInputIterator.next()
      val tfOutput = layer(tfInput)
      val (mValues, mUpdates, mResets) = metrics.map(_.streaming(tfOutput)).unzip3
      Model.EvaluationOps(tfInputIterator, tfInput, tfOutput, mValues, mUpdates, mResets)
    }
  }
}

private[learn] class SimpleSupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] private[learn](
    override val input: Input[IT, IO, ID, IS],
    override val layer: Layer[IO, I],
    val trainInput: Input[TT, TO, TD, TS],
    val trainLayer: Layer[TO, T],
    val loss: Layer[(I, T), Output],
    val optimizer: Optimizer
) extends SimpleInferenceModel[IT, IO, ID, IS, I](input, layer)
    with SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] {
  // TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

  def buildTrainingOps(
      graph: Graph = Op.currentGraph
  ): Model.SupervisedTrainingOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    Op.createWith(graph = graph) {
      val tfInputIterator = input.zip(trainInput).apply()
      val tfInput = tfInputIterator.next()
      val tfOutput = layer(tfInput._1)
      val tfTrainOutput = trainLayer(tfInput._2)
      // TODO: [LEARN] Remove this cast.
      val tfLoss = Math.cast(loss((tfOutput, tfTrainOutput)), FLOAT32, name = "LossCast")
      val tfIteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      val tfTrainOp = optimizer.minimize(tfLoss, iteration = Some(tfIteration))
      Model.SupervisedTrainingOps(tfInputIterator, tfInput, tfOutput, tfTrainOutput, tfLoss, tfTrainOp)
    }
  }

  def buildEvaluationOps(
      metrics: Seq[Metric[(I, T), Output]], graph: Graph = Op.currentGraph
  ): Model.EvaluationOps[(IT, TT), (IO, TO), (ID, TD), (IS, TS), I] = {
    Op.createWith(graph = graph) {
      val tfInputIterator = input.zip(trainInput).apply()
      val tfInput = tfInputIterator.next()
      val tfOutput = layer(tfInput._1)
      val tfTrainOutput = trainLayer(tfInput._2)
      val (mValues, mUpdates, mResets) = metrics.map(_.streaming((tfOutput, tfTrainOutput))).unzip3
      Model.EvaluationOps(tfInputIterator, tfInput, tfOutput, mValues, mUpdates, mResets)
    }
  }
}
