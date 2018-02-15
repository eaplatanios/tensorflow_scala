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
import org.platanios.tensorflow.api.ops.{Math, Op, Output, OutputLike}
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.io.data.Iterator
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.FLOAT32

/**
  * @author Emmanouil Antonios Platanios
  */
trait Model {
  protected val colocateGradientsWithOps: Boolean = false
}

trait InferenceModel[IT, IO, ID, IS, I] extends Model {
  def buildInferOps(): Model.InferOps[IT, IO, ID, IS, I]
}

trait TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] extends InferenceModel[IT, IO, ID, IS, I] {
  def buildTrainOps(): Model.TrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS]
  def buildEvaluateOps(metrics: Seq[Metric[EI, Output]]): Model.EvaluateOps[TT, TO, TD, TS, I]
}

trait SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    extends TrainableModel[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS), (I, T)] {
  def buildTrainOps(): Model.SupervisedTrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
  def buildEvaluateOps(
      metrics: Seq[Metric[(I, T), Output]]
  ): Model.EvaluateOps[(IT, TT), (IO, TO), (ID, TD), (IS, TS), I]
}

trait UnsupervisedTrainableModel[IT, IO, ID, IS, I]
    extends TrainableModel[IT, IO, ID, IS, I, IT, IO, ID, IS, I] {
  def buildTrainOps(): Model.UnsupervisedTrainOps[IT, IO, ID, IS, I]
  def buildEvaluateOps(metrics: Seq[Metric[I, Output]]): Model.EvaluateOps[IT, IO, ID, IS, I]
}

object Model {
  def unsupervised[IT, IO, IDA, ID, IS, I](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      loss: Layer[(IO, I), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients = NoClipGradients,
      colocateGradientsWithOps: Boolean = false
  ): UnsupervisedTrainableModel[IT, IO, ID, IS, I] = {
    new SimpleUnsupervisedTrainableModel(input, layer, loss, optimizer, clipGradients, colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SimpleSupervisedTrainableModel(input, layer, trainInput, trainInputLayer, loss, optimizer)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SimpleSupervisedTrainableModel(input, layer, trainInput, trainInputLayer, loss, optimizer, clipGradients)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer, clipGradients)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer, clipGradients)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer,
      clipGradients)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, trainInputLayer, loss, optimizer, colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, trainInputLayer, loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  ): SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS, T](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      trainInputLayer: Layer[TO, T],
      loss: Layer[(I, T), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[IT, IO, IDA, ID, IS, I, TT, TO, TDA, TD, TS](
      input: Input[IT, IO, IDA, ID, IS],
      layer: Layer[IO, I],
      trainLayer: Layer[(IO, TO), I],
      trainInput: Input[TT, TO, TDA, TD, TS],
      loss: Layer[(I, TO), Output],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  ): SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, TO] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TO]("TrainInputLayer"), loss, optimizer,
      clipGradients, colocateGradientsWithOps)
  }

  case class InferOps[IT, IO, ID, IS, I](inputIterator: Iterator[IT, IO, ID, IS], input: IO, output: I)

  private[learn] class TrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS](
      val inputIterator: Iterator[TT, TO, TD, TS],
      val input: TO,
      val output: I,
      val loss: Output,
      val gradientsAndVariables: Seq[(OutputLike, Variable)],
      val trainOp: Op)

  case class UnsupervisedTrainOps[IT, IO, ID, IS, I](
      override val inputIterator: Iterator[IT, IO, ID, IS],
      override val input: IO,
      override val output: I,
      override val loss: Output,
      override val gradientsAndVariables: Seq[(OutputLike, Variable)],
      override val trainOp: Op
  ) extends TrainOps[IT, IO, ID, IS, I, IT, IO, ID, IS](
    inputIterator, input, output, loss, gradientsAndVariables, trainOp)

  case class SupervisedTrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      override val inputIterator: Iterator[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
      override val input: (IO, TO),
      override val output: I,
      trainOutput: T,
      override val loss: Output,
      override val gradientsAndVariables: Seq[(OutputLike, Variable)],
      override val trainOp: Op
  ) extends TrainOps[IT, IO, ID, IS, I, (IT, TT), (IO, TO), (ID, TD), (IS, TS)](
    inputIterator, input, output, loss, gradientsAndVariables, trainOp)

  object SupervisedTrainOps {
    def apply[IT, IO, ID, IS, I](
        inputIterator: Iterator[(IT, IT), (IO, IO), (ID, ID), (IS, IS)],
        input: (IO, IO),
        output: I,
        loss: Output,
        gradientsAndVariables: Seq[(OutputLike, Variable)],
        trainOp: Op
    ): SupervisedTrainOps[IT, IO, ID, IS, I, IT, IO, ID, IS, I] = {
      SupervisedTrainOps(inputIterator, input, output, output, loss, gradientsAndVariables, trainOp)
    }
  }

  case class EvaluateOps[IT, IO, ID, IS, I](
      inputIterator: Iterator[IT, IO, ID, IS],
      input: IO,
      output: I,
      metricValues: Seq[Output],
      metricUpdates: Seq[Output],
      metricResets: Seq[Op])
}

private[learn] class SimpleInferenceModel[IT, IO, ID, IS, I] private[learn](
    val input: Input[IT, IO, _, ID, IS],
    val layer: Layer[IO, I]
) extends InferenceModel[IT, IO, ID, IS, I] {
  override def buildInferOps(): Model.InferOps[IT, IO, ID, IS, I] = {
    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext, INFERENCE)
    Model.InferOps(inputIterator, inputIteratorNext, layerOutput)
  }
}

private[learn] class SimpleUnsupervisedTrainableModel[IT, IO, ID, IS, I] private[learn](
    override val input: Input[IT, IO, _, ID, IS],
    override val layer: Layer[IO, I],
    val loss: Layer[(IO, I), Output],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
) extends SimpleInferenceModel[IT, IO, ID, IS, I](input, layer)
    with UnsupervisedTrainableModel[IT, IO, ID, IS, I] {
  // TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

  override def buildTrainOps(): Model.UnsupervisedTrainOps[IT, IO, ID, IS, I] = {
    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext, TRAINING)
    // TODO: [LEARN] Remove this cast.
    val lossOutput = Math.cast(loss((inputIteratorNext, layerOutput), TRAINING), FLOAT32, name = "LossCast")
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))
    Model.UnsupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(metrics: Seq[Metric[I, Output]]): Model.EvaluateOps[IT, IO, ID, IS, I] = {
    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext, EVALUATION)
    val streamingInstances = metrics.map(_.streaming(layerOutput))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset))
  }
}

private[learn] class SimpleSupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] private[learn](
    override val input: Input[IT, IO, _, ID, IS],
    override val layer: Layer[IO, I],
    val trainInput: Input[TT, TO, _, TD, TS],
    val trainInputLayer: Layer[TO, T],
    val loss: Layer[(I, T), Output],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
) extends SimpleInferenceModel[IT, IO, ID, IS, I](input, layer)
    with SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] {
  // TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

  override def buildTrainOps(): Model.SupervisedTrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1, TRAINING)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2, TRAINING)
    // TODO: [LEARN] Remove this cast.
    val lossOutput = Math.cast(
      loss((layerOutput, trainLayerOutput), TRAINING), FLOAT32, name = "LossCast")
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))
    Model.SupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, trainLayerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(
      metrics: Seq[Metric[(I, T), Output]]
  ): Model.EvaluateOps[(IT, TT), (IO, TO), (ID, TD), (IS, TS), I] = {
    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1, EVALUATION)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2, EVALUATION)
    val streamingInstances = metrics.map(_.streaming((layerOutput, trainLayerOutput)))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset))
  }
}

private[learn] class SupervisedConditionalTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] private[learn](
    override val input: Input[IT, IO, _, ID, IS],
    override val layer: Layer[IO, I],
    val trainLayer: Layer[(IO, TO), I],
    val trainInput: Input[TT, TO, _, TD, TS],
    val trainInputLayer: Layer[TO, T],
    val loss: Layer[(I, T), Output],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
) extends SimpleInferenceModel[IT, IO, ID, IS, I](input, layer)
    with SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] {
  // TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

  override def buildTrainOps(): Model.SupervisedTrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = trainLayer(inputIteratorNext, TRAINING)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2, TRAINING)
    // TODO: [LEARN] Remove this cast.
    val lossOutput = Math.cast(
      loss((layerOutput, trainLayerOutput), TRAINING), FLOAT32, name = "LossCast")
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))
    Model.SupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, trainLayerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(
      metrics: Seq[Metric[(I, T), Output]]
  ): Model.EvaluateOps[(IT, TT), (IO, TO), (ID, TD), (IS, TS), I] = {
    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1, EVALUATION)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2, EVALUATION)
    val streamingInstances = metrics.map(_.streaming((layerOutput, trainLayerOutput)))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset))
  }
}
