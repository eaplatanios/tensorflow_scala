/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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
import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.data.DatasetIterator
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.variables.Variable

// TODO: [LEARN] Add support for trainable models with only the loss function gradient available.

/**
  * @author Emmanouil Antonios Platanios
  */
trait Model

abstract class InferenceModel[In, Out](implicit
    evStructureIn: NestedStructure.Aux[In, _, _, _]
) extends Model {
  def buildInferOps(): Model.InferOps[In, Out]
}

abstract class TrainableModel[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble, EvalIn](implicit
    evStructureIn: NestedStructure.Aux[In, _, _, _],
    evStructureTrainIn: NestedStructure.Aux[TrainIn, _, _, _]
) extends InferenceModel[In, Out] {
  def buildTrainOps(): Model.TrainOps[TrainIn, TrainOut, Loss]
  def buildEvalOps(metrics: Seq[Metric[EvalIn, Output[Float]]]): Model.EvalOps[TrainIn, Out]
}

abstract class SupervisedTrainableModel[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble](implicit
    evIn: NestedStructure.Aux[In, _, _, _],
    evTrainIn: NestedStructure.Aux[TrainIn, _, _, _]
) extends TrainableModel[In, (In, TrainIn), Out, TrainOut, Loss, (Out, (In, TrainIn))] {
  val input: Input[In]
  val layer: Layer[In, Out]
  val trainInput: Input[TrainIn]
  val trainLayer: Layer[(In, TrainIn), TrainOut]
  val loss      : Layer[(TrainOut, (In, TrainIn)), Output[Loss]]
  val optimizer : Optimizer

  val clipGradients           : ClipGradients = NoClipGradients
  val colocateGradientsWithOps: Boolean       = false

  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)

    Model.InferOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = layerOutput)
  }

  override def buildTrainOps(): Model.TrainOps[(In, TrainIn), TrainOut, Loss] = {
    implicit val mode: Mode = TRAINING

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val trainLayerOutput = trainLayer(inputIteratorNext)
    val lossOutput = loss((trainLayerOutput, inputIteratorNext))
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))

    Model.TrainOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = trainLayerOutput,
      loss = lossOutput,
      gradientsAndVariables = gradientsAndVariables,
      trainOp = trainOp)
  }

  override def buildEvalOps(
      metrics: Seq[Metric[(Out, (In, TrainIn)), Output[Float]]]
  ): Model.EvalOps[(In, TrainIn), Out] = {
    implicit val mode: Mode = EVALUATION

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1)
    val streamingInstances = metrics.map(_.streaming((layerOutput, inputIteratorNext)))

    Model.EvalOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = layerOutput,
      metricValues = streamingInstances.map(_.value),
      metricUpdates = streamingInstances.map(_.update),
      metricResets = streamingInstances.map(_.reset).toSet)
  }
}

abstract class UnsupervisedTrainableModel[In, Out, Loss: TF : IsFloatOrDouble](implicit
    evIn: NestedStructure.Aux[In, _, _, _]
) extends TrainableModel[In, In, Out, Out, Loss, Out] {
  val input: Input[In]
  val layer: Layer[In, Out]
  val loss     : Layer[(In, Out), Output[Loss]]
  val optimizer: Optimizer
  val clipGradients           : ClipGradients = NoClipGradients
  val colocateGradientsWithOps: Boolean       = false

  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)

    Model.InferOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = layerOutput)
  }

  override def buildTrainOps(): Model.TrainOps[In, Out, Loss] = {
    implicit val mode: Mode = TRAINING

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    val lossOutput = loss((inputIteratorNext, layerOutput))
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))

    Model.TrainOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = layerOutput,
      loss = lossOutput,
      gradientsAndVariables = gradientsAndVariables,
      trainOp = trainOp)
  }

  override def buildEvalOps(metrics: Seq[Metric[Out, Output[Float]]]): Model.EvalOps[In, Out] = {
    implicit val mode: Mode = EVALUATION

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    val streamingInstances = metrics.map(_.streaming(layerOutput))

    Model.EvalOps(
      inputIterator = inputIterator,
      input = inputIteratorNext,
      output = layerOutput,
      metricValues = streamingInstances.map(_.value),
      metricUpdates = streamingInstances.map(_.update),
      metricResets = streamingInstances.map(_.reset).toSet)
  }
}

object Model {
  case class InferOps[In, Out](
      inputIterator: DatasetIterator[In],
      input: In,
      output: Out)

  case class TrainOps[TrainIn, TrainOut, Loss: TF : IsFloatOrDouble](
      inputIterator: DatasetIterator[TrainIn],
      input: TrainIn,
      output: TrainOut,
      loss: Output[Loss],
      gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])],
      trainOp: UntypedOp)

  case class EvalOps[In, Out](
      inputIterator: DatasetIterator[In],
      input: In,
      output: Out,
      metricValues: Seq[Output[Float]],
      metricUpdates: Seq[Output[Float]],
      metricResets: Set[UntypedOp])

  def simpleSupervised[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble](
      input: Input[In],
      trainInput: Input[TrainIn],
      layer: Layer[In, Out],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients = NoClipGradients,
      colocateGradientsWithOps: Boolean = false
  )(implicit
      evIn: NestedStructure.Aux[In, _, _, _],
      evTrainIn: NestedStructure.Aux[TrainIn, _, _, _]
  ): SupervisedTrainableModel[In, TrainIn, Out, Out, Loss] = {
    val trainLayer = new Layer[(In, TrainIn), Out]("SimpleSupervisedTrainLayer") {
      override val layerType: String = "SimpleSupervisedTrainLayer"
      override def forwardWithoutContext(
          input: (In, TrainIn)
      )(implicit mode: Mode): Out = {
        layer(input._1)
      }
    }

    val lossLayer = new Layer[(Out, (In, TrainIn)), Output[Loss]]("SimpleSupervisedLossLayer") {
        override val layerType: String = "SimpleSupervisedLossLayer"
        override def forwardWithoutContext(
          input: (Out, (In, TrainIn))
      )(implicit mode: Mode): Output[Loss] = {
        loss((input._1, input._2._2))
      }
    }

    val providedInput = input
    val providedTrainInput = trainInput
    val providedLayer = layer
    val providedTrainLayer = trainLayer
    val providedLoss = lossLayer
    val providedOptimizer = optimizer
    val providedClipGradients = clipGradients
    val providedColocateGradientsWithOps = colocateGradientsWithOps

    new SupervisedTrainableModel[In, TrainIn, Out, Out, Loss] {
      override val input                   : Input[In]                                 = providedInput
      override val trainInput              : Input[TrainIn]                            = providedTrainInput
      override val layer                   : Layer[In, Out]                            = providedLayer
      override val trainLayer              : Layer[(In, TrainIn), Out]                 = providedTrainLayer
      override val loss                    : Layer[(Out, (In, TrainIn)), Output[Loss]] = providedLoss
      override val optimizer               : Optimizer                                 = providedOptimizer
      override val clipGradients           : ClipGradients                             = providedClipGradients
      override val colocateGradientsWithOps: Boolean                                   = providedColocateGradientsWithOps
    }
  }

  def supervised[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble](
      input: Input[In],
      trainInput: Input[TrainIn],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), TrainOut],
      loss: Layer[(TrainOut, (In, TrainIn)), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients = NoClipGradients,
      colocateGradientsWithOps: Boolean = false
  )(implicit
      evIn: NestedStructure.Aux[In, _, _, _],
      evTrainIn: NestedStructure.Aux[TrainIn, _, _, _]
  ): SupervisedTrainableModel[In, TrainIn, Out, TrainOut, Loss] = {
    val providedInput = input
    val providedTrainInput = trainInput
    val providedLayer = layer
    val providedTrainLayer = trainLayer
    val providedLoss = loss
    val providedOptimizer = optimizer
    val providedClipGradients = clipGradients
    val providedColocateGradientsWithOps = colocateGradientsWithOps

    new SupervisedTrainableModel[In, TrainIn, Out, TrainOut, Loss] {
      override val input                   : Input[In]                                      = providedInput
      override val trainInput              : Input[TrainIn]                                 = providedTrainInput
      override val layer                   : Layer[In, Out]                                 = providedLayer
      override val trainLayer              : Layer[(In, TrainIn), TrainOut]                 = providedTrainLayer
      override val loss                    : Layer[(TrainOut, (In, TrainIn)), Output[Loss]] = providedLoss
      override val optimizer               : Optimizer                                      = providedOptimizer
      override val clipGradients           : ClipGradients                                  = providedClipGradients
      override val colocateGradientsWithOps: Boolean                                        = providedColocateGradientsWithOps
    }
  }

  def unsupervised[In, Out, Loss: TF : IsFloatOrDouble](
      input: Input[In],
      layer: Layer[In, Out],
      loss: Layer[(In, Out), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients = NoClipGradients,
      colocateGradientsWithOps: Boolean = false
  )(implicit
      evIn: NestedStructure.Aux[In, _, _, _]
  ): UnsupervisedTrainableModel[In, Out, Loss] = {
    val providedInput = input
    val providedLayer = layer
    val providedLoss = loss
    val providedOptimizer = optimizer
    val providedClipGradients = clipGradients
    val providedColocateGradientsWithOps = colocateGradientsWithOps

    new UnsupervisedTrainableModel[In, Out, Loss] {
      override val input                   : Input[In]                      = providedInput
      override val layer                   : Layer[In, Out]                 = providedLayer
      override val loss                    : Layer[(In, Out), Output[Loss]] = providedLoss
      override val optimizer               : Optimizer                      = providedOptimizer
      override val clipGradients           : ClipGradients                  = providedClipGradients
      override val colocateGradientsWithOps: Boolean                        = providedColocateGradientsWithOps
    }
  }
}
