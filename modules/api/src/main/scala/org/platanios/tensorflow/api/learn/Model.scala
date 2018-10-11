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
import org.platanios.tensorflow.api.core.types.{IsFloat32OrFloat64, TF}
import org.platanios.tensorflow.api.implicits.helpers.OutputStructure
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
trait Model {
  protected val colocateGradientsWithOps: Boolean = false
}

abstract class InferenceModel[In, Out](implicit
    evIn: OutputStructure.Aux[In, _, _],
) extends Model {
  def buildInferOps(): Model.InferOps[In, Out]
}

abstract class TrainableModel[In, TrainIn, TrainOut, Out, Loss, EvalIn](implicit
    evIn: OutputStructure.Aux[In, _, _],
    evTrainIn: OutputStructure.Aux[TrainIn, _, _]
) extends InferenceModel[In, Out] {
  def buildTrainOps(): Model.TrainOps[In, TrainIn, TrainOut, Out, Loss]
  def buildEvaluateOps(metrics: Seq[Metric[EvalIn, Output[Float]]]): Model.EvaluateOps[TrainIn, Out]
}

abstract class SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss](implicit
    evIn: OutputStructure.Aux[In, _, _],
    evTrainIn: OutputStructure.Aux[TrainIn, _, _]
) extends TrainableModel[In, (In, TrainIn), TrainOut, Out, Loss, (Out, TrainOut)] {
  override def buildTrainOps(): Model.SupervisedTrainOps[In, TrainIn, TrainOut, Out, Loss]
  override def buildEvaluateOps(metrics: Seq[Metric[(Out, TrainOut), Output[Float]]]): Model.EvaluateOps[(In, TrainIn), Out]
}

abstract class UnsupervisedTrainableModel[In, Out, Loss](implicit
    evIn: OutputStructure.Aux[In, _, _]
) extends TrainableModel[In, In, Unit, Out, Loss, Out] {
  override def buildTrainOps(): Model.UnsupervisedTrainOps[In, Out, Loss]
  override def buildEvaluateOps(metrics: Seq[Metric[Out, Output[Float]]]): Model.EvaluateOps[In, Out]
}

object Model {
  def unsupervised[In, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      loss: Layer[(In, Out), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients = NoClipGradients,
      colocateGradientsWithOps: Boolean = false
  )(implicit
      evIn: OutputStructure.Aux[In, _, _]
  ): UnsupervisedTrainableModel[In, Out, Loss] = {
    new SimpleUnsupervisedTrainableModel(input, layer, loss, optimizer, clipGradients, colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SimpleSupervisedTrainableModel(input, layer, trainInput, trainInputLayer, loss, optimizer)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SimpleSupervisedTrainableModel(input, layer, trainInput, trainInputLayer, loss, optimizer, clipGradients)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer, clipGradients)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer, clipGradients)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer,
      clipGradients)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, trainInputLayer, loss, optimizer, colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, trainInputLayer, loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervise[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SimpleSupervisedTrainableModel(
      input, layer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      trainInputLayer: Layer[TrainIn, TrainOut],
      loss: Layer[(Out, TrainOut), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainOut, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, trainInputLayer, loss, optimizer, clipGradients,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer,
      colocateGradientsWithOps = colocateGradientsWithOps)
  }

  def supervised[In, TrainIn, Out, Loss: TF : IsFloat32OrFloat64](
      input: Input[In],
      layer: Layer[In, Out],
      trainLayer: Layer[(In, TrainIn), Out],
      trainInput: Input[TrainIn],
      loss: Layer[(Out, TrainIn), Output[Loss]],
      optimizer: Optimizer,
      clipGradients: ClipGradients,
      colocateGradientsWithOps: Boolean
  )(implicit
      evIn: OutputStructure.Aux[In, _, _],
      evTrainIn: OutputStructure.Aux[TrainIn, _, _]
  ): SupervisedConditionalTrainableModel[In, TrainIn, TrainIn, Out, Loss] = {
    new SupervisedConditionalTrainableModel(
      input, layer, trainLayer, trainInput, layers.Identity[TrainIn]("TrainInputLayer"), loss, optimizer,
      clipGradients, colocateGradientsWithOps)
  }

  case class InferOps[In, Out](inputIterator: DatasetIterator[In], input: In, output: Out)

  private[learn] class TrainOps[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      val inputIterator: DatasetIterator[TrainIn],
      val input: TrainIn,
      val trainOutput: TrainOut,
      val output: Out,
      val loss: Output[Loss],
      val gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])],
      val trainOp: UntypedOp)

  case class UnsupervisedTrainOps[In, Out, Loss: TF : IsFloat32OrFloat64](
      override val inputIterator: DatasetIterator[In],
      override val input: In,
      override val output: Out,
      override val loss: Output[Loss],
      override val gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])],
      override val trainOp: UntypedOp
  ) extends TrainOps[In, In, Unit, Out, Loss](
    inputIterator, input, (), output, loss, gradientsAndVariables, trainOp)

  case class SupervisedTrainOps[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
      override val inputIterator: DatasetIterator[(In, TrainIn)],
      override val input: (In, TrainIn),
      override val output: Out,
      override val trainOutput: TrainOut,
      override val loss: Output[Loss],
      override val gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])],
      override val trainOp: UntypedOp
  ) extends TrainOps[In, (In, TrainIn), TrainOut, Out, Loss](
    inputIterator, input, trainOutput, output, loss, gradientsAndVariables, trainOp)

  object SupervisedTrainOps {
    def apply[In, Out, Loss: TF : IsFloat32OrFloat64](
        inputIterator: DatasetIterator[(In, In)],
        input: (In, In),
        output: Out,
        loss: Output[Loss],
        gradientsAndVariables: Seq[(OutputLike[Loss], Variable[Any])],
        trainOp: UntypedOp
    ): SupervisedTrainOps[In, In, Out, Out, Loss] = {
      SupervisedTrainOps(
        inputIterator = inputIterator,
        input = input,
        output = output,
        trainOutput = output,
        loss = loss,
        gradientsAndVariables = gradientsAndVariables,
        trainOp = trainOp)
    }
  }

  case class EvaluateOps[In, Out](
      inputIterator: DatasetIterator[In],
      input: In,
      output: Out,
      metricValues: Seq[Output[Float]],
      metricUpdates: Seq[Output[Float]],
      metricResets: Set[UntypedOp])
}

private[learn] class SimpleInferenceModel[In, Out](
    val input: Input[In],
    val layer: Layer[In, Out]
)(implicit
    evIn: OutputStructure.Aux[In, _, _]
) extends InferenceModel[In, Out] {
  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    Model.InferOps(inputIterator, inputIteratorNext, layerOutput)
  }
}

private[learn] class SimpleUnsupervisedTrainableModel[In, Out, Loss: TF : IsFloat32OrFloat64](
    val input: Input[In],
    val layer: Layer[In, Out],
    val loss: Layer[(In, Out), Output[Loss]],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
)(implicit
    evIn: OutputStructure.Aux[In, _, _]
) extends UnsupervisedTrainableModel[In, Out, Loss] {
  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    Model.InferOps(inputIterator, inputIteratorNext, layerOutput)
  }

  override def buildTrainOps(): Model.UnsupervisedTrainOps[In, Out, Loss] = {
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
    Model.UnsupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(metrics: Seq[Metric[Out, Output[Float]]]): Model.EvaluateOps[In, Out] = {
    implicit val mode: Mode = EVALUATION

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    val streamingInstances = metrics.map(_.streaming(layerOutput))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset).toSet)
  }
}

private[learn] class SimpleSupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
    val input: Input[In],
    val layer: Layer[In, Out],
    val trainInput: Input[TrainIn],
    val trainInputLayer: Layer[TrainIn, TrainOut],
    val loss: Layer[(Out, TrainOut), Output[Loss]],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
)(implicit
    evIn: OutputStructure.Aux[In, _, _],
    evTrainIn: OutputStructure.Aux[TrainIn, _, _]
) extends SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] {
  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    Model.InferOps(inputIterator, inputIteratorNext, layerOutput)
  }

  override def buildTrainOps(): Model.SupervisedTrainOps[In, TrainIn, TrainOut, Out, Loss] = {
    implicit val mode: Mode = TRAINING

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2)
    val lossOutput = loss((layerOutput, trainLayerOutput))
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))
    Model.SupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, trainLayerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(metrics: Seq[Metric[(Out, TrainOut), Output[Float]]]): Model.EvaluateOps[(In, TrainIn), Out] = {
    implicit val mode: Mode = EVALUATION

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2)
    val streamingInstances = metrics.map(_.streaming((layerOutput, trainLayerOutput)))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset).toSet)
  }
}

private[learn] class SupervisedConditionalTrainableModel[In, TrainIn, TrainOut, Out, Loss: TF : IsFloat32OrFloat64](
    val input: Input[In],
    val layer: Layer[In, Out],
    val trainLayer: Layer[(In, TrainIn), Out],
    val trainInput: Input[TrainIn],
    val trainInputLayer: Layer[TrainIn, TrainOut],
    val loss: Layer[(Out, TrainOut), Output[Loss]],
    val optimizer: Optimizer,
    val clipGradients: ClipGradients = NoClipGradients,
    override protected val colocateGradientsWithOps: Boolean = false
)(implicit
    evIn: OutputStructure.Aux[In, _, _],
    evTrainIn: OutputStructure.Aux[TrainIn, _, _]
) extends SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss] {
  override def buildInferOps(): Model.InferOps[In, Out] = {
    implicit val mode: Mode = INFERENCE

    val inputIterator = input()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext)
    Model.InferOps(inputIterator, inputIteratorNext, layerOutput)
  }

  override def buildTrainOps(): Model.SupervisedTrainOps[In, TrainIn, TrainOut, Out, Loss] = {
    implicit val mode: Mode = TRAINING

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = trainLayer(inputIteratorNext)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2)
    val lossOutput = loss((layerOutput, trainLayerOutput))
    val iteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
    val gradientsAndVariables = optimizer.computeGradients(
      lossOutput, colocateGradientsWithOps = colocateGradientsWithOps)
    val clippedGradientsAndVariables = clipGradients(gradientsAndVariables)
    val trainOp = optimizer.applyGradients(clippedGradientsAndVariables, Some(iteration))
    Model.SupervisedTrainOps(
      inputIterator, inputIteratorNext, layerOutput, trainLayerOutput, lossOutput, gradientsAndVariables, trainOp)
  }

  override def buildEvaluateOps(metrics: Seq[Metric[(Out, TrainOut), Output[Float]]]): Model.EvaluateOps[(In, TrainIn), Out] = {
    implicit val mode: Mode = EVALUATION

    val inputIterator = input.zip(trainInput).apply()
    val inputIteratorNext = inputIterator.next()
    val layerOutput = layer(inputIteratorNext._1)
    val trainLayerOutput = trainInputLayer(inputIteratorNext._2)
    val streamingInstances = metrics.map(_.streaming((layerOutput, trainLayerOutput)))
    Model.EvaluateOps(
      inputIterator, inputIteratorNext, layerOutput,
      streamingInstances.map(_.value), streamingInstances.map(_.update), streamingInstances.map(_.reset).toSet)
  }
}
