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

package org.platanios.tensorflow.api.learn.models

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.Indexer.NewAxis
import org.platanios.tensorflow.api.core.types.{IsInt32OrInt64OrFloat16OrFloat32OrFloat64, IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.learn.{Counter, Model, UnsupervisedTrainableModel}
import org.platanios.tensorflow.api.learn.layers.Input
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.variables.{RandomNormalInitializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output, Random}

import scala.collection.mutable

class RBM[T: TF : IsInt32OrInt64OrFloat16OrFloat32OrFloat64](
    val input: Input[Output[T]],
    val numHidden: Int,
    val meanField: Boolean = true,
    val numSamples: Int = 100,
    val meanFieldCD: Boolean = false,
    val cdSteps: Int = 1,
    val optimizer: Optimizer,
    val name: String = "RBM"
) extends UnsupervisedTrainableModel[Output[T], Output[T], Float] {
  type InferOps = Model.InferOps[Output[T], Output[T]]
  type TrainOps = Model.UnsupervisedTrainOps[Output[T], Output[T], Float]
  type EvalOps = Model.EvaluateOps[Output[T], Output[T]]

  val numInputs: Int = input.shape.apply(1)

  protected val nextInputCache: mutable.Map[Graph, Output[T]]                               = mutable.Map.empty
  protected val variablesCache: mutable.Map[Graph, (Variable[T], Variable[T], Variable[T])] = mutable.Map.empty
  protected val inferOpsCache : mutable.Map[Graph, InferOps]                                = mutable.Map.empty
  protected val trainOpsCache : mutable.Map[Graph, TrainOps]                                = mutable.Map.empty
  protected val evalOpsCache  : mutable.Map[Graph, EvalOps]                                 = mutable.Map.empty

  override def buildInferOps(): Model.InferOps[Output[T], Output[T]] = {
    inferOpsCache.getOrElseUpdate(Op.currentGraph, {
      val inputIterator = input()
      val nextInput = nextInputCache.getOrElseUpdate(Op.currentGraph, inputIterator.next())
      val (vb, hb, w) = variables()
      // Use the mean field approximation or contrastive divergence to compute the hidden values.
      var hProb = RBM.conditionalHGivenV(nextInput, hb, w)
      val output = {
        if (meanField) {
          hProb
        } else {
          var i = 0
          var hSamples = List.empty[Output[T]]
          while (i < numSamples) {
            val hSample = RBM.sampleBinary(hProb)
            val vProb = RBM.conditionalVGivenH(hSample, vb, w)
            val vSample = RBM.sampleBinary(vProb)
            hProb = RBM.conditionalHGivenV(vSample, hb, w)
            hSamples :+= hSample
            i += 1
          }
          Math.mean(Basic.stack(hSamples, axis = 0), axes = 0)
        }
      }
      Model.InferOps(inputIterator, nextInput, output)
    })
  }

  override def buildTrainOps(): Model.UnsupervisedTrainOps[Output[T], Output[T], Float] = {
    trainOpsCache.getOrElseUpdate(Op.currentGraph, {
      val inferOps = buildInferOps()
      val (vb, hb, w) = variables()
      val vSample = contrastiveDivergence(inferOps.input, vb, hb, w)
      val vFreeEnergy = RBM.freeEnergy(inferOps.input, vb, hb, w)
      val vSampleFreeEnergy = RBM.freeEnergy(vSample, vb, hb, w)
      val loss = Math.mean(vFreeEnergy - vSampleFreeEnergy).toFloat
      val step = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
      val gradientsAndVariables = optimizer.computeGradients(
        loss,
        colocateGradientsWithOps = colocateGradientsWithOps)
      val trainOp = optimizer.applyGradients(gradientsAndVariables, Some(step))
      Model.UnsupervisedTrainOps(
        inferOps.inputIterator, inferOps.input, inferOps.output,
        loss, gradientsAndVariables, trainOp)
    })
  }

  override def buildEvaluateOps(
      metrics: Seq[Metric[Output[T], Output[Float]]]
  ): Model.EvaluateOps[Output[T], Output[T]] = {
    evalOpsCache.getOrElseUpdate(Op.currentGraph, {
      val inferOps = buildInferOps()
      val streamingInstances = metrics.map(_.streaming(inferOps.output))
      Model.EvaluateOps(
        inferOps.inputIterator, inferOps.input, inferOps.output,
        streamingInstances.map(_.value), streamingInstances.map(_.update),
        streamingInstances.map(_.reset).toSet)
    })
  }

  protected def variables(): (Variable[T], Variable[T], Variable[T]) = {
    variablesCache.getOrElseUpdate(Op.currentGraph, {
      val vb = Variable.getVariable[T](s"$name/VisibleBias", Shape(numInputs), ZerosInitializer)
      val hb = Variable.getVariable[T](s"$name/HiddenBias", Shape(numHidden), ZerosInitializer)
      val w = Variable.getVariable[T](s"$name/Weights", Shape(numInputs, numHidden), RandomNormalInitializer(0.0f, 0.01f))
      (vb, hb, w)
    })
  }

  /** Runs a `k`-step Gibbs sampling chain to sample from the probability distribution of an RBM. */
  protected def contrastiveDivergence(
      initialV: Output[T],
      vb: Variable[T],
      hb: Variable[T],
      w: Variable[T]
  ): Output[T] = {
    var i = 0
    var v = initialV
    while (i < cdSteps) {
      val hProb = RBM.conditionalHGivenV(v, hb, w)
      val h = if (meanFieldCD) hProb else RBM.sampleBinary(hProb)
      val vProb = RBM.conditionalVGivenH(h, vb, w)
      v = if (meanFieldCD) vProb else RBM.sampleBinary(vProb)
      i += 1
    }
    Basic.stopGradient(v)
  }
}

object RBM {
  def apply[T: TF : IsInt32OrInt64OrFloat16OrFloat32OrFloat64](
      input: Input[Output[T]],
      numHidden: Int,
      meanField: Boolean = true,
      numSamples: Int = 100,
      meanFieldCD: Boolean = false,
      cdSteps: Int = 1,
      optimizer: Optimizer,
      name: String = "RBM"
  ): RBM[T] = {
    new RBM[T](input, numHidden, meanField, numSamples, meanFieldCD, cdSteps, optimizer, name)
  }

  private[RBM] def conditionalHGivenV[T: TF : IsNotQuantized](
      v: Output[T],
      hb: Variable[T],
      w: Variable[T]
  ): Output[T] = {
    Math.sigmoid(Math.add(hb.value, Math.matmul(v, w.value)))
  }

  private[RBM] def conditionalVGivenH[T: TF : IsNotQuantized](
      h: Output[T],
      vb: Variable[T],
      w: Variable[T]
  ): Output[T] = {
    Math.sigmoid(Math.add(vb.value, Math.matmul(h, w.value, transposeB = true)))
  }

  private[RBM] def sampleBinary[T: TF : IsInt32OrInt64OrFloat16OrFloat32OrFloat64](
      p: Output[T]
  ): Output[T] = {
    NN.relu(Math.sign(p - Random.randomUniform[T, Long](p.shape)))
  }

  private[RBM] def freeEnergy[T: TF : IsNotQuantized](
      v: Output[T],
      vb: Variable[T],
      hb: Variable[T],
      w: Variable[T]
  ): Output[T] = {
    val condTerm = -Math.sum(
      Math.log(Math.exp(hb.value + Math.matmul(v, w.value)) + Basic.ones[T](Shape())),
      axes = 1,
      keepDims = true)
    val biasTerm = -Math.matmul(v, Basic.transpose(vb.value(NewAxis)))
    Math.add(condTerm, biasTerm)
  }
}
