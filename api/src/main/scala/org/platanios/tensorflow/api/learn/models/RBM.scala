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

package org.platanios.tensorflow.api.learn.models

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.Indexer.NewAxis
import org.platanios.tensorflow.api.learn.{Counter, Model, UnsupervisedTrainableModel}
import org.platanios.tensorflow.api.learn.layers.Input
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.ops.variables.{RandomNormalInitializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output, Random}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

import scala.collection.mutable

class RBM(
    val input: Input[Tensor, Output, DataType, Shape],
    val numHidden: Int,
    val meanField: Boolean = true,
    val numSamples: Int = 100,
    val meanFieldCD: Boolean = false,
    val cdSteps: Int = 1,
    val optimizer: Optimizer,
    val name: String = "RBM"
) extends UnsupervisedTrainableModel[Tensor, Output, DataType, Shape, Output] {
  type InferOps = Model.InferenceOps[Tensor, Output, DataType, Shape, Output]
  type TrainOps = Model.UnsupervisedTrainingOps[Tensor, Output, DataType, Shape, Output]
  type EvalOps = Model.EvaluationOps[Tensor, Output, DataType, Shape, Output]

  val dataType: DataType = input.dataType
  val numInputs: Int = input.shape(1)

  private[this] val nextInputCache: mutable.Map[Graph, Output] = mutable.Map.empty
  private[this] val variablesCache: mutable.Map[Graph, (Variable, Variable, Variable)] = mutable.Map.empty
  private[this] val inferOpsCache: mutable.Map[Graph, InferOps] = mutable.Map.empty
  private[this] val trainOpsCache: mutable.Map[Graph, TrainOps] = mutable.Map.empty
  private[this] val evalOpsCache: mutable.Map[Graph, EvalOps] = mutable.Map.empty

  override def buildInferenceOps(graph: Graph = Op.currentGraph): InferOps = {
    inferOpsCache.getOrElseUpdate(graph, {
      Op.createWith(graph) {
        val inputIterator = input()
        val nextInput = nextInputCache.getOrElseUpdate(graph, inputIterator.next())
        val (vb, hb, w) = variables(graph)
        // Use the mean field approximation or contrastive divergence to compute the hidden values.
        var hProb = RBM.conditionalHGivenV(nextInput, hb, w)
        val output = {
          if (meanField) {
            hProb
          } else {
            var i = 0
            var hSamples = List.empty[Output]
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
        Model.InferenceOps(inputIterator, nextInput, output)
      }
    })
  }

  override def buildTrainingOps(graph: Graph = Op.currentGraph): TrainOps = {
    trainOpsCache.getOrElseUpdate(graph, {
      val inferOps = buildInferenceOps(graph)
      Op.createWith(graph) {
        val (vb, hb, w) = variables(graph)
        val vSample = contrastiveDivergence(inferOps.input, vb, hb, w)
        val vFreeEnergy = RBM.freeEnergy(inferOps.input, vb, hb, w)
        val vSampleFreeEnergy = RBM.freeEnergy(vSample, vb, hb, w)
        val loss = Math.mean(vFreeEnergy - vSampleFreeEnergy)
        val step = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
        val trainOp = optimizer.minimize(loss, iteration = Some(step))
        Model.UnsupervisedTrainingOps(inferOps.inputIterator, inferOps.input, inferOps.output, loss, trainOp)
      }
    })
  }

  override def buildEvaluationOps(
      metrics: Seq[Metric[Output, Output]], graph: Graph = Op.currentGraph
  ): EvalOps = {
    evalOpsCache.getOrElseUpdate(graph, {
      val inferOps = buildInferenceOps(graph)
      Op.createWith(graph) {
        val (mValues, mUpdates, mResets) = metrics.map(_.streaming(inferOps.output)).unzip3
        Model.EvaluationOps(inferOps.inputIterator, inferOps.input, inferOps.output, mValues, mUpdates, mResets)
      }
    })
  }

  private[this] def variables(graph: Graph = Op.currentGraph): (Variable, Variable, Variable) = {
    variablesCache.getOrElseUpdate(graph, {
      val vb = Variable.getVariable(s"$name/VisibleBias", dataType, Shape(numInputs), ZerosInitializer)
      val hb = Variable.getVariable(s"$name/HiddenBias", dataType, Shape(numHidden), ZerosInitializer)
      val w = Variable.getVariable(
        s"$name/Weights", dataType, Shape(numInputs, numHidden), RandomNormalInitializer(0.0f, 0.01f))
      (vb, hb, w)
    })
  }

  /** Runs a `k`-step Gibbs sampling chain to sample from the probability distribution of an RBM. */
  private[this] def contrastiveDivergence(initialV: Output, vb: Variable, hb: Variable, w: Variable): Output = {
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
  def apply(
      input: Input[Tensor, Output, DataType, Shape],
      numHidden: Int,
      meanField: Boolean = true,
      numSamples: Int = 100,
      meanFieldCD: Boolean = false,
      cdSteps: Int = 1,
      optimizer: Optimizer,
      name: String = "RBM"
  ): RBM = {
    new RBM(input, numHidden, meanField, numSamples, meanFieldCD, cdSteps, optimizer, name)
  }

  private[RBM] def conditionalHGivenV(v: Output, hb: Variable, w: Variable): Output = {
    Math.sigmoid(Math.add(hb.value, Math.matmul(v, w.value)))
  }

  private[RBM] def conditionalVGivenH(h: Output, vb: Variable, w: Variable): Output = {
    Math.sigmoid(Math.add(vb.value, Math.matmul(h, w.value, transposeB = true)))
  }

  private[RBM] def sampleBinary(p: Output): Output = {
    NN.relu(Math.sign(p - Random.randomUniform(p.dataType, p.shape, 0, 1)))
  }

  private[RBM] def freeEnergy(v: Output, vb: Variable, hb: Variable, w: Variable): Output = {
    val condTerm = -Math.sum(Math.log(1 + Math.exp(Math.add(hb.value, Math.matmul(v, w.value)))), axes = 1, keepDims = true)
    val biasTerm = -Math.matmul(v, Basic.transpose(vb.value(NewAxis)))
    Math.add(condTerm, biasTerm)
  }
}
