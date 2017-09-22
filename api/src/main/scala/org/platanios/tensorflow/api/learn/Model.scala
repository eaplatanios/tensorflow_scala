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

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait Model

object Model {
  case class InferenceOps[IT, IO, ID, IS, I](input: tf.Iterator[IT, IO, ID, IS], output: I)

  case class TrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      input: tf.Iterator[(IT, TT), (IO, TO), (ID, TD), (IS, TS)],
      output: I,
      trainOutput: T,
      loss: tf.Output,
      trainOp: tf.Op)

  object TrainOps {
    def apply[IT, IO, ID, IS, I](
        input: tf.Iterator[(IT, IT), (IO, IO), (ID, ID), (IS, IS)],
        output: I,
        loss: tf.Output,
        trainOp: tf.Op): TrainOps[IT, IO, ID, IS, I, IT, IO, ID, IS, I] = {
      TrainOps(input, output, output, loss, trainOp)
    }
  }

  trait API {
    def model[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
        input: Input[IT, IO, ID, IS],
        layer: Layer[IO, I],
        trainingInput: Input[TT, TO, TD, TS],
        trainingInputLayer: Layer[TO, T],
        loss: Layer[(I, T), tf.Output],
        optimizer: tf.train.Optimizer): TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
      new TrainableModel(input, layer, trainingInput, trainingInputLayer, loss, optimizer)
    }
  }

  object API extends API
}

class InferenceModel[IT, IO, ID, IS, I] private[learn](
    val input: Input[IT, IO, ID, IS],
    val layer: Layer[IO, I]) extends Model {
  def buildInferenceOps(graph: Graph = tf.currentGraph): Model.InferenceOps[IT, IO, ID, IS, I] = {
    tf.createWith(graph) {
      val tfInput = input()
      val tfOutput = layer(tfInput.next())
      Model.InferenceOps(tfInput, tfOutput)
    }
  }
}

class TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] private[learn](
    override val input: Input[IT, IO, ID, IS],
    override val layer: Layer[IO, I],
    val trainInput: Input[TT, TO, TD, TS],
    val trainLayer: Layer[TO, T],
    val loss: Layer[(I, T), tf.Output],
    val optimizer: tf.train.Optimizer) extends InferenceModel[IT, IO, ID, IS, I](input, layer) {
  def buildTrainOps(graph: Graph = tf.currentGraph): Model.TrainOps[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    tf.createWith(graph = graph) {
      val tfInput = input.zip(trainInput).apply()
      val tfInputNext = tfInput.next()
      val tfOutput = layer(tfInputNext._1)
      val tfTrainingOutput = trainLayer(tfInputNext._2)
      // TODO: [LEARN] !!! Remove this cast.
      val tfLoss = tf.cast(loss((tfOutput, tfTrainingOutput)), FLOAT32, name = "LearnLossCast")
      val tfIteration = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, graph)
      val tfTrainOp = optimizer.minimize(tfLoss, iteration = Some(tfIteration))
      Model.TrainOps(tfInput, tfOutput, tfTrainingOutput, tfLoss, tfTrainOp)
    }
  }
}
