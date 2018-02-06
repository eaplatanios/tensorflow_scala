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

package org.platanios.tensorflow.examples

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.rnn.RNN
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{BasicLSTMCell, DropoutWrapper, LSTMTuple}
import org.platanios.tensorflow.data.text.PTBLoader

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
object RNNTutorialUsingPTB {
  private[this] val logger = Logger(LoggerFactory.getLogger("Tutorials / RNN-PTB"))

  val batchSize   : Int = 20
  val numSteps    : Int = 20
  val prefetchSize: Int = 10

  val dataType              : DataType = FLOAT32
  val vocabularySize        : Int      = 10000
  val numHidden             : Int      = 200
  val numLayers             : Int      = 1
  val dropoutKeepProbability: Float    = 0.5f

  object RNNOutputLayer extends tf.learn.Layer[LSTMTuple, Output]("RNNOutputLayer") {
    override val layerType: String = "RNNOutputLayer"

    override protected def _forward(input: LSTMTuple, mode: tf.learn.Mode): Output = {
      val weights = tf.variable("OutputWeights", dataType, Shape(numHidden, vocabularySize))
      val bias = tf.variable("OutputBias", dataType, Shape(vocabularySize))
      val output = tf.linear(tf.reshape(input.output, Shape(-1, numHidden)), weights.value, bias.value)
      // We reshape the output logits to feed into the sequence loss layer
      tf.reshape(output, Shape(batchSize, numSteps, vocabularySize))
    }
  }

  private[this] val model = {
    val input = tf.learn.Input(INT32, Shape(-1, -1))
    val trainInput = tf.learn.Input(INT32, Shape(-1, -1))
    // Slightly better results can be obtained with forget gate biases initialized to 1 but the hyper-parameters of the
    // model would need to be different than those reported in the paper.
    val rnnCell = DropoutWrapper(
      "DropoutCell", BasicLSTMCell("LSTMCell", numHidden, FLOAT32, forgetBias = 0.0f), 0.00001f)
    // TODO: Add multi-RNN cell.
    val rnn = RNN("RNN", rnnCell, timeMajor = false)
    val layer = tf.learn.device("/device:CPU:0") {
      tf.learn.Embedding("Embedding", vocabularySize, numHidden, dataType)
    } >> tf.learn.Dropout("Embedding/Dropout", dropoutKeepProbability) >> rnn >> RNNOutputLayer
    val loss = tf.learn.SequenceLoss("Loss/SequenceLoss", averageAcrossTimeSteps = false, averageAcrossBatch = true) >>
        tf.learn.Sum("Loss/Sum") >>
        tf.learn.ScalarSummary("Loss/Summary", "Loss")
    val optimizer = tf.train.GradientDescent(1.0)
    tf.learn.Model(input, layer, trainInput, loss, optimizer, tf.learn.ClipGradientsByGlobalNorm(5.0f))
  }

  def main(args: Array[String]): Unit = {
    val dataset = PTBLoader.load(Paths.get("datasets/PTB"))
    val trainDataset =
      PTBLoader.tokensToBatchedTFDataset(dataset.train, batchSize, numSteps, "TrainDataset")
          .repeat()
          .prefetch(prefetchSize)

    val summariesDir = Paths.get("temp/rnn-ptb")
    val estimator = tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir)),
      tf.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(10)),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(10)),
        tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))
    estimator.train(() => trainDataset, tf.learn.StopCriteria(maxSteps = Some(10000)))
  }
}
