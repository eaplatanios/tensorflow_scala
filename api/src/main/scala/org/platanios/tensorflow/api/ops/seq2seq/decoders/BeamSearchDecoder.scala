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

package org.platanios.tensorflow.api.ops.seq2seq.decoders

import org.platanios.tensorflow.api.core.{NewAxis, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.{ControlFlow, WhileLoopVariable}
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps

// TODO: Abstract away the log-softmax/scoring function.

/** Recurrent Neural Network (RNN) that uses beam search to find the highest scoring sequence (i.e., perform decoding).
  *
  * @param  cell             RNN cell to use for decoding.
  * @param  initialCellState Initial RNN cell state to use for starting the decoding process.
  * @param  embeddingFn      Function that takes an `INT32` vector of IDs and returns the corresponding embedded values
  *                          that will be passed to the decoder input.
  * @param  beginTokens      `INT32` vector with length equal to the batch size, which contains the begin-of-sequence
  *                          token IDs.
  * @param  endToken         `INT32` scalar containing the end-of-sequence token ID (i.e., token ID which marks the end
  *                          of decoding).
  * @param  beamWidth        Beam width to use for the beam search while decoding.
  * @param  lengthPenalty    Length penalty method.
  * @param  outputLayer      Output layer to use that is applied at the outputs of the provided RNN cell before
  *                          returning them.
  * @param  name             Name prefix used for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BeamSearchDecoder[S, SS](
    override val cell: RNNCell[Output, Shape, S, SS],
    val initialCellState: S,
    val embeddingFn: (Output) => Output,
    val beginTokens: Output,
    val endToken: Output,
    val beamWidth: Int,
    val lengthPenalty: LengthPenalty = NoPenalty,
    val outputLayer: Output => Output = (o: Output) => o,
    override val name: String = "BeamSearchRNNDecoder"
)(implicit
    evOutput: WhileLoopVariable.Aux[Output, Shape],
    evS: WhileLoopVariable.Aux[S, SS]
) extends Decoder[
    Output, Shape, S, SS,
    BeamSearchDecoder.Output, (Shape, Shape, Shape),
    BeamSearchDecoder.State[S, SS], (SS, Shape, Shape, Shape),
    BeamSearchDecoder.FinalOutput, BeamSearchDecoder.State[S, SS]](cell, name) {
  // TODO: Handle the tiling inside the decoder itself.
  // evS.map(initialCellState, BeamSearchDecoder.tileBatch(_, beamWidth))
  private[this] val tiledInitialCellState: S = initialCellState

  if (evS.outputs(tiledInitialCellState).exists(_.rank == -1))
    throw InvalidArgumentException("All tensors in the state need to have known rank for the beam search decoder.")

  if (beginTokens.rank != 1)
    throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
  if (endToken.rank != 0)
    throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

  /** Scalar `INT32` tensor representing the batch size of the input values. */
  override val batchSize: Output = Op.createWithNameScope(name, Set(beginTokens.op)) {
    Basic.size(beginTokens)
  }

  private[this] val processedInitialCellState: S = Op.createWithNameScope(name, Set(batchSize.op)) {
    evS.mapWithShape(
      tiledInitialCellState, cell.stateShape, BeamSearchDecoder.maybeSplitBatchBeams(_, _, batchSize, beamWidth))
  }

  private[this] val processedBeginTokens: Output = Op.createWithNameScope(name) {
    Basic.tile(Basic.expandDims(beginTokens, 1), Basic.stack(Seq(1, beamWidth)))
  }

  private[this] val beginInput: Output = Op.createWithNameScope(name, Set(processedBeginTokens.op)) {
    embeddingFn(processedBeginTokens)
  }

  /** Describes whether the decoder keeps track of finished states.
    *
    * Most decoders will emit a true/false `finished` value independently at each time step. In this case, the
    * `dynamicDecode()` function keeps track of which batch entries have already finished, and performs a logical OR to
    * insert new batches to the finished set.
    *
    * Some decoders, however, shuffle batches/beams between time steps and `dynamicDecode()` will mix up the finished
    * state across these entries because it does not track the reshuffling across time steps. In this case, it is up to
    * the decoder to declare that it will keep track of its own finished state by setting this property to `true`.
    *
    * The beam-search decoder shuffles its beams and their finished state. For this reason, it conflicts with the
    * `dynamicDecode` function's tracking of finished states. Setting this property to `true` avoids early stopping of
    * decoding due to mismanagement of the finished state in `dynamicDecode`.
    */
  override val tracksOwnFinished: Boolean = true

  override def zeroOutput(): BeamSearchDecoder.Output = {
    // We assume the data type of the cell is the same as the initial cell state's first component tensor data type.
    val dataType = evS.outputs(tiledInitialCellState).head.dataType
    val zScores = evOutput.zero(batchSize, dataType, Shape(beamWidth), "ZeroScores")
    val zPredictedIDs = evOutput.zero(batchSize, INT32, Shape(beamWidth), "ZeroPredictedIDs")
    val zParentIDs = evOutput.zero(batchSize, INT32, Shape(beamWidth), "ZeroParentIDs")
    BeamSearchDecoder.Output(zScores, zPredictedIDs, zParentIDs)
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize(): (Output, Output, BeamSearchDecoder.State[S, SS]) = {
    Op.createWithNameScope(s"$name/Initialize", Set(batchSize.op)) {
      val finished = Basic.oneHot(
        indices = Basic.zeros(INT32, batchSize.expandDims(0)), depth = beamWidth,
        onValue = false, offValue = true, dataType = BOOLEAN)
      val dataType = evS.outputs(processedInitialCellState).head.dataType
      import dataType.supportedType
      val initialState = BeamSearchDecoder.State[S, SS](
        rnnState = processedInitialCellState,
        logProbabilities = Basic.oneHot(
          indices = Basic.zeros(INT32, batchSize.expandDims(0)), depth = beamWidth,
          onValue = Basic.constant(0, dataType), offValue = Basic.constant(dataType.min, dataType),
          dataType = dataType),
        finished = finished,
        sequenceLengths = Basic.zeros(INT64, Basic.stack(Seq(batchSize, beamWidth))))
      (finished, beginInput, initialState)
    }
  }

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) the decoder output for this step, (ii) the next decoder state, (iii) the next input,
    *         and (iv) a scalar `BOOLEAN` tensor specifying whether decoding has finished.
    */
  override def next(
      time: Output, input: Output, state: BeamSearchDecoder.State[S, SS]
  ): (BeamSearchDecoder.Output, BeamSearchDecoder.State[S, SS], Output, Output) = {
    Op.createWithNameScope(s"$name/Next") {
      val mergedInput = evOutput.mapWithShape(
        input, input.shape(2 ::), BeamSearchDecoder.mergeBatchBeams(_, _, batchSize, beamWidth))
      val mergedCellState = evS.mapWithShape(
        state.rnnState, cell.stateShape, BeamSearchDecoder.maybeMergeBatchBeams(_, _, batchSize, beamWidth))
      val mergedNextTuple = cell(Tuple(mergedInput, mergedCellState))
      val nextTupleOutput = outputLayer(evOutput.mapWithShape(
        mergedNextTuple.output, mergedNextTuple.output.shape(1 ::),
        BeamSearchDecoder.splitBatchBeams(_, _, batchSize, beamWidth)))
      val nextTupleState = evS.mapWithShape(
        mergedNextTuple.state, cell.stateShape, BeamSearchDecoder.maybeSplitBatchBeams(_, _, batchSize, beamWidth))

      // Perform the beam search step
      val staticBatchSize = Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1)

      // Calculate the current lengths of the predictions
      val predictionLengths = state.sequenceLengths
      val previouslyFinished = state.finished

      // Calculate the total log probabilities for the new hypotheses (final shape = [batchSize, beamWidth, vocabSize])
      val stepLogProbabilities = BeamSearchDecoder.maskLogProbabilities(
        NN.logSoftmax(nextTupleOutput), endToken, previouslyFinished)
      val totalLogProbabilities = state.logProbabilities.expandDims(2) + stepLogProbabilities

      // Calculate the continuation lengths by adding to all continuing beams
      val vocabSize = {
        if (nextTupleOutput.shape(-1) != -1)
          Basic.constant(nextTupleOutput.shape(-1))
        else
          Basic.shape(nextTupleOutput)(-1)
      }

      var lengthsToAdd = Basic.oneHot(
        indices = Basic.fill(endToken.dataType, Basic.stack(Seq(batchSize, beamWidth)))(endToken),
        depth = vocabSize, onValue = 0L, offValue = 1L, dataType = INT64)
      val addMask = Math.logicalNot(previouslyFinished).cast(INT64)
      val newPredictionLengths = Math.add(lengthsToAdd * addMask.expandDims(2), predictionLengths.expandDims(2))

      // Calculate the scores for each beam
      val scores = lengthPenalty(totalLogProbabilities, newPredictionLengths)

      // During the first time step we only consider the initial beam
      val scoresFlat = Basic.reshape(scores, Basic.stack(Seq(batchSize, -1)))

      // Pick the next beams according to the specified successors function
      val nextBeamSize = Basic.constant(beamWidth, INT32, name = "BeamWidth")
      val (nextBeamScores, wordIndices) = NN.topK(scoresFlat, nextBeamSize)
      nextBeamScores.setShape(Shape(staticBatchSize, beamWidth))
      wordIndices.setShape(Shape(staticBatchSize, beamWidth))

      // Pick out the log probabilities, beam indices, and states according to the chosen predictions
      val nextBeamLogProbabilities = evOutput.map(totalLogProbabilities, BeamSearchDecoder.gather(
        wordIndices, _, batchSize, vocabSize * beamWidth, Seq(-1), name = "NextBeamLogProbabilities"))
      val nextPredictedIDs = Math.mod(wordIndices, vocabSize, name = "NextBeamPredictedIDs").cast(INT32)
      val nextParentIDs = Math.divide(wordIndices, vocabSize, name = "NextBeamParentIDs").cast(INT32)

      // Append the new IDs to the current predictions
      val gatheredFinished = evOutput.map(previouslyFinished, BeamSearchDecoder.gather(
          nextParentIDs, _, batchSize, beamWidth, Seq(-1), name = "NextBeamFinishedGather"))
      val nextFinished = Math.logicalOr(
        gatheredFinished, Math.equal(nextPredictedIDs, endToken), name = "NextBeamFinished")

      // Calculate the length of the next predictions:
      //   1. Finished beams remain unchanged.
      //   2. Beams that just finished (i.e., `endToken` predicted) have their length increased by 1.
      //   3. Beams that have not yet finished have their length increased by 1.
      lengthsToAdd = Math.logicalNot(gatheredFinished).cast(INT64)
      var nextPredictionLengths = evOutput.map(state.sequenceLengths, BeamSearchDecoder.gather(
        nextParentIDs, _, batchSize, beamWidth, Seq(-1), name = "NextBeamLengthsGather"))
      nextPredictionLengths = nextPredictionLengths + lengthsToAdd

      // Pick out the cell state according to the next beam parent IDs. We use a different gather shape here because the
      // cell state tensors (i.e., the tensors that would be gathered from) all have rank greater than two and we
      // need to preserve those dimensions.
      val gatheredNextTupleState = evS.map(nextTupleState, BeamSearchDecoder.maybeGather(
        nextParentIDs, _, batchSize, beamWidth, Seq(batchSize * beamWidth, -1), name = "NextBeamStateGather"))

      val nextState = BeamSearchDecoder.State[S, SS](
        gatheredNextTupleState, nextBeamLogProbabilities, nextPredictionLengths, nextFinished)
      val output = BeamSearchDecoder.Output(nextBeamScores, nextPredictedIDs, nextParentIDs)

      val nextInput = ControlFlow.cond(
        nextFinished.all(),
        () => beginInput,
        () => embeddingFn(nextPredictedIDs))

      (output, nextState, nextInput, nextFinished)
    }
  }

  /** Finalizes the output of the decoding process.
    *
    * @param  output Final output after decoding.
    * @param  state  Final state after decoding.
    * @return Finalized output and state to return from the decoding process.
    */
  override def finalize(
      output: BeamSearchDecoder.Output,
      state: BeamSearchDecoder.State[S, SS],
      sequenceLengths: Output
  ): (BeamSearchDecoder.FinalOutput, BeamSearchDecoder.State[S, SS], Output) = {
    // Get the maximum sequence length across all beams for each batch
    val maxSequenceLengths = state.sequenceLengths.max(Tensor(1)).cast(INT32)
    val predictedIDs = BeamSearchDecoder.gatherTree(
      output.predictedIDs, output.parentIDs, maxSequenceLengths, endToken)
    val finalOutput = BeamSearchDecoder.FinalOutput(predictedIDs, output)
    val finalState = state.copy[S, SS](rnnState = evS.map(
      state.rnnState, BeamSearchDecoder.maybeSortTensorArrayBeams(_, state.sequenceLengths, output.parentIDs)))
    (finalOutput, finalState, finalState.sequenceLengths)
  }
}

object BeamSearchDecoder {
  def apply[S, SS](
      cell: RNNCell[ops.Output, Shape, S, SS],
      initialCellState: S,
      embeddingFn: (ops.Output) => ops.Output,
      beginTokens: ops.Output,
      endToken: ops.Output,
      beamWidth: Int,
      lengthPenalty: LengthPenalty = NoPenalty,
      outputLayer: ops.Output => ops.Output = (o: ops.Output) => o,
      name: String = "BeamSearchRNNDecoder"
  )(implicit
      evOutput: WhileLoopVariable.Aux[ops.Output, Shape],
      evS: WhileLoopVariable.Aux[S, SS]
  ): BeamSearchDecoder[S, SS] = {
    new BeamSearchDecoder[S, SS](
      cell, initialCellState, embeddingFn, beginTokens, endToken, beamWidth, lengthPenalty, outputLayer, name)
  }

  case class Output(scores: ops.Output, predictedIDs: ops.Output, parentIDs: ops.Output)

  object Output {
    implicit def outputWhileLoopVariable(implicit
        evOutput: WhileLoopVariable.Aux[ops.Output, Shape]
    ): WhileLoopVariable.Aux[Output, (Shape, Shape, Shape)] = {
      new WhileLoopVariable[Output] {
        override type ShapeType = (Shape, Shape, Shape)

        override def zero(
            batchSize: ops.Output, dataType: DataType, shape: (Shape, Shape, Shape), name: String = "Zero"
        ): Output = Op.createWithNameScope(name) {
          Output(
            evOutput.zero(batchSize, dataType, shape._1, "Scores"),
            evOutput.zero(batchSize, dataType, shape._2, "PredictedIDs"),
            evOutput.zero(batchSize, dataType, shape._3, "ParentIDs"))
        }

        override def size(value: Output): Int = {
          evOutput.size(value.scores) + evOutput.size(value.predictedIDs) + evOutput.size(value.parentIDs)
        }

        override def outputs(value: Output): Seq[ops.Output] = {
          evOutput.outputs(value.scores) ++ evOutput.outputs(value.predictedIDs) ++ evOutput.outputs(value.parentIDs)
        }

        override def shapes(shape: (Shape, Shape, Shape)): Seq[Shape] = {
          evOutput.shapes(shape._1) ++ evOutput.shapes(shape._2) ++ evOutput.shapes(shape._3)
        }

        override def segmentOutputs(value: Output, values: Seq[ops.Output]): (Output, Seq[ops.Output]) = {
          (Output(values(0), values(1), values(2)), values.drop(3))
        }

        override def segmentShapes(value: Output, shapes: Seq[Shape]): ((Shape, Shape, Shape), Seq[Shape]) = {
          ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
        }

        override def map(value: Output, mapFn: ops.Symbol => ops.Symbol): Output = {
          Output(
            evOutput.map(value.scores, mapFn),
            evOutput.map(value.predictedIDs, mapFn),
            evOutput.map(value.parentIDs, mapFn))
        }

        override def mapWithShape(
            value: Output, shape: (Shape, Shape, Shape), mapFn: (ops.Symbol, Shape) => ops.Symbol
        ): Output = {
          Output(
            evOutput.mapWithShape(value.scores, shape._1, mapFn),
            evOutput.mapWithShape(value.predictedIDs, shape._2, mapFn),
            evOutput.mapWithShape(value.parentIDs, shape._2, mapFn))
        }
      }
    }
  }

  case class State[S, SS](
      rnnState: S,
      logProbabilities: ops.Output,
      sequenceLengths: ops.Output,
      finished: ops.Output
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS]
  )

  object State {
    implicit def stateWhileLoopVariable[S, SS](implicit
        evOutput: WhileLoopVariable.Aux[ops.Output, Shape],
        evS: WhileLoopVariable.Aux[S, SS]
    ): WhileLoopVariable.Aux[State[S, SS], (SS, Shape, Shape, Shape)] = {
      new WhileLoopVariable[State[S, SS]] {
        override type ShapeType = (SS, Shape, Shape, Shape)

        override def zero(
            batchSize: ops.Output, dataType: DataType, shape: (SS, Shape, Shape, Shape), name: String = "Zero"
        ): State[S, SS] = Op.createWithNameScope(name) {
          State(
            evS.zero(batchSize, dataType, shape._1, "RNNState"),
            evOutput.zero(batchSize, dataType, shape._2, "LogProbabilities"),
            evOutput.zero(batchSize, dataType, shape._3, "SequenceLengths"),
            evOutput.zero(batchSize, dataType, shape._4, "Finished"))
        }

        override def size(value: State[S, SS]): Int = {
          evS.size(value.rnnState) + evOutput.size(value.logProbabilities) +
              evOutput.size(value.sequenceLengths) + evOutput.size(value.finished)
        }

        override def outputs(value: State[S, SS]): Seq[ops.Output] = {
          evS.outputs(value.rnnState) ++ evOutput.outputs(value.logProbabilities) ++
              evOutput.outputs(value.sequenceLengths) ++ evOutput.outputs(value.finished)
        }

        override def shapes(shape: (SS, Shape, Shape, Shape)): Seq[Shape] = {
          evS.shapes(shape._1) ++ evOutput.shapes(shape._2) ++ evOutput.shapes(shape._3) ++ evOutput.shapes(shape._4)
        }

        override def segmentOutputs(value: State[S, SS], values: Seq[ops.Output]): (State[S, SS], Seq[ops.Output]) = {
          val (rnnState, valuesTail) = evS.segmentOutputs(value.rnnState, values)
          (State(rnnState, valuesTail(0), valuesTail(1), valuesTail(2)), valuesTail.drop(3))
        }

        override def segmentShapes(value: State[S, SS], shapes: Seq[Shape]): ((SS, Shape, Shape, Shape), Seq[Shape]) = {
          val (rnnStateShape, shapesTail) = evS.segmentShapes(value.rnnState, shapes)
          ((rnnStateShape, shapesTail(0), shapesTail(1), shapesTail(2)), shapesTail.drop(3))
        }

        override def map(value: State[S, SS], mapFn: ops.Symbol => ops.Symbol): State[S, SS] = {
          State(
            evS.map(value.rnnState, mapFn),
            evOutput.map(value.logProbabilities, mapFn),
            evOutput.map(value.sequenceLengths, mapFn),
            evOutput.map(value.finished, mapFn))
        }

        override def mapWithShape(
            value: State[S, SS], shape: (SS, Shape, Shape, Shape), mapFn: (ops.Symbol, Shape) => ops.Symbol
        ): State[S, SS] = {
          State(
            evS.mapWithShape(value.rnnState, shape._1, mapFn),
            evOutput.mapWithShape(value.logProbabilities, shape._2, mapFn),
            evOutput.mapWithShape(value.sequenceLengths, shape._3, mapFn),
            evOutput.mapWithShape(value.finished, shape._4, mapFn))
        }
      }
    }
  }

  /** Final outputs returned by the beam search after all decoding is finished.
    *
    * @param  predictedIDs Tensor of shape `[batchSize, T, beamWidth]` (or `[T, batchSize, beamWidth]`,
    *                      if `outputTimeMajor == true`) containing the final prediction IDs. The beams are ordered
    *                      from best to worst.
    * @param  output       State of the beam search at the end of decoding.
    */
  case class FinalOutput(predictedIDs: ops.Output, output: Output)

  object FinalOutput {
    implicit def finalOutputWhileLoopVariable(implicit
        evOpsOutput: WhileLoopVariable.Aux[ops.Output, Shape],
        evOutput: WhileLoopVariable.Aux[Output, (Shape, Shape, Shape)]
    ): WhileLoopVariable.Aux[FinalOutput, (Shape, (Shape, Shape, Shape))] = {
      new WhileLoopVariable[FinalOutput] {
        override type ShapeType = (Shape, (Shape, Shape, Shape))

        override def zero(
            batchSize: ops.Output, dataType: DataType, shape: (Shape, (Shape, Shape, Shape)), name: String = "Zero"
        ): FinalOutput = Op.createWithNameScope(name) {
          FinalOutput(
            evOpsOutput.zero(batchSize, dataType, shape._1, "PredictedIDs"),
            evOutput.zero(batchSize, dataType, shape._2, "BeamSearchOutput"))
        }

        override def size(value: FinalOutput): Int = {
          evOpsOutput.size(value.predictedIDs) + evOutput.size(value.output)
        }

        override def outputs(value: FinalOutput): Seq[ops.Output] = {
          evOpsOutput.outputs(value.predictedIDs) ++ evOutput.outputs(value.output)
        }

        override def shapes(shape: (Shape, (Shape, Shape, Shape))): Seq[Shape] = {
          evOpsOutput.shapes(shape._1) ++ evOutput.shapes(shape._2)
        }

        override def segmentOutputs(value: FinalOutput, values: Seq[ops.Output]): (FinalOutput, Seq[ops.Output]) = {
          val (output, valuesTail) = evOutput.segmentOutputs(value.output, values.tail)
          (FinalOutput(values(0), output), valuesTail)
        }

        override def segmentShapes(
            value: FinalOutput, shapes: Seq[Shape]
        ): ((Shape, (Shape, Shape, Shape)), Seq[Shape]) = {
          val (outputShape, shapesTail) = evOutput.segmentShapes(value.output, shapes.tail)
          ((shapes(0), outputShape), shapesTail)
        }

        override def map(value: FinalOutput, mapFn: ops.Symbol => ops.Symbol): FinalOutput = {
          FinalOutput(
            evOpsOutput.map(value.predictedIDs, mapFn),
            evOutput.map(value.output, mapFn))
        }

        override def mapWithShape(
            value: FinalOutput, shape: (Shape, (Shape, Shape, Shape)), mapFn: (ops.Symbol, Shape) => ops.Symbol
        ): FinalOutput = {
          FinalOutput(
            evOpsOutput.mapWithShape(value.predictedIDs, shape._1, mapFn),
            evOutput.mapWithShape(value.output, shape._2, mapFn))
        }
      }
    }
  }

  /** Masks log probabilities. The result is that finished beams allocate all probability mass to `endToken` and
    * unfinished beams remain unchanged.
    *
    * @param  logProbabilities Log probability for each hypothesis, which is a tensor with shape
    *                          `[batchSize, beamWidth, vocabSize]`.
    * @param  endToken         `INT32` scalar tensor containing the end-of-sequence token ID.
    * @param  finished         `BOOLEAN` tensor of shape `[batchSize, beamWidth]` that specifies which elements in the
    *                          beam have finished decoding.
    * @return Tensor of shape `[batchSize, beamWidth, vocabSize]`, where unfinished beams stay unchanged and finished
    *         beams are replaced with a tensor with all probability mass allocated to `endToken`.
    */
  private[BeamSearchDecoder] def maskLogProbabilities(
      logProbabilities: ops.Output, endToken: ops.Output, finished: ops.Output
  ): ops.Output = {
    val vocabSize = Basic.shape(logProbabilities)(2)
    // Finished examples are replaced with a vector that has all its probability mass on `endToken`
    val dType = logProbabilities.dataType
    import dType.supportedType
    val dTypeMin = Tensor(dType.min).slice(0)
    val finishedRow = Basic.oneHot(endToken, vocabSize, Basic.zeros(dType, Shape()), Basic.constant(dTypeMin))
    val finishedLogProbabilities = Basic.tile(
      finishedRow.reshape(Shape(1, 1, -1)), Basic.concatenate(Seq(Basic.shape(finished), Tensor(1)), 0))
    val finishedMask = Basic.tile(finished.expandDims(2), Basic.stack(Seq(1, 1, vocabSize)))
    Math.select(finishedMask, finishedLogProbabilities, logProbabilities)
  }

  /** The `gatherTree` op calculates the full beams from the per-step IDs and parent beam IDs.
    *
    * On a CPU, if an out-of-bounds parent ID is found, an error is returned. On a GPU, if an out-of-bounds parent ID
    * is found, a `-1` is stored in the corresponding output value and the execution for that beam returns early.
    *
    * For a given beam, past the time step containing the first decoded `endToken`, all values are filled in with
    * `endToken`.
    *
    * @param  stepIDs            Tensor with shape `[maxTime, batchSize, beamWidth]`, containing the step IDs.
    * @param  parentIDs          Tensor with shape `[maxTime, batchSize, beamWidth]`, containing the parent IDs.
    * @param  maxSequenceLengths Tensor with shape `[batchSize]`, containing the sequence lengths.
    * @param  endToken           Scalar tensor containing the end-of-sequence token ID.
    * @param  name               Name for the created op.
    * @return Created op output.
    */
  private[seq2seq] def gatherTree(
      stepIDs: ops.Output,
      parentIDs: ops.Output,
      maxSequenceLengths: ops.Output,
      endToken: ops.Output,
      name: String = "GatherTree"
  ): ops.Output = {
    Op.Builder("GatherTree", name)
        .addInput(stepIDs)
        .addInput(parentIDs)
        .addInput(maxSequenceLengths)
        .addInput(endToken)
        .build().outputs(0)
  }

  @throws[InvalidArgumentException]
  private[decoders] def tileBatch(value: ops.Symbol, multiplier: Int): ops.Symbol = {
    value match {
      case output: ops.Output =>
        if (output.rank == -1) {
          throw InvalidArgumentException("The provided tensor must have statically known rank.")
        } else if (output.rank == 0) {
          val tiling = Tensor(multiplier)
          val tiled = Basic.tile(output.expandDims(0), tiling)
          tiled.setShape(Shape(multiplier))
          tiled
        } else {
          val outputShape = Basic.shape(output)
          val tiling = ArrayBuffer.fill(output.rank + 1)(1)
          tiling(1) = multiplier
          val tiledStaticBatchSize = if (output.shape(0) != -1) output.shape(0) * multiplier else -1
          val tiled = Basic.tile(output.expandDims(1), tiling).reshape(
            Basic.concatenate(Seq((outputShape(0) * multiplier).expandDims(0), outputShape(1 ::))))
          if (output.rank > 1)
            tiled.setShape(Shape(tiledStaticBatchSize) ++ output.shape(1 ::))
          else
            tiled.setShape(Shape(tiledStaticBatchSize))
          tiled
        }
      case _: TensorArray => value
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  @throws[InvalidArgumentException]
  def tileForBeamSearch[S](value: S, beamWidth: Int)(implicit evS: WhileLoopVariable[S]): S = {
    evS.map(value, BeamSearchDecoder.tileBatch(_, beamWidth))
  }

  /** Maybe converts the provided tensor structure from batches by beams into batches of beams, by merging them
    * accordingly.
    *
    * More precisely, `value` consists of tensors with shape `[batchSize * beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize, beamWidth] ++ ...`.
    *
    * @param  value Value to reshape.
    * @param  shape Depth shape of the value.
    * @return Reshaped state.
    * @throws InvalidArgumentException If `value` is of an unsupported type, or if it contains any tensors of unknown
    *                                  rank, or if, after reshaping, the new tensor is not shaped
    *                                  `[batchSize, beamWidth] ++ ...` (assuming that both `batchSize` and `beamWidth`
    *                                  are known statically).
    */
  @throws[InvalidArgumentException]
  def maybeSplitBatchBeams(value: ops.Symbol, shape: Shape, batchSize: ops.Output, beamWidth: Int): ops.Symbol = {
    value match {
      case output: ops.Output =>
        if (output.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($output) to have known rank, but it was unknown.")
        else if (output.rank == 0)
          value
        else
          splitBatchBeams(value, shape, batchSize, beamWidth)
      case _: TensorArray => value
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Converts the provided tensor structure from batches by beams into batches of beams, by merging them accordingly.
    *
    * More precisely, `value` consists of tensors with shape `[batchSize * beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize, beamWidth] ++ ...`.
    *
    * @param  value Value to reshape.
    * @param  shape Depth shape of the value.
    * @return Reshaped state.
    * @throws InvalidArgumentException If `value` is of an unsupported type.
    * @throws InvalidShapeException    If the provided value contains any tensors of unknown rank, or if, after
    *                                  reshaping, the new tensor is not shaped `[batchSize, beamWidth] ++ ...`
    *                                  (assuming that both `batchSize` and `beamWidth` are known statically).
    */
  @throws[InvalidArgumentException]
  @throws[InvalidShapeException]
  private[BeamSearchDecoder] def splitBatchBeams(
      value: ops.Symbol, shape: Shape, batchSize: ops.Output, beamWidth: Int
  ): ops.Symbol = {
    (value, shape) match {
      case (output: ops.Output, s: Shape) =>
        val valueShape = Basic.shape(output)
        val reshapedValue = Basic.reshape(output, Basic.concatenate(Seq(
          batchSize(NewAxis), Tensor(batchSize.dataType, beamWidth).toOutput,
          valueShape(1 ::).cast(batchSize.dataType)), axis = 0))
        val staticBatchSize = ops.Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1)
        val expectedReshapedShape = Shape(staticBatchSize, beamWidth) ++ s
        if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape))
          throw InvalidShapeException(
            "Unexpected behavior when reshaping between beam width and batch size. " +
                s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
                s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
                "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
        reshapedValue.setShape(expectedReshapedShape)
        reshapedValue
      case (_: TensorArray, _) => value
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Maybe converts the provided tensor structure from a batch of beams into a batch by beams, by merging them
    * accordingly.
    *
    * More precisely, `value` consists of tensors with shape `[batchSize, beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize * beamWidth] ++ ...`.
    *
    * @param  value Value to reshape.
    * @param  shape Depth shape of the value.
    * @return Reshaped state.
    * @throws InvalidArgumentException If `value` is of an unsupported type, or if it contains any tensors of unknown
    *                                  rank, or if, after reshaping, the new tensor is not shaped
    *                                  `[batchSize * beamWidth] ++ ...` (assuming that both `batchSize` and `beamWidth`
    *                                  are known statically).
    */
  @throws[InvalidArgumentException]
  private[BeamSearchDecoder] def maybeMergeBatchBeams(
      value: ops.Symbol, shape: Shape, batchSize: ops.Output, beamWidth: Int
  ): ops.Symbol = {
    value match {
      case output: ops.Output =>
        if (output.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($output) to have known rank, but it was unknown.")
        else if (output.rank == 0)
          value
        else
          mergeBatchBeams(value, shape, batchSize, beamWidth)
      case _: TensorArray => value
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Converts the provided tensor structure from a batch of beams into a batch by beams, by merging them accordingly.
    *
    * More precisely, `value` consists of tensors with shape `[batchSize, beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize * beamWidth] ++ ...`.
    *
    * @param  value Value to reshape.
    * @param  shape Depth shape of the value.
    * @return Reshaped state.
    * @throws InvalidArgumentException If `value` is of an unsupported type.
    * @throws InvalidShapeException    If the provided value contains any tensors of unknown rank, or if, after
    *                                  reshaping, the new tensor is not shaped `[batchSize * beamWidth] ++ ...`
    *                                  (assuming that both `batchSize` and `beamWidth` are known statically).
    */
  @throws[InvalidArgumentException]
  @throws[InvalidShapeException]
  private[BeamSearchDecoder] def mergeBatchBeams(
      value: ops.Symbol, shape: Shape, batchSize: ops.Output, beamWidth: Int
  ): ops.Symbol = {
    (value, shape) match {
      case (output: ops.Output, s: Shape) =>
        val valueShape = Basic.shape(output)
        val reshapedValue = Basic.reshape(output, Basic.concatenate(Seq(
          batchSize(NewAxis) * Tensor(batchSize.dataType, beamWidth).toOutput,
          valueShape(2 ::).cast(batchSize.dataType)), axis = 0))
        val staticBatchSize = ops.Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1)
        val batchSizeBeamWidth = if (staticBatchSize != -1) staticBatchSize * beamWidth else -1
        val expectedReshapedShape = Shape(batchSizeBeamWidth) ++ s
        if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape))
          throw InvalidShapeException(
            "Unexpected behavior when reshaping between beam width and batch size. " +
                s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
                s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
                "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
        reshapedValue.setShape(expectedReshapedShape)
        reshapedValue
      case (_: TensorArray, _) => value
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Maybe gathers the right indices from the provided `gatherFrom` value. This works by reshaping all tensors in
    * `gatherFrom` to `gatherShape` (e.g., `Seq(-1)`) and then gathering from that according to the `gatherIndices`,
    * which are offset by the right amount in order to preserve the batch order.
    *
    * @param  gatherIndices Indices that we use to gather from `gatherFrom`.
    * @param  gatherFrom    Value to gather from.
    * @param  batchSize     Input batch size.
    * @param  rangeSize     Number of values in each range. Likely equal to the beam width.
    * @param  gatherShape   What we should reshape `gatherFrom` to in order to preserve the correct values. An example
    *                       is when `gatherFrom` is the attention from an `AttentionWrapperState` with shape
    *                       `[batchSize, beamWidth, attentionSize]`. There, we want to preserve the `attentionSize`
    *                       elements, and so `gatherShape` is set to `Seq(batchSize * beamWidth, -1)`. Then, upon
    *                       reshape, we still have the `attentionSize` elements, as desired.
    * @return Value containing the gathered tensors of shapes `tf.shape(gatherFrom)(0 :: 1 + gatherShape.size())`.
    * @throws InvalidArgumentException If `gatherFrom` is of an unsupported type, or if it contains any tensors of
    *                                  unknown rank.
    */
  @throws[InvalidArgumentException]
  private[BeamSearchDecoder] def maybeGather(
      gatherIndices: ops.Output,
      gatherFrom: ops.Symbol,
      batchSize: ops.Output,
      rangeSize: ops.Output,
      gatherShape: Seq[ops.Output],
      name: String = "GatherTensorHelper"
  ): ops.Symbol = {
    gatherFrom match {
      case gatherFromOutput: ops.Output =>
        if (gatherFromOutput.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($gatherFromOutput) to have known rank, but it was unknown.")
        else if (gatherFromOutput.rank < gatherShape.size)
          gatherFrom
        else
          gather(gatherIndices, gatherFrom, batchSize, rangeSize, gatherShape, name)
      case _: TensorArray => gatherFrom
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Gathers the right indices from the provided `gatherFrom` value. This works by reshaping all tensors in
    * `gatherFrom` to `gatherShape` (e.g., `Seq(-1)`) and then gathering from that according to the `gatherIndices`,
    * which are offset by the right amount in order to preserve the batch order.
    *
    * @param  gatherIndices Indices that we use to gather from `gatherFrom`.
    * @param  gatherFrom    Value to gather from.
    * @param  batchSize     Input batch size.
    * @param  rangeSize     Number of values in each range. Likely equal to the beam width.
    * @param  gatherShape   What we should reshape `gatherFrom` to in order to preserve the correct values. An example
    *                       is when `gatherFrom` is the attention from an `AttentionWrapperState` with shape
    *                       `[batchSize, beamWidth, attentionSize]`. There, we want to preserve the `attentionSize`
    *                       elements, and so `gatherShape` is set to `Seq(batchSize * beamWidth, -1)`. Then, upon
    *                       reshape, we still have the `attentionSize` elements, as desired.
    * @return Value containing the gathered tensors of shapes `tf.shape(gatherFrom)(0 :: 1 + gatherShape.size())`.
    * @throws InvalidArgumentException If `gatherFrom` is of an unsupported type.
    */
  @throws[InvalidArgumentException]
  private[BeamSearchDecoder] def gather(
      gatherIndices: ops.Output,
      gatherFrom: ops.Symbol,
      batchSize: ops.Output,
      rangeSize: ops.Output,
      gatherShape: Seq[ops.Output],
      name: String = "GatherTensorHelper"
  ): ops.Symbol = {
    gatherFrom match {
      case gatherFromOutput: ops.Output =>
        Op.createWithNameScope(name) {
          val range = (Math.range(0, batchSize) * rangeSize).expandDims(1)
          val reshapedGatherIndices = (gatherIndices + range).reshape(Shape(-1))
          var output = Basic.gather(gatherFromOutput.reshape(Basic.stack(gatherShape)), reshapedGatherIndices)
          val finalShape = Basic.shape(gatherFromOutput)(0 :: (1 + gatherShape.size))
          val staticBatchSize = ops.Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1)
          val finalStaticShape = Shape(staticBatchSize) ++ gatherFromOutput.shape(1 :: (1 + gatherShape.size))
          output = Basic.reshape(output, finalShape, name = "Output")
          output.setShape(finalStaticShape)
          output
        }
      case _: TensorArray => gatherFrom
      case _ => throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  private[BeamSearchDecoder] def maybeSortTensorArrayBeams(
      value: ops.Symbol, sequenceLengths: ops.Output, parentIDs: ops.Output
  ): ops.Symbol = {
    value match {
      case ta: TensorArray =>
        val maxTime = Basic.shape(parentIDs)(0)
        val batchSize = Basic.shape(parentIDs)(1)
        val beamWidth = Basic.shape(parentIDs)(2)

        // Generate beam IDs that will be reordered by the `gatherTree` op
        val beamIDs = Basic.tile(Math.range(0, beamWidth)(NewAxis, NewAxis), Basic.stack(Seq(maxTime, batchSize, 1)))
        val mask = Basic.sequenceMask(sequenceLengths, maxTime, INT32).transpose(Seq(2, 0, 1))

        // Use `beamWidth + 1` to mark the end of the beam
        val maskedBeamIDs = (beamIDs * mask) + (1 - mask) * (beamWidth + 1)
        val maxSequenceLengths = sequenceLengths.max(Tensor(1)).cast(INT32)
        var sortedBeamIDs = gatherTree(maskedBeamIDs, parentIDs, maxSequenceLengths, beamWidth + 1)

        // For out of range steps, we simply copy the same beam
        sortedBeamIDs = Math.select(mask.cast(BOOLEAN), sortedBeamIDs, beamIDs)

        // Gather from each tensor in `value` according to `sortedBeamIDs`
        val size = ta.size()
        val collector = TensorArray.create(size, ta.dataType, dynamicSize = false)
        val collected = ControlFlow.whileLoop(
          (loopVariables: (ops.Output, TensorArray)) => loopVariables._1 < size,
          (loopVariables: (ops.Output, TensorArray)) => {
            val i = loopVariables._1
            val gathered = gather(
              sortedBeamIDs.gather(i), ta.read(i), batchSize, beamWidth, Seq(batchSize * beamWidth, -1))
            (i + 1, loopVariables._2.write(i, gathered.asInstanceOf[ops.Output]))
          },
          (Basic.constant(0), collector),
          parallelIterations = 1)
        collected._2
      case _ => value
    }
  }
}
