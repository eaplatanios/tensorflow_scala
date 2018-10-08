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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.{ControlFlow, WhileLoopVariable}
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.tensors.Tensor

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps

// TODO: Abstract away the log-softmax/scoring function.

/** Recurrent Neural Network (RNN) that uses beam search to find the highest scoring sequence (i.e., perform decoding).
  *
  * @param  cell                RNN cell to use for decoding.
  * @param  initialCellState    Initial RNN cell state to use for starting the decoding process.
  * @param  embeddingFn         Function that takes an vector of IDs and returns the corresponding embedded
  *                             values that will be passed to the decoder input.
  * @param  beginTokens         Vector with length equal to the batch size, which contains the begin-of-sequence
  *                             token IDs.
  * @param  endToken            Scalar containing the end-of-sequence token ID (i.e., token ID which marks the end of
  *                             decoding).
  * @param  beamWidth           Beam width to use for the beam search while decoding.
  * @param  lengthPenalty       Length penalty method.
  * @param  outputLayer         Output layer to use that is applied at the outputs of the provided RNN cell before
  *                             returning them.
  * @param  reorderTensorArrays If `true`, `TensorArray`s' elements within the cell state will be reordered according to
  *                             the beam search path. If the `TensorArray` can be reordered, the stacked form will be
  *                             returned. Otherwise, the `TensorArray` will be returned as is. Set this flag to `false`
  *                             if the cell state contains any `TensorArray`s that are not amenable to reordering.
  * @param  name                Name prefix used for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BeamSearchDecoder[T, State, StateShape](
    override val cell: RNNCell[Output[T], Shape, State, StateShape],
    val initialCellState: State,
    val embeddingFn: Output[Int] => Output[T],
    val beginTokens: Output[Int],
    val endToken: Output[Int],
    val beamWidth: Int,
    val lengthPenalty: LengthPenalty = NoPenalty,
    val outputLayer: Output[T] => Output[T] = (o: Output[T]) => o,
    val reorderTensorArrays: Boolean = true,
    override val name: String = "BeamSearchRNNDecoder"
)(implicit
    evT: WhileLoopVariable.Aux[Output[T], Shape],
    evInt: WhileLoopVariable.Aux[Output[Int], Shape],
    evFloat: WhileLoopVariable.Aux[Output[Float], Shape],
    evBoolean: WhileLoopVariable.Aux[Output[Boolean], Shape],
    evState: WhileLoopVariable.Aux[State, StateShape]
) extends Decoder[
    Output[T], Shape, State, StateShape,
    BeamSearchDecoder.DecoderOutput, (Shape, Shape, Shape),
    BeamSearchDecoder.DecoderState[State, StateShape], (StateShape, Shape, Shape, Shape),
    BeamSearchDecoder.FinalOutput, BeamSearchDecoder.DecoderState[State, StateShape]](cell, name) {
  // TODO: Handle the tiling inside the decoder itself.
  // evS.map(initialCellState, BeamSearchDecoder.tileBatch(_, beamWidth))
  private val tiledInitialCellState: State = initialCellState

  if (evState.outputs(tiledInitialCellState).exists(_.rank == -1))
    throw InvalidArgumentException("All tensors in the state need to have known rank for the beam search decoder.")

  if (beginTokens.rank != 1)
    throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
  if (endToken.rank != 0)
    throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

  /** Scalar tensor representing the batch size of the input values. */
  override val batchSize: Output[Int] = {
    Op.nameScope(name) {
      Basic.size(beginTokens).castTo[Int]
    }
  }

  private val processedInitialCellState: State = {
    Op.nameScope(name) {
      evState.mapWithShape(
        tiledInitialCellState,
        cell.stateShape,
        BeamSearchDecoder.maybeSplitBatchBeams(_, _, batchSize, beamWidth))
    }
  }

  private val processedBeginTokens: Output[Int] = {
    Op.nameScope(name) {
      Basic.tile(Basic.expandDims(beginTokens, 1), Basic.stack[Int](Seq(1, beamWidth)))
    }
  }

  private val beginInput: Output[T] = {
    Op.nameScope(name) {
      embeddingFn(processedBeginTokens)
    }
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
  override val tracksOwnFinished: Boolean = {
    true
  }

  override def zeroOutput(): BeamSearchDecoder.DecoderOutput = {
    // We assume the data type of the cell is the same as the initial cell state's first component tensor data type.
    val zScores = evFloat.zero(batchSize, Shape(beamWidth), "ZeroScores")
    val zPredictedIDs = evInt.zero(batchSize, Shape(beamWidth), "ZeroPredictedIDs")
    val zParentIDs = evInt.zero(batchSize, Shape(beamWidth), "ZeroParentIDs")
    BeamSearchDecoder.DecoderOutput(zScores, zPredictedIDs, zParentIDs)
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize(): (Output[Boolean], Output[T], BeamSearchDecoder.DecoderState[State, StateShape]) = {
    Op.nameScope(s"$name/Initialize") {
      val finished = Basic.oneHot(
        indices = Basic.zeros[Int, Int](batchSize.expandDims(0)),
        depth = beamWidth,
        dataType = BOOLEAN,
        onValue = false,
        offValue = true)
      val initialState = BeamSearchDecoder.DecoderState[State, StateShape](
        modelState = processedInitialCellState,
        logProbabilities = Basic.oneHot(
          indices = Basic.zeros[Int, Int](batchSize.expandDims(0)),
          depth = beamWidth,
          dataType = FLOAT32,
          onValue = Basic.zeros[Float](Shape()),
          offValue = Basic.constant(Float.MinValue)),
        finished = finished,
        sequenceLengths = Basic.zeros[Int, Int](Basic.stack[Int](Seq(batchSize, beamWidth))))
      (finished, beginInput, initialState)
    }
  }

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) the decoder output, (ii) the next state, (iii) the next inputs, and (iv) a scalar
    *         tensor specifying whether sampling has finished.
    */
  override def next(
      time: Output[Int],
      input: Output[T],
      state: BeamSearchDecoder.DecoderState[State, StateShape]
  ): (BeamSearchDecoder.DecoderOutput, BeamSearchDecoder.DecoderState[State, StateShape], Output[T], Output[Boolean]) = {
    implicit val evTF: TF[T] = TF.fromDataType(input.dataType)
    Op.nameScope(s"$name/Next") {
      val mergedInput = evT.mapWithShape(
        input, input.shape(2 ::),
        BeamSearchDecoder.mergeBatchBeams(_, _, batchSize, beamWidth))
      val mergedCellState = evState.mapWithShape(
        state.modelState, cell.stateShape,
        BeamSearchDecoder.maybeMergeBatchBeams(_, _, batchSize, beamWidth))
      val mergedNextTuple = cell(Tuple(mergedInput, mergedCellState))
      val nextTupleOutput = outputLayer(evT.mapWithShape(
        mergedNextTuple.output, mergedNextTuple.output.shape(1 ::),
        BeamSearchDecoder.splitBatchBeams(_, _, batchSize, beamWidth)))
      val nextTupleState = evState.mapWithShape(
        mergedNextTuple.state, cell.stateShape,
        BeamSearchDecoder.maybeSplitBatchBeams(_, _, batchSize, beamWidth))

      // Perform the beam search step.
      val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)

      // Calculate the current lengths of the predictions.
      val predictionLengths = state.sequenceLengths
      val previouslyFinished = state.finished

      // Calculate the total log probabilities for the new hypotheses (final shape = [batchSize, beamWidth, vocabSize]).
      val stepLogProbabilities = BeamSearchDecoder.maskLogProbabilities(
        NN.logSoftmax(nextTupleOutput.castTo[Float]), endToken, previouslyFinished)
      val totalLogProbabilities = state.logProbabilities.expandDims(2) + stepLogProbabilities

      // Calculate the continuation lengths by adding to all continuing search states.
      val vocabSize = {
        if (nextTupleOutput.shape(-1) != -1)
          Basic.constant(nextTupleOutput.shape(-1))
        else
          Basic.shape(nextTupleOutput).castTo[Int].slice(-1)
      }

      var lengthsToAdd = Basic.oneHot(
        indices = Basic.fill[Int, Int](Basic.stack[Int](Seq(batchSize, beamWidth)))(endToken),
        depth = vocabSize,
        dataType = Int,
        onValue = 0,
        offValue = 1)
      val addMask = Math.logicalNot(previouslyFinished).castTo[Int]
      val newPredictionLengths = Math.add(
        lengthsToAdd * addMask.expandDims(2),
        predictionLengths.expandDims(2))

      // Calculate the scores for each search state.
      val scores = lengthPenalty(totalLogProbabilities, newPredictionLengths).castTo[Float]

      // During the first time step we only consider the initial search state.
      val scoresFlat = Basic.reshape(scores, Basic.stack[Int](Seq(batchSize, -1)))

      // Pick the next search states according to the specified successors function.
      val nextBeamSize = Basic.constant(beamWidth, name = "BeamWidth")
      val (nextBeamScores, wordIndices) = NN.topK(scoresFlat, nextBeamSize)
      nextBeamScores.setShape(Shape(staticBatchSize, beamWidth))
      wordIndices.setShape(Shape(staticBatchSize, beamWidth))

      // Pick out the log probabilities, search state indices, and states according to the chosen predictions.
      val nextBeamLogProbabilities = evFloat.map(totalLogProbabilities, BeamSearchDecoder.gather(
        wordIndices, _, batchSize, vocabSize * beamWidth, Seq(-1), name = "NextBeamLogProbabilities"))
      val nextPredictedIDs = Math.mod(wordIndices, vocabSize, name = "NextBeamPredictedIDs").castTo[Int]
      val nextParentIDs = Math.divide(wordIndices, vocabSize, name = "NextBeamParentIDs").castTo[Int]

      // Append the new IDs to the current predictions.
      val gatheredFinished = evBoolean.map(previouslyFinished, BeamSearchDecoder.gather(
        nextParentIDs, _, batchSize, beamWidth, Seq(-1), name = "NextBeamFinishedGather"))
      val nextFinished = Math.logicalOr(
        gatheredFinished, Math.equal(nextPredictedIDs, endToken),
        name = "NextBeamFinished")

      // Calculate the length of the next predictions:
      //   1. Finished search states remain unchanged.
      //   2. Search states that just finished (i.e., `endToken` predicted) have their length increased by 1.
      //   3. Search states that have not yet finished have their length increased by 1.
      var nextPredictionLengths = evInt.map(state.sequenceLengths, BeamSearchDecoder.gather(
        nextParentIDs, _, batchSize, beamWidth, Seq(-1), name = "NextBeamLengthsGather"))
      nextPredictionLengths = nextPredictionLengths + Math.logicalNot(gatheredFinished).castTo[Int]

      // Pick out the cell state according to the next search state parent IDs. We use a different gather shape here
      // because the cell state tensors (i.e., the tensors that would be gathered from) all have rank greater than two
      // and we need to preserve those dimensions.
      val gatheredNextTupleState = evState.map(nextTupleState, BeamSearchDecoder.maybeGather(
        nextParentIDs, _, batchSize, beamWidth, Seq(batchSize * beamWidth, -1),
        name = "NextBeamStateGather"))

      val nextState = BeamSearchDecoder.DecoderState[State, StateShape](
        gatheredNextTupleState, nextBeamLogProbabilities, nextPredictionLengths, nextFinished)
      val output = BeamSearchDecoder.DecoderOutput(nextBeamScores, nextPredictedIDs, nextParentIDs)

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
      output: BeamSearchDecoder.DecoderOutput,
      state: BeamSearchDecoder.DecoderState[State, StateShape],
      sequenceLengths: Output[Int]
  ): (BeamSearchDecoder.FinalOutput, BeamSearchDecoder.DecoderState[State, StateShape], Output[Int]) = {
    // Get the maximum sequence length across all search states for each batch
    val maxSequenceLengths = state.sequenceLengths.max(Tensor(1)).castTo[Int]
    val predictedIDs = BeamSearchDecoder.gatherTree(
      output.predictedIDs, output.parentIDs, maxSequenceLengths, endToken)
    val finalOutput = BeamSearchDecoder.FinalOutput(predictedIDs, output)
    var finalState = state
    if (reorderTensorArrays)
      finalState = state.copy[State, StateShape](modelState = evState.map(
        state.modelState, BeamSearchDecoder.maybeSortTensorArrayBeams(
          _, state.sequenceLengths, output.parentIDs, batchSize, beamWidth)))
    (finalOutput, finalState, finalState.sequenceLengths)
  }
}

object BeamSearchDecoder {
  protected[BeamSearchDecoder] val logger = Logger(LoggerFactory.getLogger("Ops / Beam Search Decoder"))

  def apply[T, State, StateShape](
      cell: RNNCell[Output[T], Shape, State, StateShape],
      initialCellState: State,
      embeddingFn: Output[Int] => Output[T],
      beginTokens: Output[Int],
      endToken: Output[Int],
      beamWidth: Int,
      lengthPenalty: LengthPenalty = NoPenalty,
      outputLayer: Output[T] => Output[T] = (o: Output[T]) => o,
      reorderTensorArrays: Boolean = true,
      name: String = "BeamSearchRNNDecoder"
  )(implicit
      evT: WhileLoopVariable.Aux[Output[T], Shape],
      evInt: WhileLoopVariable.Aux[Output[Int], Shape],
      evFloat: WhileLoopVariable.Aux[Output[Float], Shape],
      evBoolean: WhileLoopVariable.Aux[Output[Boolean], Shape],
      evState: WhileLoopVariable.Aux[State, StateShape]
  ): BeamSearchDecoder[T, State, StateShape] = {
    new BeamSearchDecoder[T, State, StateShape](
      cell, initialCellState, embeddingFn, beginTokens, endToken,
      beamWidth, lengthPenalty, outputLayer, reorderTensorArrays, name)
  }

  case class DecoderOutput(
      scores: Output[Float],
      predictedIDs: Output[Int],
      parentIDs: Output[Int])

  object DecoderOutput {
    implicit def outputWhileLoopVariable(implicit
        evFloatOutput: WhileLoopVariable.Aux[Output[Float], Shape],
        evIntOutput: WhileLoopVariable.Aux[Output[Int], Shape]
    ): WhileLoopVariable.Aux[DecoderOutput, (Shape, Shape, Shape)] = {
      new WhileLoopVariable[DecoderOutput] {
        override type ShapeType = (Shape, Shape, Shape)

        override def zero(
            batchSize: Output[Int],
            shape: (Shape, Shape, Shape),
            name: String = "Zero"
        ): DecoderOutput = {
          Op.nameScope(name) {
            DecoderOutput(
              scores = evFloatOutput.zero(batchSize, shape._1, "Scores"),
              predictedIDs = evIntOutput.zero(batchSize, shape._2, "PredictedIDs"),
              parentIDs = evIntOutput.zero(batchSize, shape._3, "ParentIDs"))
          }
        }

        override def size(value: DecoderOutput): Int = {
          evFloatOutput.size(value.scores) +
              evIntOutput.size(value.predictedIDs) +
              evIntOutput.size(value.parentIDs)
        }

        override def outputs(value: DecoderOutput): Seq[Output[Any]] = {
          evFloatOutput.outputs(value.scores) ++
              evIntOutput.outputs(value.predictedIDs) ++
              evIntOutput.outputs(value.parentIDs)
        }

        override def shapes(shape: (Shape, Shape, Shape)): Seq[Shape] = {
          evFloatOutput.shapes(shape._1) ++
              evIntOutput.shapes(shape._2) ++
              evIntOutput.shapes(shape._3)
        }

        override def segmentOutputs(
            value: DecoderOutput,
            values: Seq[Output[Any]]
        ): (DecoderOutput, Seq[Output[Any]]) = {
          (DecoderOutput(
            scores = values(0).asInstanceOf[Output[Float]],
            predictedIDs = values(1).asInstanceOf[Output[Int]],
            parentIDs = values(2).asInstanceOf[Output[Int]]),
              values.drop(3))
        }

        override def segmentShapes(
            value: DecoderOutput,
            shapes: Seq[Shape]
        ): ((Shape, Shape, Shape), Seq[Shape]) = {
          ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
        }

        override def map(
            value: DecoderOutput,
            mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
        ): DecoderOutput = {
          DecoderOutput(
            scores = evFloatOutput.map(value.scores, mapFn),
            predictedIDs = evIntOutput.map(value.predictedIDs, mapFn),
            parentIDs = evIntOutput.map(value.parentIDs, mapFn))
        }

        override def mapWithShape(
            value: DecoderOutput,
            shape: (Shape, Shape, Shape),
            mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
        ): DecoderOutput = {
          DecoderOutput(
            scores = evFloatOutput.mapWithShape(value.scores, shape._1, mapFn),
            predictedIDs = evIntOutput.mapWithShape(value.predictedIDs, shape._2, mapFn),
            parentIDs = evIntOutput.mapWithShape(value.parentIDs, shape._2, mapFn))
        }
      }
    }
  }

  case class DecoderState[State, StateShape](
      modelState: State,
      logProbabilities: Output[Float],
      sequenceLengths: Output[Int],
      finished: Output[Boolean]
  )(implicit evS: WhileLoopVariable.Aux[State, StateShape])

  object DecoderState {
    implicit def stateWhileLoopVariable[S, SS](implicit
        evState: WhileLoopVariable.Aux[S, SS],
        evFloatOutput: WhileLoopVariable.Aux[Output[Float], Shape],
        evIntOutput: WhileLoopVariable.Aux[Output[Int], Shape],
        evBooleanOutput: WhileLoopVariable.Aux[Output[Boolean], Shape]
    ): WhileLoopVariable.Aux[DecoderState[S, SS], (SS, Shape, Shape, Shape)] = {
      new WhileLoopVariable[DecoderState[S, SS]] {
        override type ShapeType = (SS, Shape, Shape, Shape)

        override def zero(
            batchSize: Output[Int],
            shape: (SS, Shape, Shape, Shape),
            name: String = "Zero"
        ): DecoderState[S, SS] = {
          Op.nameScope(name) {
            DecoderState(
              modelState = evState.zero(batchSize, shape._1, "RNNState"),
              logProbabilities = evFloatOutput.zero(batchSize, shape._2, "LogProbabilities"),
              sequenceLengths = evIntOutput.zero(batchSize, shape._3, "SequenceLengths"),
              finished = evBooleanOutput.zero(batchSize, shape._4, "Finished"))
          }
        }

        override def size(value: DecoderState[S, SS]): Int = {
          evState.size(value.modelState) +
              evFloatOutput.size(value.logProbabilities) +
              evIntOutput.size(value.sequenceLengths) +
              evBooleanOutput.size(value.finished)
        }

        override def outputs(value: DecoderState[S, SS]): Seq[Output[Any]] = {
          evState.outputs(value.modelState) ++
              evFloatOutput.outputs(value.logProbabilities) ++
              evIntOutput.outputs(value.sequenceLengths) ++
              evBooleanOutput.outputs(value.finished)
        }

        override def shapes(shape: (SS, Shape, Shape, Shape)): Seq[Shape] = {
          evState.shapes(shape._1) ++
              evFloatOutput.shapes(shape._2) ++
              evIntOutput.shapes(shape._3) ++
              evBooleanOutput.shapes(shape._4)
        }

        override def segmentOutputs(
            value: DecoderState[S, SS],
            values: Seq[Output[Any]]
        ): (DecoderState[S, SS], Seq[Output[Any]]) = {
          val (modelState, valuesTail) = evState.segmentOutputs(value.modelState, values)
          (DecoderState(
            modelState = modelState,
            logProbabilities = valuesTail(0).asInstanceOf[Output[Float]],
            sequenceLengths = valuesTail(1).asInstanceOf[Output[Int]],
            finished = valuesTail(2).asInstanceOf[Output[Boolean]]),
              valuesTail.drop(3))
        }

        override def segmentShapes(
            value: DecoderState[S, SS],
            shapes: Seq[Shape]
        ): ((SS, Shape, Shape, Shape), Seq[Shape]) = {
          val (modelStateShape, shapesTail) = evState.segmentShapes(value.modelState, shapes)
          ((modelStateShape, shapesTail(0), shapesTail(1), shapesTail(2)), shapesTail.drop(3))
        }

        override def map(
            value: DecoderState[S, SS],
            mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
        ): DecoderState[S, SS] = {
          DecoderState(
            modelState = evState.map(value.modelState, mapFn),
            logProbabilities = evFloatOutput.map(value.logProbabilities, mapFn),
            sequenceLengths = evIntOutput.map(value.sequenceLengths, mapFn),
            finished = evBooleanOutput.map(value.finished, mapFn))
        }

        override def mapWithShape(
            value: DecoderState[S, SS],
            shape: (SS, Shape, Shape, Shape),
            mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
        ): DecoderState[S, SS] = {
          DecoderState(
            modelState = evState.mapWithShape(value.modelState, shape._1, mapFn),
            logProbabilities = evFloatOutput.mapWithShape(value.logProbabilities, shape._2, mapFn),
            sequenceLengths = evIntOutput.mapWithShape(value.sequenceLengths, shape._3, mapFn),
            finished = evBooleanOutput.mapWithShape(value.finished, shape._4, mapFn))
        }
      }
    }
  }

  /** Final outputs returned by the beam search after all decoding is finished.
    *
    * @param  predictedIDs Tensor of shape `[batchSize, T, beamWidth]` (or `[T, batchSize, beamWidth]`,
    *                      if `outputTimeMajor == true`) containing the final prediction IDs. The search states are
    *                      ordered from best to worst.
    * @param  output       State of the beam search at the end of decoding.
    */
  case class FinalOutput(
      predictedIDs: Output[Int],
      output: BeamSearchDecoder.DecoderOutput)

  object FinalOutput {
    implicit def finalOutputWhileLoopVariable(implicit
        evIntOutput: WhileLoopVariable.Aux[Output[Int], Shape],
        evDecoderOutput: WhileLoopVariable.Aux[DecoderOutput, (Shape, Shape, Shape)]
    ): WhileLoopVariable.Aux[FinalOutput, (Shape, (Shape, Shape, Shape))] = {
      new WhileLoopVariable[FinalOutput] {
        override type ShapeType = (Shape, (Shape, Shape, Shape))

        override def zero(
            batchSize: Output[Int],
            shape: (Shape, (Shape, Shape, Shape)),
            name: String = "Zero"
        ): FinalOutput = {
          Op.nameScope(name) {
            FinalOutput(
              evIntOutput.zero(batchSize, shape._1, "PredictedIDs"),
              evDecoderOutput.zero(batchSize, shape._2, "BeamSearchOutput"))
          }
        }

        override def size(value: FinalOutput): Int = {
          evIntOutput.size(value.predictedIDs) +
              evDecoderOutput.size(value.output)
        }

        override def outputs(value: FinalOutput): Seq[Output[Any]] = {
          evIntOutput.outputs(value.predictedIDs) ++
              evDecoderOutput.outputs(value.output)
        }

        override def shapes(shape: (Shape, (Shape, Shape, Shape))): Seq[Shape] = {
          evIntOutput.shapes(shape._1) ++
              evDecoderOutput.shapes(shape._2)
        }

        override def segmentOutputs(
            value: FinalOutput,
            values: Seq[Output[Any]]
        ): (FinalOutput, Seq[Output[Any]]) = {
          val (output, valuesTail) = evDecoderOutput.segmentOutputs(value.output, values.tail)
          (FinalOutput(values(0).asInstanceOf[Output[Int]], output), valuesTail)
        }

        override def segmentShapes(
            value: FinalOutput,
            shapes: Seq[Shape]
        ): ((Shape, (Shape, Shape, Shape)), Seq[Shape]) = {
          val (outputShape, shapesTail) = evDecoderOutput.segmentShapes(value.output, shapes.tail)
          ((shapes(0), outputShape), shapesTail)
        }

        override def map(
            value: FinalOutput,
            mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
        ): FinalOutput = {
          FinalOutput(
            evIntOutput.map(value.predictedIDs, mapFn),
            evDecoderOutput.map(value.output, mapFn))
        }

        override def mapWithShape(
            value: FinalOutput,
            shape: (Shape, (Shape, Shape, Shape)),
            mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
        ): FinalOutput = {
          FinalOutput(
            evIntOutput.mapWithShape(value.predictedIDs, shape._1, mapFn),
            evDecoderOutput.mapWithShape(value.output, shape._2, mapFn))
        }
      }
    }
  }

  /** Masks log probabilities. The result is that finished search states allocate all probability mass to `endToken` and
    * unfinished search states remain unchanged.
    *
    * @param  logProbabilities Log probability for each hypothesis, which is a tensor with shape
    *                          `[batchSize, beamWidth, vocabSize]`.
    * @param  endToken         Scalar tensor containing the end-of-sequence token ID.
    * @param  finished         Tensor of shape `[batchSize, beamWidth]` that specifies which elements in the beam have
    *                          finished decoding.
    * @return Tensor of shape `[batchSize, beamWidth, vocabSize]`, where unfinished search states stay unchanged and
    *         finished search states are replaced with a tensor with all probability mass allocated to `endToken`.
    */
  private[BeamSearchDecoder] def maskLogProbabilities(
      logProbabilities: Output[Float],
      endToken: Output[Int],
      finished: Output[Boolean]
  ): Output[Float] = {
    val vocabSize = Basic.shape(logProbabilities).castTo[Int].slice(2)
    // Finished examples are replaced with a vector that has all its probability mass on `endToken`
    val finishedRow = Basic.oneHot(
      indices = endToken,
      depth = vocabSize,
      dataType = FLOAT32,
      onValue = Basic.zeros[Float](Shape()),
      offValue = Basic.constant(Float.MinValue))
    val finishedLogProbabilities = Basic.tile(
      input = finishedRow.reshape(Shape(1, 1, -1)),
      multiples = Basic.concatenate[Int](Seq(Basic.shape(finished).castTo[Int], Tensor(1)), 0))
    val finishedMask = Basic.tile(
      input = finished.expandDims(2),
      multiples = Basic.stack[Int](Seq(1, 1, vocabSize)))
    Math.select(finishedMask, finishedLogProbabilities, logProbabilities)
  }

  /** The `gatherTree` op calculates the full search states from the per-step IDs and parent beam IDs.
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
      stepIDs: Output[Int],
      parentIDs: Output[Int],
      maxSequenceLengths: Output[Int],
      endToken: Output[Int],
      name: String = "GatherTree"
  ): Output[Int] = {
    Op.Builder[(Output[Int], Output[Int], Output[Int], Output[Int]), Output[Int]](
      opType = "GatherTree",
      name = name,
      input = (stepIDs, parentIDs, maxSequenceLengths, endToken)
    ).build().output
  }

  @throws[InvalidArgumentException]
  private[decoders] def tileBatch[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      multiplier: Int
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(value.dataType)
    value match {
      case output: Output[T] =>
        if (output.rank == -1) {
          throw InvalidArgumentException("The provided tensor must have statically known rank.")
        } else if (output.rank == 0) {
          val tiling = Tensor(multiplier)
          val tiled = Basic.tile(output.expandDims(0), tiling)
          tiled.setShape(Shape(multiplier))
          tiled.asInstanceOf[OL[T]]
        } else {
          val outputShape = Basic.shape(output).castTo[Int]
          val tiling = ArrayBuffer.fill(output.rank + 1)(1)
          tiling(1) = multiplier
          val tiledStaticBatchSize = if (output.shape(0) != -1) output.shape(0) * multiplier else -1
          val tiled = Basic.tile(output.expandDims(1), tiling).reshape(
            Basic.concatenate(Seq((outputShape(0) * multiplier).expandDims(0), outputShape(1 ::))))
          if (output.rank > 1)
            tiled.setShape(Shape(tiledStaticBatchSize) ++ output.shape(1 ::))
          else
            tiled.setShape(Shape(tiledStaticBatchSize))
          tiled.asInstanceOf[OL[T]]
        }
      case _: TensorArray[T] =>
        value
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  @throws[InvalidArgumentException]
  def tileForBeamSearch[S](
      value: S,
      beamWidth: Int
  )(implicit evS: WhileLoopVariable[S]): S = {
    evS.map(value, tileBatch(_, beamWidth))
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
  def maybeSplitBatchBeams[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      shape: Shape,
      batchSize: Output[Int],
      beamWidth: Int
  ): OL[T] = {
    value match {
      case output: Output[T] =>
        if (output.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($output) to have known rank, but it was unknown.")
        else if (output.rank == 0)
          value
        else
          splitBatchBeams(value, shape, batchSize, beamWidth)
      case _: TensorArray[T] =>
        value
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
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
  private[BeamSearchDecoder] def splitBatchBeams[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      shape: Shape,
      batchSize: Output[Int],
      beamWidth: Int
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(value.dataType)
    (value, shape) match {
      case (output: Output[T], s: Shape) =>
        val valueShape = Basic.shape(output).castTo[Int]
        val reshapedValue = Basic.reshape(output, Basic.concatenate(Seq(
          batchSize(NewAxis), Tensor(beamWidth).toOutput,
          valueShape(1 ::).castTo[Int]), axis = 0))
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        val expectedReshapedShape = Shape(staticBatchSize, beamWidth) ++ s
        if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape)) {
          throw InvalidShapeException(
            "Unexpected behavior when reshaping between beam width and batch size. " +
                s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
                s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
                "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
        }
        reshapedValue.setShape(expectedReshapedShape)
        reshapedValue.asInstanceOf[OL[T]]
      case (_: TensorArray[T], _) =>
        value
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Maybe converts the provided tensor structure from a batch of search states into a batch by beams, by merging them
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
  private[BeamSearchDecoder] def maybeMergeBatchBeams[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      shape: Shape,
      batchSize: Output[Int],
      beamWidth: Int
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(value.dataType)
    value match {
      case output: Output[T] =>
        if (output.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($output) to have known rank, but it was unknown.")
        else if (output.rank == 0)
          value
        else
          mergeBatchBeams(value, shape, batchSize, beamWidth)
      case _: TensorArray[T] =>
        value
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Converts the provided tensor structure from a batch of search states into a batch by beams, by merging them
    * accordingly>
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
  private[BeamSearchDecoder] def mergeBatchBeams[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      shape: Shape,
      batchSize: Output[Int],
      beamWidth: Int
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(value.dataType)
    (value, shape) match {
      case (output: Output[T], s: Shape) =>
        val valueShape = Basic.shape(output).castTo[Int]
        val reshapedValue = Basic.reshape(output, Basic.concatenate(Seq(
          batchSize(NewAxis) * Tensor(beamWidth).toOutput,
          valueShape(2 ::).castTo[Int]), axis = 0))
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        val batchSizeBeamWidth = if (staticBatchSize != -1) staticBatchSize * beamWidth else -1
        val expectedReshapedShape = Shape(batchSizeBeamWidth) ++ s
        if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape)) {
          throw InvalidShapeException(
            "Unexpected behavior when reshaping between beam width and batch size. " +
                s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
                s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
                "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
        }
        reshapedValue.setShape(expectedReshapedShape)
        reshapedValue.asInstanceOf[OL[T]]
      case (_: TensorArray[T], _) =>
        value
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
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
  private[BeamSearchDecoder] def maybeGather[T, OL[A] <: OutputLikeOrTensorArray[A]](
      gatherIndices: Output[Int],
      gatherFrom: OL[T],
      batchSize: Output[Int],
      rangeSize: Output[Int],
      gatherShape: Seq[Output[Int]],
      name: String = "GatherTensorHelper"
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(gatherFrom.dataType)
    gatherFrom match {
      case gatherFromOutput: Output[T] =>
        if (gatherFromOutput.rank == -1)
          throw InvalidArgumentException(s"Expected tensor ($gatherFromOutput) to have known rank, but it was unknown.")
        else if (gatherFromOutput.rank < gatherShape.size)
          gatherFrom
        else
          gather(gatherIndices, gatherFrom, batchSize, rangeSize, gatherShape, name)
      case _: TensorArray[T] =>
        gatherFrom
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
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
  private[BeamSearchDecoder] def gather[T, OL[A] <: OutputLikeOrTensorArray[A]](
      gatherIndices: Output[Int],
      gatherFrom: OL[T],
      batchSize: Output[Int],
      rangeSize: Output[Int],
      gatherShape: Seq[Output[Int]],
      name: String = "GatherTensorHelper"
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(gatherFrom.dataType)
    gatherFrom match {
      case gatherFromOutput: Output[T] =>
        Op.nameScope(name) {
          val range = (Math.range(0, batchSize) * rangeSize).expandDims(1)
          val reshapedGatherIndices = (gatherIndices + range).reshape(Shape(-1))
          var output = Basic.gather(gatherFromOutput.reshape(Basic.stack(gatherShape)), reshapedGatherIndices, axis = 0)
          val finalShape = Basic.shape(gatherFromOutput).castTo[Int].slice(0 :: (1 + gatherShape.size))
          val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
          val finalStaticShape = Shape(staticBatchSize) ++ gatherFromOutput.shape(1 :: (1 + gatherShape.size))
          output = Basic.reshape(output, finalShape, name = "Output")
          output.setShape(finalStaticShape)
          output.asInstanceOf[OL[T]]
        }
      case _: TensorArray[T] =>
        gatherFrom
      case _ =>
        throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Checks are known statically and can be reshaped to `[batchSize, beamSize, -1]` and logs a warning
    * if they cannot. */
  private[BeamSearchDecoder] def checkStaticBatchBeam(
      shape: Shape,
      batchSize: Int,
      beamWidth: Int
  ): Boolean = {
    if (batchSize != -1 && shape(0) != -1 &&
        (shape(0) != batchSize * beamWidth ||
            (shape.rank > 1 && shape(1) != -1 &&
                (shape(0) != batchSize || shape(1) != beamWidth)))) {
      val reshapedShape = Shape(batchSize, beamWidth, -1)
      logger.warn(
        s"Tensor array reordering expects elements to be reshapable to '$reshapedShape' which is incompatible with " +
            s"the current shape '$shape'. Consider setting `reorderTensorArrays` to `false` to disable tensor array " +
            "reordering during the beam search.")
      false
    } else {
      true
    }
  }

  /** Returns an assertion op checking that the elements of the stacked tensor array (i.e., `tensor`) can be reshaped
    * to `[batchSize, beamSize, -1]`. At this point, the tensor array elements have a known rank of at least `1`. */
  private[BeamSearchDecoder] def checkBatchBeam[T](
      tensor: Output[T],
      batchSize: Output[Int],
      beamWidth: Output[Int]
  ): UntypedOp = {
    implicit val evTF: TF[T] = TF.fromDataType(tensor.dataType)
    Checks.assert(
      condition = {
        val shape = Basic.shape(tensor).castTo[Int]
        if (tensor.rank == 2)
          Math.equal(shape(1), batchSize * beamWidth)
        else
          Math.logicalOr(
            Math.equal(shape(1), batchSize * beamWidth),
            Math.logicalAnd(
              Math.equal(shape(1), batchSize),
              Math.equal(shape(2), beamWidth)))
      },
      data = Seq("Tensor array reordering expects elements to be reshapable to '[batchSize, beamSize, -1]' which is " +
          s"incompatible with the dynamic shape of '${tensor.name}' elements. Consider setting `reorderTensorArrays` " +
          "to `false` to disable tensor array reordering during the beam search."))
  }

  /** Maybe sorts the search states within a tensor array.
    *
    * @param  value           Symbol to be sorted. This will only be sorted if it is a tensor array of size `maxTime`
    *                         that contains tensors with shape `[batchSize, beamWidth, s]` or
    *                         `[batchSize * beamWidth, s]`, where `s` is the depth shape.
    * @param  sequenceLengths Tensor containing the sequence lengths, with shape `[batchSize, beamWidth]`.
    * @param  parentIDs       Tensor containing the parent indices, with shape `[maxTime, batchSize, beamWidth]`.
    * @param  batchSize       Batch size.
    * @param  beamWidth       Beam width.
    * @return A tensor array where the search states are sorted in each tensor, or `value` itself, if it is not a tensor
    *         array or does not meet the shape requirements.
    */
  private[BeamSearchDecoder] def maybeSortTensorArrayBeams[T, OL[A] <: OutputLikeOrTensorArray[A]](
      value: OL[T],
      sequenceLengths: Output[Int],
      parentIDs: Output[Int],
      batchSize: Output[Int],
      beamWidth: Int
  ): OL[T] = {
    implicit val evTF: TF[T] = TF.fromDataType(value.dataType)
    value match {
      case ta: TensorArray[T] if (!ta.inferShape || ta.elementShape.isEmpty) ||
          ta.elementShape.get(0) == -1 ||
          ta.elementShape.get(1) < 1 =>
        val shape = ta.elementShape match {
          case Some(s) if ta.inferShape => Shape(s(0))
          case _ => Shape(-1)
        }
        logger.warn(
          s"The tensor array '${ta.handle.name}' in the cell state is not amenable to sorting based on the beam " +
              s"search result. For a tensor array to be sorted, its elements shape must be defined and have at least " +
              s"a rank of 1. However, the elements shape in the provided tensor array is: $shape.")
        ta.asInstanceOf[OL[T]]
      case ta: TensorArray[T] if !checkStaticBatchBeam(
        shape = Shape(ta.elementShape.get(0)),
        batchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1),
        beamWidth = beamWidth) =>
        ta.asInstanceOf[OL[T]]
      case ta: TensorArray[T] =>
        val stackedTensorArray = ta.stack()
        Op.createWith(controlDependencies = Set(checkBatchBeam(stackedTensorArray, batchSize, beamWidth))) {
          val maxTime = Basic.shape(parentIDs).castTo[Int].slice(0)
          val batchSize = Basic.shape(parentIDs).castTo[Int].slice(1)
          val beamWidth = Basic.shape(parentIDs).castTo[Int].slice(2)

          // Generate search state indices that will be reordered by the `gatherTree` op.
          val searchStateIndices = Basic.tile(
            Math.range(0, beamWidth).slice(NewAxis, NewAxis),
            Basic.stack[Int](Seq(maxTime, batchSize, 1)))
          val mask = Basic.sequenceMask(sequenceLengths, maxTime).castTo[Int].transpose(Seq(2, 0, 1))

          // Use `beamWidth + 1` to mark the end of the beam.
          val maskedSearchStateIndices = (searchStateIndices * mask) + (1 - mask) * (beamWidth + 1)
          val maxSequenceLengths = sequenceLengths.max(Tensor(1)).castTo[Int]
          var sortedSearchStateIndices = gatherTree(
            maskedSearchStateIndices, parentIDs, maxSequenceLengths, beamWidth + 1)

          // For out of range steps, we simply copy the same beam.
          sortedSearchStateIndices = Math.select(mask.castTo[Boolean], sortedSearchStateIndices, searchStateIndices)

          // Generate indices for `gatherND`.
          val timeIndices = Basic.tile(
            Math.range(0, maxTime).slice(NewAxis, NewAxis),
            Basic.stack[Int](Seq(1, batchSize, beamWidth)))
          val batchIndices = Basic.tile(
            Math.range(0, batchSize).slice(NewAxis, NewAxis),
            Basic.stack[Int](Seq(1, maxTime, beamWidth))).transpose(Seq(1, 0, 2))
          val indices = Basic.stack(Seq(timeIndices, batchIndices, sortedSearchStateIndices), -1)

          // Gather from a tensor with collapsed additional dimensions.
          val finalShape = Basic.shape(stackedTensorArray).castTo[Int]
          val gatherFrom = stackedTensorArray.reshape(Basic.stack[Int](Seq(maxTime, batchSize, beamWidth, -1)))
          Basic.gatherND(gatherFrom, indices).reshape(finalShape).asInstanceOf[OL[T]]
        }
      case _ =>
        value
    }
  }
}
