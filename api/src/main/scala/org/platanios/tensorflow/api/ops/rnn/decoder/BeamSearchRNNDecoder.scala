///* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.tensorflow.api.ops.rnn.decoder
//
//import org.platanios.tensorflow.api.Implicits._
//import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
//import org.platanios.tensorflow.api.{Shape, ops}
//import org.platanios.tensorflow.api.ops.{Basic, Output, TensorArray}
//import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
//import org.platanios.tensorflow.api.ops.rnn.cell.RNNCell
//import org.platanios.tensorflow.api.tensors.Tensor
//
//import scala.language.postfixOps
//
///** Recurrent Neural Network (RNN) that uses beam search to find the highest scoring sequence.
//  *
//  * @author Emmanouil Antonios Platanios
//  */
//class BeamSearchRNNDecoder[T: RNNCell.Supported](
//    override val cell: RNNCell.Tuple[T] => RNNCell.Tuple[T],
//    override val cellOutputSize: Seq[Int],
//    override val initialCellState: Seq[Output],
//    val beginTokens: Output,
//    val endToken: Output,
//    val beamWidth: Int,
//    val lengthPenaltyWeight: Float = 0.0f,
//    override val name: String = "BeamSearchRNNDecoder"
//)(implicit
//    whileLoopEvT: WhileLoopVariable.Aux[T, _],
//    whileLoopEvOutput: WhileLoopVariable.Aux[BeamSearchRNNDecoder.Output[Output], _],
//    whileLoopEvOutputTensorArray: WhileLoopVariable.Aux[BeamSearchRNNDecoder.Output[TensorArray], _],
//    whileLoopEvStateT: WhileLoopVariable.Aux[BeamSearchRNNDecoder.State, _]
//) extends RNNDecoder[
//    T,
//    BeamSearchRNNDecoder.Output[Output],
//    BeamSearchRNNDecoder.Output[TensorArray],
//    BeamSearchRNNDecoder.State](
//  cell,
//  cellOutputSize,
//  initialCellState,
//  name
//) {
//  if (initialCellState.exists(_.rank == -1))
//    throw InvalidArgumentException("All tensors in the state need to have known rank for the beam search decoder.")
//
//  /** Scalar `INT32` tensor representing the batch size of the input values. */
//  override val batchSize: Output = Basic.size(beginTokens)
//
//  /** Describes whether the decoder keeps track of finished states.
//    *
//    * Most decoders will emit a true/false `finished` value independently at each time step. In this case, the
//    * `dynamicDecode()` function keeps track of which batch entries have already finished, and performs a logical OR to
//    * insert new batches to the finished set.
//    *
//    * Some decoders, however, shuffle batches/beams between time steps and `dynamicDecode()` will mix up the finished
//    * state across these entries because it does not track the reshuffling across time steps. In this case, it is up to
//    * the decoder to declare that it will keep track of its own finished state by setting this property to `true`.
//    */
//  override val tracksOwnFinished: Boolean = true
//
//  private[this] def splitBatchBeams(value: Output, shape: Shape): Output = {
//    val valueShape = Basic.shape(value)
//    val reshapedValue = Basic.reshape(value, Basic.concatenate(Seq(
//      batchSize.expandDims(0),
//      Tensor(batchSize.dataType, beamWidth).toOutput,
//      valueShape(1 ::).cast(batchSize.dataType)), axis = 0))
//    val staticBatchSize = Output.constantValue(batchSize).scalar.asInstanceOf[Int]
//    val expectedReshapedShape = Shape(staticBatchSize, beamWidth) ++ shape
//    if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape))
//      throw InvalidShapeException(
//        "Unexpected behavior when reshaping between beam width and batch size. " +
//            s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
//            s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
//            "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
//    reshapedValue.setShape(expectedReshapedShape)
//    reshapedValue
//  }
//
//  override def createZeroOutputTensorArrays(): (
//      BeamSearchRNNDecoder.Output[Output], BeamSearchRNNDecoder.Output[TensorArray]) = {
//    val zeroOutput = BeamSearchRNNDecoder.Output()
//
//    val zeroOutputs = cellOutputSize.map(s => Basic.fill(initialCellState.head.dataType, Basic.stack(Seq(batchSize, s)))(0))
//    val rnnCellZeroOutput = RNNCell.Supported[T].fromOutputs(zeroOutputs)
//    val rnnCellTensorArrays = zeroOutputs.map(output => {
//      TensorArray.create(0, output.dataType, dynamicSize = true, elementShape = output.shape)
//    })
//    val zeroOutput = BasicRNNDecoder.Output[T](rnnCellZeroOutput, rnnCellZeroOutput)
//    val tensorArrays = BasicRNNDecoder.Output[Seq[TensorArray]](rnnCellTensorArrays, rnnCellTensorArrays)
//    (zeroOutput, tensorArrays)
//  }
//
//}
//
//object BeamSearchRNNDecoder {
//  case class State(
//      rnnState: Seq[ops.Output], logProbabilities: ops.Output, sequenceLengths: ops.Output, finished: ops.Output)
//
//  case class Output[T](
//      scores: T, predictedIDs: T, parentIDs: T
//  )(implicit whileLoopEvT: WhileLoopVariable.Aux[T, _])
//}
