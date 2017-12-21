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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.ops.Basic.BasicOps
import org.platanios.tensorflow.api.ops.Clip.ClipOps
import org.platanios.tensorflow.api.ops.Embedding.{OutputParameters, VariableParameters}
import org.platanios.tensorflow.api.ops.Math.MathOps
import org.platanios.tensorflow.api.ops.NN.NNOps
import org.platanios.tensorflow.api.ops.Sparse.SparseOps
import org.platanios.tensorflow.api.ops.Statistics.StatisticsOps
import org.platanios.tensorflow.api.ops.Text.TextOps
import org.platanios.tensorflow.api.ops.variables.{PartitionedVariable, Variable}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow.ControlFlowOps
import org.platanios.tensorflow.api.tensors.TensorConvertible

/** Groups together all implicits related to constructing symbolic ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Ops {
  implicit def opToControlFlowOps(op: Op): ControlFlowOps = ControlFlowOps(op)

  implicit def tensorConvertibleToOutput[T](value: T)(implicit ev: TensorConvertible[T]): Output = {
    ev.toTensor(value).toOutput
  }

  implicit def outputLikeToOutput[T <: OutputLike](outputLike: T): Output = outputLike.toOutput

  implicit def outputToOp(output: Output): Op = output.op
  implicit def outputToInitialValueFunction(output: Output): () => Output = () => output

  implicit def outputToBasicOps(value: Output): BasicOps = BasicOps(value)
  implicit def outputConvertibleToBasicOps[T](value: T)(implicit f: (T) => Output): BasicOps = BasicOps(f(value))

  implicit def outputToClipOps(value: Output): ClipOps = ClipOps(value)
  implicit def outputConvertibleToClipOps[T](value: T)(implicit f: (T) => Output): ClipOps = ClipOps(f(value))

  implicit def outputToMathOps(value: Output): MathOps = MathOps(value)
  implicit def outputConvertibleToMathOps[T](value: T)(implicit f: (T) => Output): MathOps = MathOps(f(value))

  implicit def outputToNNOps(value: Output): NNOps = NNOps(value)
  implicit def outputConvertibleToNNOps[T](value: T)(implicit f: (T) => Output): NNOps = NNOps(f(value))

  implicit def sparseOutputToNNOps(value: SparseOutput): SparseOps = SparseOps(value)

  implicit def outputToStatisticsOps(value: Output): StatisticsOps = StatisticsOps(value)
  implicit def outputConvertibleToStatisticsOps[T](value: T)(implicit f: (T) => Output): StatisticsOps = {
    StatisticsOps(f(value))
  }

  implicit def outputToTextOps(value: Output): TextOps = TextOps(value)
  implicit def outputConvertibleToTextOps[T](value: T)(implicit f: (T) => Output): TextOps = TextOps(f(value))

  implicit def singlePartitionEmbeddingMap(parameters: EmbeddingParameters): EmbeddingMap = {
    EmbeddingMap(Seq(parameters))
  }

  implicit def multiplePartitionsEmbeddingMap(parameters: Seq[EmbeddingParameters]): EmbeddingMap = {
    EmbeddingMap(parameters)
  }

  implicit def partitionedVariableEmbeddingMap(parameters: PartitionedVariable): EmbeddingMap = {
    EmbeddingMap(parameters.map(VariableParameters).toSeq)
  }

  implicit def outputToEmbeddingMap(parameters: Output): EmbeddingMap = OutputParameters(parameters)
  implicit def variableToEmbeddingMap(parameters: Variable): EmbeddingMap = VariableParameters(parameters)
}
