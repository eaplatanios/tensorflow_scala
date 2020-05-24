/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.ops.Embedding.{OutputParameters, VariableParameters}
import org.platanios.tensorflow.api.ops.{EmbeddingMap, EmbeddingParameters, Output}
import org.platanios.tensorflow.api.ops.variables.Variable

trait EmbeddingImplicits {
  implicit def singlePartitionEmbeddingMap[T: TF](
      parameters: EmbeddingParameters[T]
  ): EmbeddingMap[T] = {
    EmbeddingMap(Seq(parameters))
  }

  implicit def multiplePartitionsEmbeddingMap[T: TF](
      parameters: Seq[EmbeddingParameters[T]]
  ): EmbeddingMap[T] = {
    EmbeddingMap(parameters)
  }

  implicit def outputToEmbeddingMap[T: TF : IsNotQuantized](
      parameters: Output[T]
  ): EmbeddingMap[T] = {
    OutputParameters(parameters)
  }

  implicit def variableToEmbeddingMap[T: TF : IsNotQuantized](
      parameters: Variable[T]
  ): EmbeddingMap[T] = {
    VariableParameters(parameters)
  }
}
