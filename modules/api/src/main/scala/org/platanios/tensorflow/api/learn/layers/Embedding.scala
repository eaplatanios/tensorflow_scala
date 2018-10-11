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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

object Embedding {
  private[layers] trait API {
    type Embedding[T] = layers.Embedding[T]

    val Embedding: layers.Embedding.type = layers.Embedding
  }

  object API extends API
}

case class Embedding[T: TF : IsNotQuantized](
    override val name: String,
    vocabularySize: Int,
    embeddingSize: Int,
    partitionStrategy: ops.Embedding.PartitionStrategy = ops.Embedding.ModStrategy,
    transformFn: Output[T] => Output[T] = null,
    maxNorm: Tensor[T] = null
) extends Layer[Output[Int], Output[T]](name) {
  override val layerType: String = "Embedding"

  override def forwardWithoutContext(
      input: Output[Int]
  )(implicit mode: Mode): Output[T] = {
    val embeddingMap = getParameter[T]("EmbeddingMap", Shape(vocabularySize, embeddingSize))
    ops.Embedding.embeddingLookup(
      embeddingMap, input, partitionStrategy, transformFn,
      if (maxNorm == null) null else ops.Basic.constant(maxNorm),
      name)
  }
}
