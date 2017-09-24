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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
trait Loss[T] extends NetworkLayer[T, Output]

object Loss {
  trait API {
    type Loss[T] = layers.Loss[T]

    val L2Loss                   : layers.L2Loss.type                    = layers.L2Loss
    val SoftmaxCrossEntropy      : layers.SoftmaxCrossEntropy.type       = layers.SoftmaxCrossEntropy
    val SparseSoftmaxCrossEntropy: layers.SparseSoftmaxCrossEntropy.type = layers.SparseSoftmaxCrossEntropy
    val SigmoidCrossEntropy      : layers.SigmoidCrossEntropy.type       = layers.SigmoidCrossEntropy
  }

  object API extends API
}

case class L2Loss private[layers](override val name: String = "L2Loss") extends Loss[(Output, Output)] {
  override val layerType: String                       = s"L2Loss"
  override val forward  : ((Output, Output)) => Output = input => {
    ops.NN.l2Loss(input._1 - input._2, name = name)
  }
}

case class SoftmaxCrossEntropy private[layers](override val name: String = "SoftmaxCrossEntropy")
    extends Loss[(Output, Output)] {
  override val layerType: String                       = s"SoftmaxCrossEntropy"
  override val forward  : ((Output, Output)) => Output = input => {
    ops.NN.softmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SparseSoftmaxCrossEntropy private[layers](override val name: String = "SparseSoftmaxCrossEntropy")
    extends Loss[(Output, Output)] {
  override val layerType: String                       = s"SparseSoftmaxCrossEntropy"
  override val forward  : ((Output, Output)) => Output = input => {
    ops.NN.sparseSoftmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SigmoidCrossEntropy private[layers](override val name: String = "SigmoidCrossEntropy")
    extends Loss[(Output, Output)] {
  override val layerType: String                       = s"SigmoidCrossEntropy"
  override val forward  : ((Output, Output)) => Output = input => {
    ops.NN.sigmoidCrossEntropy(input._1, input._2, name = name)
  }
}
