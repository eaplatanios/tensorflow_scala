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

import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.learn.layers

/**
  * @author Emmanouil Antonios Platanios
  */
trait Loss[T] extends NetworkLayer[T, tf.Output]

object Loss {
  trait API {
    type Loss[T] = layers.Loss[T]
    type L2Loss = layers.L2Loss
    type SoftmaxCrossEntropy = layers.SoftmaxCrossEntropy
    type SparseSoftmaxCrossEntropy = layers.SparseSoftmaxCrossEntropy
    type SigmoidCrossEntropy = layers.SigmoidCrossEntropy

    def l2Loss(name: String = "L2Loss"): L2Loss = L2Loss(name = name)

    def softmaxCrossEntropy(name: String = "SoftmaxCrossEntropy"): SoftmaxCrossEntropy = {
      SoftmaxCrossEntropy(name = name)
    }

    def sparseSoftmaxCrossEntropy(name: String = "SparseSoftmaxCrossEntropy"): SparseSoftmaxCrossEntropy = {
      SparseSoftmaxCrossEntropy(name = name)
    }

    def sigmoidCrossEntropy(name: String = "SigmoidCrossEntropy"): SigmoidCrossEntropy = {
      SigmoidCrossEntropy(name = name)
    }
  }

  object API extends API
}

case class L2Loss private[layers](override val name: String = "L2Loss") extends Loss[(tf.Output, tf.Output)] {
  override val layerType: String                              = s"L2Loss"
  override val forward  : ((tf.Output, tf.Output)) => tf.Output = input => {
    tf.l2Loss(input._1 - input._2, name = name)
  }
}

case class SoftmaxCrossEntropy private[layers](override val name: String = "SoftmaxCrossEntropy")
    extends Loss[(tf.Output, tf.Output)] {
  override val layerType: String                              = s"SoftmaxCrossEntropy"
  override val forward  : ((tf.Output, tf.Output)) => tf.Output = input => {
    tf.softmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SparseSoftmaxCrossEntropy private[layers](override val name: String = "SparseSoftmaxCrossEntropy")
    extends Loss[(tf.Output, tf.Output)] {
  override val layerType: String                                = s"SparseSoftmaxCrossEntropy"
  override val forward  : ((tf.Output, tf.Output)) => tf.Output = input => {
    tf.sparseSoftmaxCrossEntropy(input._1, input._2, name = name)
  }
}

case class SigmoidCrossEntropy private[layers](override val name: String = "SigmoidCrossEntropy")
    extends Loss[(tf.Output, tf.Output)] {
  override val layerType: String                              = s"SigmoidCrossEntropy"
  override val forward  : ((tf.Output, tf.Output)) => tf.Output = input => {
    tf.sigmoidCrossEntropy(input._1, input._2, name = name)
  }
}
