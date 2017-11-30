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

package org.platanios.tensorflow.api.ops.seq2seq.decoders

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Output}
import org.platanios.tensorflow.api.types.FLOAT32

/** Length penalty function to be used while decoding. */
trait LengthPenalty {
  def apply(scores: Output, sequenceLengths: Output): Output
}

/** No length penalty. */
case object NoPenalty extends LengthPenalty {
  override def apply(scores: Output, sequenceLengths: Output): Output = scores
}

/** Google length penalty function described in
  * [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144.)
  * The penalty is equal to `((5 + sequenceLengths) / 6) ^ weight`, where all operations are performed element-wise.
  *
  * @param  weight Length penalty weight (disabled if set to `0.0f`).
  */
case class GooglePenalty(weight: Float) extends LengthPenalty {
  override def apply(scores: Output, sequenceLengths: Output): Output = {
    if (weight == 0.0f) {
      scores
    } else {
      val penaltyFactor = Basic.constant(weight, name = "PenaltyFactor")
      scores / Math.divide((5.0f + sequenceLengths.cast(FLOAT32)) ^ penaltyFactor, 6.0f ^ penaltyFactor)
    }
  }
}
