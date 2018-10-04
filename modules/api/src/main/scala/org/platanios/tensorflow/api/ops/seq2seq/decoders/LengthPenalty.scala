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

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.IsNotQuantized

/** Length penalty function to be used while decoding. */
trait LengthPenalty {
  def apply[T: IsNotQuantized](
      scores: Output[T],
      sequenceLengths: Output[Int]
  ): Output[T]
}

/** No length penalty. */
case object NoPenalty extends LengthPenalty {
  override def apply[T: IsNotQuantized](
      scores: Output[T],
      sequenceLengths: Output[Int]
  ): Output[T] = {
    scores
  }
}

/** Exponential length penalty function. The penalty is equal to `sequenceLengths ^ alpha`, where all operations a re
  * performed element-wise.
  *
  * @param  alpha Length penalty weight (disabled if set to `0.0f`).
  */
case class ExponentialPenalty(alpha: Float) extends LengthPenalty {
  override def apply[T: IsNotQuantized](
      scores: Output[T],
      sequenceLengths: Output[Int]
  ): Output[T] = {
    if (alpha == 0.0f) {
      scores
    } else {
      Op.nameScope("LengthPenalty") {
        val penaltyFactor = Basic.constant(alpha, name = "PenaltyFactor").castTo(scores.dataType)
        scores / (sequenceLengths.castTo(scores.dataType) ^ penaltyFactor)
      }
    }
  }
}

/** Google length penalty function described in
  * [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144.)
  * The penalty is equal to `((5 + sequenceLengths) / 6) ^ alpha`, where all operations are performed element-wise.
  *
  * @param  alpha Length penalty weight (disabled if set to `0.0f`).
  */
case class GooglePenalty(alpha: Float) extends LengthPenalty {
  override def apply[T: IsNotQuantized](
      scores: Output[T],
      sequenceLengths: Output[Int]
  ): Output[T] = {
    if (alpha == 0.0f) {
      scores
    } else {
      Op.nameScope("LengthPenalty") {
        val five = Basic.constant(5.0f, name = "Five").castTo(scores.dataType)
        val six = Basic.constant(6.0f, name = "Six").castTo(scores.dataType)
        val lengths = sequenceLengths.castTo(scores.dataType)
        val penaltyFactor = Basic.constant(alpha, name = "PenaltyFactor").castTo(scores.dataType)
        scores / Math.divide((five + lengths) ^ penaltyFactor, six ^ penaltyFactor)
      }
    }
  }
}
