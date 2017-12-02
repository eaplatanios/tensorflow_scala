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

package org.platanios.tensorflow.api.ops.seq2seq

import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/**
  * @author Emmanouil Antonios Platanios
  */
package object decoders {
  private[seq2seq] trait API {
    type Decoder[O, OS, S, SS, DO, DOS, DS, DSS, DFO, DFS] = org.platanios.tensorflow.api.ops.seq2seq.decoders.Decoder[O, OS, S, SS, DO, DOS, DS, DSS, DFO, DFS]
    type BasicDecoder[O, OS, S, SS] = org.platanios.tensorflow.api.ops.seq2seq.decoders.BasicDecoder[O, OS, S, SS]
    type BeamSearchDecoder[S, SS] = org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder[S, SS]

    val BasicDecoder     : org.platanios.tensorflow.api.ops.seq2seq.decoders.BasicDecoder.type      = decoders.BasicDecoder
    val BeamSearchDecoder: org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder.type = decoders.BeamSearchDecoder

    def beamSearchRNNDecoderTileBatch[T: WhileLoopVariable](value: T, multiplier: Int): T = {
      implicitly[WhileLoopVariable[T]].map(value, BeamSearchDecoder.tileBatch(_, multiplier))
    }
  }
}
