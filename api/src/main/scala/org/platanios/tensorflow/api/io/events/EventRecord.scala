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

package org.platanios.tensorflow.api.io.events

import com.google.protobuf.ByteString
import org.tensorflow.framework.TensorProto

/**
  * @author Emmanouil Antonios Platanios
  */
trait EventRecord[T] {
  val wallTime: Double
  val step: Long
  val value: T
}

case class ScalarEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: Float
) extends EventRecord[Float]

case class ImageEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: ImageValue
) extends EventRecord[ImageValue]

case class ImageValue(encodedImage: ByteString, width: Int, height: Int, colorSpace: Int)

case class AudioEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: AudioValue
) extends EventRecord[AudioValue]

case class AudioValue(
    encodedAudio: ByteString, contentType: String, sampleRate: Float, numChannels: Long, lengthFrames: Long)

case class HistogramEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: HistogramValue
) extends EventRecord[HistogramValue]

case class HistogramValue(
    min: Double, max: Double, num: Double, sum: Double, sumSquares: Double, bucketLimits: Seq[Double],
    buckets: Seq[Double])

case class CompressedHistogramEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: Seq[HistogramValue]
) extends EventRecord[Seq[HistogramValue]]

case class TensorEventRecord(
    override val wallTime: Double,
    override val step: Long,
    override val value: TensorProto
) extends EventRecord[TensorProto]
