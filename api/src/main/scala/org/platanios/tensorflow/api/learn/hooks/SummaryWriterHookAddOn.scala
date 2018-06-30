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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.io.events.{SummaryFileWriter, SummaryFileWriterCache}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

import org.tensorflow.framework.{HistogramProto, Summary}

import java.nio.file.Path

/** Add-on trait for hooks that provides convenience methods for using a summary writer.
  *
  * @author Emmanouil Antonios Platanios
  */
trait SummaryWriterHookAddOn extends Hook {
  val summaryDir: Path

  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  override private[learn] def internalBegin(): Unit = {
    summaryWriter = Option(summaryDir).map(SummaryFileWriterCache.get(_))
    super.internalBegin()
  }

  override private[learn] def internalEnd(session: Session): Unit = {
    summaryWriter.foreach(_.flush())
    super.internalEnd(session)
  }

  protected def writeSummary(step: Long, tag: String, value: Tensor[_ <: DataType]): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setTensor(value.toTensorProto))
          .build(), step))
  }

  protected def writeSummary(step: Long, tag: String, value: Float): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setSimpleValue(value))
          .build(), step))
  }

  protected def writeSummary(step: Long, tag: String, value: HistogramProto): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setHisto(value))
          .build(), step))
  }

  protected def writeSummary(step: Long, tag: String, value: Summary.Image): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setImage(value))
          .build(), step))
  }

  protected def writeSummary(step: Long, tag: String, value: Summary.Audio): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setAudio(value))
          .build(), step))
  }
}
