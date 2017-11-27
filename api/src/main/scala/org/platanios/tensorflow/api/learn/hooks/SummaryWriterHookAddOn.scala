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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.io.events.{SummaryFileWriter, SummaryFileWriterCache}

import org.tensorflow.framework.Summary

import java.nio.file.Path

/** Add-on trait for hooks that provides convenience methods for using a summary writer.
  *
  * @author Emmanouil Antonios Platanios
  */
trait SummaryWriterHookAddOn {
  private[this] var summaryWriter: Option[SummaryFileWriter] = None

  protected def summaryWriterBegin(summaryDir: Path): Unit = {
    summaryWriter = Option(summaryDir).map(SummaryFileWriterCache.get(_))
  }

  protected def summaryWriterWrite(step: Long, tag: String, value: Float): Unit = {
    summaryWriter.foreach(_.writeSummary(
      Summary.newBuilder()
          .addValue(Summary.Value.newBuilder()
              .setTag(tag)
              .setSimpleValue(value))
          .build(), step))
  }

  protected def summaryWriterEnd(): Unit = {
    summaryWriter.foreach(_.flush())
  }
}
