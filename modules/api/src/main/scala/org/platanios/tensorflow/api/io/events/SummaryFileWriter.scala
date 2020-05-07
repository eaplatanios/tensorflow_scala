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

package org.platanios.tensorflow.api.io.events

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.proto._

import com.google.protobuf.ByteString

import java.nio.charset.StandardCharsets
import java.nio.file.Path

import scala.collection.JavaConverters._
import scala.collection.mutable

/** Writes `Summary` protocol buffers to event files for use with TensorBoard.
  *
  * The [[SummaryFileWriter]] class provides a mechanism to create an event file in a given directory and add summaries
  * and events to it. The class updates the file contents asynchronously. This allows a training program to call methods
  * to add data to the file directly from the training loop, without slowing down training.
  *
  * On construction the summary writer creates a new event file in `workingDir`. This event file will contain `Event`
  * protocol buffers constructed when you call one of the following functions: `writeGraph()`, `writeSummary()`,
  * `writeSessionLog()`, or `writeRunMetadata()`.
  *
  * If you pass a `Graph` to the constructor it is write to the event file, which is equivalent to calling
  * `writeGraph()` later on.
  *
  * TensorBoard will pick up the graph from the file and display it graphically so you can interactively explore it. You
  * will usually pass the graph from the session in which you launched it:
  * {{{
  *   // Create a graph.
  *   val graph = Graph()
  *   ...
  *   // Launch the graph in a session.
  *   val session = Session(graph)
  *   // Create a summary file writer and add the graph to the event file.
  *   val writer = SummaryWriter(workingDir, session.graph)
  * }}}
  *
  * @param  workingDir     Directory in which to write the event file.
  * @param  graph          Graph to write to the event file when constructing the summary file writer.
  * @param  queueCapacity  Maximum number of events pending to be written to disk before a call to `write()` blocks.
  * @param  flushFrequency Specifies how often to flush the written events to disk (in seconds).
  * @param  filenameSuffix Filename suffix to use for the event file.
  *
  * @author Emmanouil Antonios Platanios
  */
class SummaryFileWriter private[io](
    override val workingDir: Path,
    private[this] val graph: Graph = null,
    override val queueCapacity: Int = 10,
    override val flushFrequency: Int = 10,
    override val filenameSuffix: String = ""
) extends EventFileWriter(workingDir, queueCapacity, flushFrequency, filenameSuffix) {
  if (graph != null)
    writeGraph(graph)

  /** Contains tags of summary values that have been encountered already. The motivation here is that the
    * [[SummaryFileWriter]] only keeps the metadata property (which is a `SummaryMetadata` proto) of the first summary
    * value encountered for each tag. The [[SummaryFileWriter]] strips away the summary metadata for all subsequent
    * summary values with tags seen previously. This saves space. */
  private[this] val seenSummaryTags: mutable.Set[String] = mutable.HashSet.empty[String]

  /** Contains used tags for `session.run()` outputs. */
  private[this] val usedSessionRunTags: mutable.Set[String] = mutable.HashSet.empty[String]

  /** Writes a [[Graph]] to the event file along with its meta-information.
    *
    * This method wraps the provided graph in an `Event` protocol buffer and writes it to the event file. The provided
    * graph will be displayed by TensorBoard. Most users pass a graph in the constructor instead.
    *
    * @param  graph Graph to write.
    * @param  step  Global step number to record with the graph.
    */
  def writeGraph(graph: Graph, step: Long = 0L): Unit = {
    val metaGraphDef = graph.toMetaGraphDef()
    write(eventBuilder(step)
              .setGraphDef(metaGraphDef.getGraphDef.toByteString)
              .setMetaGraphDef(metaGraphDef.toByteString)
              .build())
  }

  /** Writes a [[GraphDef]] to the event file.
    *
    * This method wraps the provided [[GraphDef]] in an `Event` protocol buffer and writes it to the event file.
    *
    * @param  graphDef [[GraphDef]] to write.
    * @param  step     Global step number to record with the [[GraphDef]].
    */
  def writeGraphDef(graphDef: GraphDef, step: Long = 0L): Unit = {
    write(eventBuilder(step).setGraphDef(graphDef.toByteString).build())
  }

  /** Writes a [[MetaGraphDef]] to the event file.
    *
    * This method wraps the provided [[MetaGraphDef]] in an `Event` protocol buffer and writes it to the event file.
    *
    * @param  metaGraphDef [[MetaGraphDef]] to write.
    * @param  step         Global step number to record with the [[MetaGraphDef]].
    */
  def writeMetaGraphDef(metaGraphDef: MetaGraphDef, step: Long = 0L): Unit = {
    write(eventBuilder(step).setMetaGraphDef(metaGraphDef.toByteString).build())
  }

  /** Writes a `Summary` protocol buffer to the event file given a string representation of that protocol buffer.
    *
    * This method wraps the provided summary in an `Event` protocol buffer and writes it to the event file.
    *
    * You can pass the result of evaluating any summary op (e.g., using `Session.run()`) to this function.
    *
    * @param  summary String representation of the summary to write.
    * @param  step    Global step number to record with the summary.
    */
  def writeSummaryString(summary: String, step: Long): Unit = {
    writeSummary(Summary.parseFrom(ByteString.copyFrom(summary.getBytes(StandardCharsets.ISO_8859_1))), step)
  }

  /** Writes a `Summary` protocol buffer to the event file.
    *
    * This method wraps the provided summary in an `Event` protocol buffer and writes it to the event file.
    *
    * @param  summary Summary to write.
    * @param  step    Global step number to record with the summary.
    */
  def writeSummary(summary: Summary, step: Long): Unit = {
    val summaryBuilder = Summary.newBuilder(summary)
    summaryBuilder.clearValue()
    // We strip the summary metadata for values with tags that we have seen before in order to save space. We just store
    // the metadata on the first value with a specific tag.
    summary.getValueList.asScala.foreach(value => {
      if (value.hasMetadata && seenSummaryTags.contains(value.getTag)) {
        // This tag has been encountered before and we thus strip the metadata.
        summaryBuilder.addValue(Summary.Value.newBuilder(value).clearMetadata().build())
      } else {
        summaryBuilder.addValue(value)
        // We encounter a value with a tag we have not encountered previously. We record its tag in order to remember to
        // strip the metadata from future values with the same tag.
        seenSummaryTags.add(value.getTag)
      }
    })
    write(eventBuilder(step).setSummary(summaryBuilder).build())
  }

  /** Writes a [[SessionLog]] to the event file.
    *
    * This method wraps the provided session log in an `Event` protocol buffer and writes it to the event file.
    *
    * @param  sessionLog Session log to write.
    * @param  step       Global step number to record with the session log.
    */
  def writeSessionLog(sessionLog: SessionLog, step: Long): Unit = {
    write(eventBuilder(step).setSessionLog(sessionLog).build())
  }

  /** Writes run metadata information for a single `Session.run()` call to the event file.
    *
    * @param  runMetadata Run metadata to write.
    * @param  tag         Tag name for these metadata.
    * @param  step        Global step number to record with the summary.
    * @throws IllegalArgumentException If the provided tag has already been used for this event type.
    */
  @throws[IllegalArgumentException]
  def writeRunMetadata(runMetadata: RunMetadata, tag: String, step: Long): Unit = {
    if (usedSessionRunTags.contains(tag))
      throw new IllegalArgumentException(s"The provided tag ($tag) has already been used for this event type.")
    usedSessionRunTags.add(tag)
    val taggedRunMetadataBuilder = TaggedRunMetadata.newBuilder()
    taggedRunMetadataBuilder.setTag(tag)
    // Store the `RunMetadata` object as bytes in order to have postponed (lazy) deserialization when used later.
    taggedRunMetadataBuilder.setRunMetadata(runMetadata.toByteString)
    write(eventBuilder(step).setTaggedRunMetadata(taggedRunMetadataBuilder).build())
  }

  private[this] def eventBuilder(step: Long = 0L): Event.Builder = {
    Event.newBuilder().setWallTime(System.currentTimeMillis().toDouble / 1000.0).setStep(step)
  }
}

object SummaryFileWriter {
  /** Creates a new [[SummaryFileWriter]].
    *
    * @param  workingDir     Directory in which to write the event file.
    * @param  graph          Graph to write to the event file when constructing the summary file writer.
    * @param  queueCapacity  Maximum number of events pending to be written to disk before a call to `write()` blocks.
    * @param  flushFrequency Specifies how often to flush the written events to disk (in seconds).
    * @param  filenameSuffix Filename suffix to use for the event file.
    * @return Constructed summary file writer.
    */
  def apply(
      workingDir: Path,
      graph: Graph = null,
      queueCapacity: Int = 10,
      flushFrequency: Int = 10,
      filenameSuffix: String = ""
  ): SummaryFileWriter = {
    new SummaryFileWriter(workingDir, graph, queueCapacity, flushFrequency, filenameSuffix)
  }
}
