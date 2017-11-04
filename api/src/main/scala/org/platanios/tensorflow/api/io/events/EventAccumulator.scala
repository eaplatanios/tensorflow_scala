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

import org.platanios.tensorflow.api.io.DirectoryLoader
import org.platanios.tensorflow.api.utilities.Reservoir

import com.google.protobuf.ByteString
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework._
import org.tensorflow.util.Event
import org.tensorflow.util.SessionLog.SessionStatus

import java.nio.file.{Files, Path}

import scala.collection.JavaConverters._
import scala.collection.mutable

/** Accumulates event values collected from the provided path.
  *
  * The [[EventAccumulator]] is intended to provide a convenient interface for loading event data written during a
  * TensorFlow run (or otherwise). TensorFlow writes out event ProtoBuf objects, which have a timestamp and step number
  * associated with them, and often also contain a [[Summary]]. Summaries can store different kinds of data like a
  * scalar value, an image, audio, or a histogram. Each summary also has a tag associated with it, which we use to
  * organize logically related data. The [[EventAccumulator]] supports retrieving the event and summary data by their
  * tags.
  *
  * Calling `tags` returns a map from event types to the associated tags for those types, that were found in the loaded
  * event files. Then, various functional endpoints (e.g., `scalars(tag)`) allow for the retrieval of all data
  * associated with each tag.
  *
  * The `reload()` method synchronously loads all of the data written so far.
  *
  * @param  path                    Path to a directory containing TensorFlow events files, or a single TensorFlow
  *                                 events file. The accumulator will load events from this path.
  * @param  sizeGuidance            Information on how much data the event accumulator should store in memory. The
  *                                 default size guidance tries not to store too much so as to avoid consuming all of
  *                                 the client's memory. The `sizeGuidance` should be a map from event types to integers
  *                                 representing the number of items to keep in memory, per tag for items of that event
  *                                 type. If the size is `0`, then all events are stored. Images, audio, and histograms
  *                                 tend to be very large and thus storing all of them is not recommended.
  * @param  histogramCompressionBps Information on how the event accumulator should compress histogram data for the
  *                                 [[CompressedHistogramEventType]] event type.
  * @param  purgeOrphanedData       Boolean value indicating whether to discard any events that were "orphaned" by a
  *                                 TensorFlow restart.
  *
  * @author Emmanouil Antonios Platanios
  */
case class EventAccumulator(
    path: Path,
    sizeGuidance: Map[EventType, Int] = EventAccumulator.DEFAULT_SIZE_GUIDANCE,
    histogramCompressionBps: Seq[Int] = EventAccumulator.DEFAULT_HISTOGRAM_COMPRESSION_BPS,
    purgeOrphanedData: Boolean = true) {
  private[this] val eventLoader: () => Iterator[Event] = EventAccumulator.eventLoaderFromPath(path)

  private[this] object EventLoaderLock

  private[this] var _firstEventTimeStamp: Double = -1.0
  private[this] var _fileVersion        : Float  = -1.0f

  private[this] var _mostRecentWallTime: Double = -1L
  private[this] var _mostRecentStep    : Long   = -1L

  private[this] val _actualSizeGuidance = EventAccumulator.DEFAULT_SIZE_GUIDANCE ++ sizeGuidance

  private[this] val _reservoirs: Map[EventType, Reservoir[String, _ <: EventRecord[_]]] = Map(
    ScalarEventType -> Reservoir[String, ScalarEventRecord](_actualSizeGuidance(ScalarEventType)),
    ImageEventType -> Reservoir[String, ImageEventRecord](_actualSizeGuidance(ImageEventType)),
    AudioEventType -> Reservoir[String, AudioEventRecord](_actualSizeGuidance(AudioEventType)),
    HistogramEventType -> Reservoir[String, HistogramEventRecord](_actualSizeGuidance(HistogramEventType)),
    CompressedHistogramEventType -> Reservoir[String, CompressedHistogramEventRecord](
      _actualSizeGuidance(CompressedHistogramEventType), alwaysKeepLast = false)
  )

  private[this] var _graphDef          : ByteString                   = _
  private[this] var _graphFromMetaGraph: Boolean                      = false
  private[this] var _metaGraphDef      : ByteString                   = _
  private[this] var _taggedRunMetadata : Map[String, ByteString]      = Map.empty[String, ByteString]
  private[this] var _summaryMetadata   : Map[String, SummaryMetadata] = Map.empty[String, SummaryMetadata]

  // Keep a mapping from plugin name to a map from tag to plugin data content obtained from the summary metadata for
  // that plugin (this is not the entire summary metadata proto - only the content for that plugin). The summary writer
  // only keeps the content on the first event encountered per tag, and so we must store that first instance of content
  // for each tag.
  private[this] val _pluginTagContent: mutable.Map[String, mutable.Map[String, String]] = mutable.Map.empty

  /** Loads all events added since the last call to `reload()` and returns this event accumulator. If `reload()` was
    * never called before, then it loads all events in the path. */
  def reload(): EventAccumulator = EventLoaderLock synchronized {
    eventLoader().foreach(processEvent)
    this
  }

  /** Returns the timestamp (in seconds) of the first event.
    *
    * If the first event has been loaded (either by this method or by `reload()`, then this method returns immediately.
    * Otherwise, it loads the first event and then returns. Note that this means that calling `reload()` will cause this
    * method to block until `reload()` has finished. */
  def firstEventTimeStamp: Double = {
    if (_firstEventTimeStamp >= 0) {
      _firstEventTimeStamp
    } else {
      EventLoaderLock synchronized {
        try {
          processEvent(eventLoader().next())
          _firstEventTimeStamp
        } catch {
          case t: Throwable => throw new IllegalStateException("No event timestamp could be found.", t)
        }
      }
    }
  }

  /** Returns all scalar events associated with the provided summary tag. */
  def scalars(tag: String): List[ScalarEventRecord] = {
    _reservoirs(ScalarEventType).asInstanceOf[Reservoir[String, ScalarEventRecord]].items(tag)
  }

  /** Returns all image events associated with the provided summary tag. */
  def images(tag: String): List[ImageEventRecord] = {
    _reservoirs(ImageEventType).asInstanceOf[Reservoir[String, ImageEventRecord]].items(tag)
  }

  /** Returns all audio events associated with the provided summary tag. */
  def audio(tag: String): List[AudioEventRecord] = {
    _reservoirs(AudioEventType).asInstanceOf[Reservoir[String, AudioEventRecord]].items(tag)
  }

  /** Returns all histogram events associated with the provided summary tag. */
  def histograms(tag: String): List[HistogramEventRecord] = {
    _reservoirs(HistogramEventType).asInstanceOf[Reservoir[String, HistogramEventRecord]].items(tag)
  }

  /** Returns all compressed histogram events associated with the provided summary tag. */
  def compressedHistograms(tag: String): List[CompressedHistogramEventRecord] = {
    _reservoirs(CompressedHistogramEventType).asInstanceOf[Reservoir[String, CompressedHistogramEventRecord]].items(tag)
  }

  /** Returns all tensor events associated with the provided summary tag. */
  def tensors(tag: String): List[TensorEventRecord] = {
    _reservoirs(TensorEventType).asInstanceOf[Reservoir[String, TensorEventRecord]].items(tag)
  }

  /** Returns the graph definition, if there is one.
    *
    * If the graph is stored directly, the method returns it. If no graph is stored directly, but a meta-graph is stored
    * containing a graph, the method returns that graph. */
  @throws[IllegalStateException]
  def graph: GraphDef = {
    if (_graphDef != null)
      GraphDef.parseFrom(_graphDef)
    else
      throw new IllegalStateException("There is no graph in this event accumulator.")
  }

  /** Returns the meta-graph definition, if there is one. */
  @throws[IllegalStateException]
  def metaGraph: MetaGraphDef = {
    if (_metaGraphDef != null)
      MetaGraphDef.parseFrom(_metaGraphDef)
    else
      throw new IllegalStateException("There is no meta-graph in this event accumulator.")
  }

  /** Returns the run metadata associated with the provided summary tag. */
  @throws[IllegalArgumentException]
  def runMetadata(tag: String): RunMetadata = {
    if (!_taggedRunMetadata.contains(tag))
      throw new IllegalArgumentException("There is no run metadata for the provided tag name.")
    RunMetadata.parseFrom(_taggedRunMetadata(tag))
  }

  /** Returns the summary metadata associated with the provided summary tag. */
  def summaryMetadata(tag: String): SummaryMetadata = {
    _summaryMetadata(tag)
  }

  /** Returns a map from tags to content specific to the specified plugin. */
  def pluginTagToContent(pluginName: String): Option[Map[String, String]] = {
    _pluginTagContent.get(pluginName).map(_.toMap)
  }

  /** Returns a sequence with paths to all the registered assets for the provided plugin name.
    *
    * If a plugins directory does not exist in the managed directory, then this method returns an empty list. This
    * maintains compatibility with old log directories that contain no plugin sub-directories.
    */
  def pluginAssets(pluginName: String): Seq[Path] = {
    EventPluginUtilities.listPluginAssets(path, pluginName)
  }

  /** Retrieves a particular plugin asset from the managed directory and returns it as a string. */
  def retrievePluginAsset(pluginName: String, assetName: String): String = {
    EventPluginUtilities.retrievePluginAsset(path, pluginName, assetName)
  }

  /** Returns a map from event types to all corresponding tags that have been accumulated. */
  def tags: Map[EventType, Seq[String]] = Map(
    ScalarEventType -> _reservoirs(ScalarEventType).keys.toSeq,
    ImageEventType -> _reservoirs(ImageEventType).keys.toSeq,
    AudioEventType -> _reservoirs(AudioEventType).keys.toSeq,
    HistogramEventType -> _reservoirs(HistogramEventType).keys.toSeq,
    CompressedHistogramEventType -> _reservoirs(CompressedHistogramEventType).keys.toSeq,
    TensorEventType -> _reservoirs(TensorEventType).keys.toSeq,
    // We use a heuristic here: if a meta-graph is available, but a graph is not, then we assume that the meta-graph
    // contains the graph.
    // TODO: I don't really get this.
    GraphEventType -> Seq((_graphDef != null).toString),
    MetaGraphEventType -> Seq((_metaGraphDef != null).toString),
    RunMetadataEventType -> _taggedRunMetadata.keys.toSeq
  )

  /** Processes a newly-loaded event. */
  private[this] def processEvent(event: Event): Unit = {
    if (_firstEventTimeStamp < 0)
      _firstEventTimeStamp = event.getWallTime
    if (event.getWhatCase == Event.WhatCase.FILE_VERSION) {
      val newFileVersion = {
        val tokens = event.getFileVersion.split("brain.Event:")
        try {
          tokens.last.toFloat
        } catch {
          // This should never happen according to the definition of the file version field specified in event.proto.
          case _: NumberFormatException =>
            EventAccumulator.logger.warn(
              "Invalid event.proto file_version. Defaulting to use of out-of-order event.step logic " +
                  "for purging expired events.")
            -1f
        }
      }
      if (_fileVersion >= 0 && _fileVersion != newFileVersion) {
        // This should not happen.
        EventAccumulator.logger.warn(
          "Found new file version for event. This will affect purging logic for TensorFlow restarts. " +
              s"Old: ${_fileVersion}. New: $newFileVersion.")
      }
      _fileVersion = newFileVersion
    }
    maybePurgeOrphanedData(event)

    // Process the event.
    event.getWhatCase match {
      case Event.WhatCase.GRAPH_DEF =>
        // GraphDef and MetaGraphDef are handled in a special way: If no GraphDef event is available, but a MetaGraphDef is,
        // and it contains a GraphDef, then we use that GraphDef for our graph. If a GraphDef event is available, then we
        // always prefer it to the GraphDef inside the MetaGraphDef.
        if (_graphDef != null)
          EventAccumulator.logger.warn(
            "Found more than one graph event per run, or there was a meta-graph containing a graph definition, " +
                "as well as one or more graph events. Overwriting the graph with the newest event.")
        _graphDef = event.getGraphDef
        _graphFromMetaGraph = false
      case Event.WhatCase.META_GRAPH_DEF =>
        if (_metaGraphDef != null)
          EventAccumulator.logger.warn(
            "Found more than one meta-graph event per run. Overwriting the meta-graph with the newest event.")
        _metaGraphDef = event.getMetaGraphDef
        if (_graphDef == null || _graphFromMetaGraph) {
          // We may have a GraphDef in the meta-graph. If so, and no GraphDef is directly available, we use this one
          // instead.
          val metaGraphDef = MetaGraphDef.parseFrom(_metaGraphDef)
          if (metaGraphDef.hasGraphDef) {
            if (_graphDef != null)
              EventAccumulator.logger.warn(
                "Found multiple meta-graphs containing graph definitions, but did not find any graph events. " +
                    "Overwriting the graph with the newest meta-graph version.")
            _graphDef = metaGraphDef.getGraphDef.toByteString
            _graphFromMetaGraph = true
          }
        }
      case Event.WhatCase.TAGGED_RUN_METADATA =>
        val tag = event.getTaggedRunMetadata.getTag
        if (_taggedRunMetadata.contains(tag))
          EventAccumulator.logger.warn(
            s"Found more than one run metadata event with tag '$tag'. Overwriting it with the newest event.")
        _taggedRunMetadata += tag -> event.getTaggedRunMetadata.getRunMetadata
      case Event.WhatCase.SUMMARY =>
        event.getSummary.getValueList.asScala.foreach(value => {
          if (value.hasMetadata) {
            val tag = value.getTag
            // We only store the first instance of the metadata. This check is important: the `FileWriter` does strip
            // metadata from all values except the first one per each tag. However, a new `FileWriter` is created every
            // time a training job stops and restarts. Hence, we must also ignore non-initial metadata in this logic.
            if (!_summaryMetadata.contains(tag)) {
              _summaryMetadata += tag -> value.getMetadata
              val pluginData = value.getMetadata.getPluginData
              if (pluginData.getPluginName != null) {
                _pluginTagContent
                    .getOrElseUpdate(pluginData.getPluginName, mutable.Map.empty[String, String])
                    .update(tag, pluginData.getContent.toStringUtf8)
              } else {
                EventAccumulator.logger.warn(s"The summary with tag '$tag' is oddly not associated with any plugin.")
              }
            }
          }
          value.getValueCase match {
            case Summary.Value.ValueCase.SIMPLE_VALUE =>
              val record = ScalarEventRecord(event.getWallTime, event.getStep, value.getSimpleValue)
              _reservoirs(ScalarEventType).asInstanceOf[Reservoir[String, ScalarEventRecord]].add(value.getTag, record)
            case Summary.Value.ValueCase.IMAGE =>
              val image = value.getImage
              val imageValue = ImageValue(
                image.getEncodedImageString, image.getWidth, image.getHeight, image.getColorspace)
              val record = ImageEventRecord(event.getWallTime, event.getStep, imageValue)
              _reservoirs(ImageEventType).asInstanceOf[Reservoir[String, ImageEventRecord]].add(value.getTag, record)
            case Summary.Value.ValueCase.AUDIO =>
              val audio = value.getAudio
              val audioValue = AudioValue(
                audio.getEncodedAudioString, audio.getContentType, audio.getSampleRate, audio.getNumChannels,
                audio.getLengthFrames)
              val record = AudioEventRecord(event.getWallTime, event.getStep, audioValue)
              _reservoirs(AudioEventType).asInstanceOf[Reservoir[String, AudioEventRecord]].add(value.getTag, record)
            case Summary.Value.ValueCase.HISTO =>
              val histogram = value.getHisto
              val histogramValue = HistogramValue(
                histogram.getMin, histogram.getMax, histogram.getNum, histogram.getSum, histogram.getSumSquares,
                histogram.getBucketLimitList.asScala.map(_.toDouble), histogram.getBucketList.asScala.map(_.toDouble))
              val record = HistogramEventRecord(event.getWallTime, event.getStep, histogramValue)
              _reservoirs(HistogramEventType).asInstanceOf[Reservoir[String, HistogramEventRecord]].add(value.getTag, record)
            // TODO: [EVENTS] Compress histogram and add to the compressed histograms reservoir.
            case Summary.Value.ValueCase.TENSOR =>
              val tag = {
                if (value.getTag == null) {
                  // This tensor summary was created using the old method that used plugin assets.
                  // We must still continue to support it.
                  value.getNodeName
                } else {
                  value.getTag
                }
              }
              val record = TensorEventRecord(event.getWallTime, event.getStep, value.getTensor)
              _reservoirs(TensorEventType).asInstanceOf[Reservoir[String, TensorEventRecord]].add(tag, record)
            case _ => EventAccumulator.logger.warn(s"Unrecognized value type (${value.getValueCase}) is ignored.")
          }
        })
      case _ => ()
    }
  }

  //region Purging Methods

  /** Purges orphaned data due to a TensorFlow crash, if that is deemed necessary.
    *
    * When TensorFlow crashes at step `T+O` and restarts at step `T`, any events written after step `T` are now
    * "orphaned" and will be at best misleading if they are included. This method attempts to determine if there is
    * orphaned data, and purge it if it is found.
    *
    * @param  event Event to use as reference for the purge.
    */
  private[this] def maybePurgeOrphanedData(event: Event): Unit = {
    if (purgeOrphanedData) {
      // Check if the event happened after a crash, and purge expired tags.
      if (_fileVersion >= 2) {
        // If the file version is recent enough, we can use the session log events to check for restarts.
        checkForRestartAndMaybePurge(event)
      } else {
        // If there is no file version or if the file version is too old, we default to the old logic of checking for
        // out of order steps.
        checkForOutOfOrderStepAndMaybePurge(event)
      }
    }
  }

  /** Checks and discards expired events using `SessionLog.START`.
    *
    * Checks for a `SessionLog.START` event and purges all previously seen events with larger steps, because they are
    * out of date. It is possible that this logic will cause the first few event messages to be discarded because the
    * TensorFlow supervisor threading does not guarantee that the `START` message is deterministically written first.
    *
    * This method is preferred over `checkForOutOfOrderStepAndMaybePurge` which can inadvertently discard events due to
    * the TensorFlow supervisor threading behavior.
    *
    * @param  event Event to use as reference for the purge.
    */
  private[this] def checkForRestartAndMaybePurge(event: Event): Unit = {
    if (event.getSessionLog != null && event.getSessionLog.getStatus == SessionStatus.START)
      purge(event, byTags = false)
  }

  /** Checks for an out-of-order event step and discards any expired events.
    *
    * Checks if the provided event is out of order relative to the global most recent step. If it is, then the method
    * purges ant outdated summaries for tags that the event contains.
    *
    * @param  event Event to use as reference for the purge.
    */
  private[this] def checkForOutOfOrderStepAndMaybePurge(event: Event): Unit = {
    if (event.getStep < _mostRecentStep && event.getWhatCase == Event.WhatCase.SUMMARY) {
      purge(event, byTags = true)
    } else {
      _mostRecentWallTime = event.getWallTime
      _mostRecentStep = event.getStep
    }
  }

  /** Purges all events that have occurred after the provided event step.
    *
    * If `byTags` is `true`, then the method purges all events that occurred after the provided event step, but only for
    * the tags that the event has. Non-sequential event steps suggest that a TensorFlow restart occurred, and we discard
    * the out-of-order events in order to obtain a consistent view of the data.
    *
    * Discarding by tags is the safer method, when we are unsure whether a restart has occurred, given that threading in
    * the TensorFlow supervisor can cause events with different tags to arrive with un-synchronized step values.
    *
    * If `byTags` is `false`, then the method purges all events with step greater than the provided event step. This can
    * be used when we are certain that a TensorFlow restart has occurred and these events can be discarded.
    *
    * @param  event  Event to use as reference for the purge.
    * @param  byTags Boolean value indicating whether to purge all out-of-order events or only those that are associated
    *                with the provided reference event.
    */
  private[this] def purge(event: Event, byTags: Boolean): Unit = {
    // Keep data that has a step less than the event step in the reservoirs.
    val notExpired = (e: EventRecord[_]) => e.step < event.getStep
    val expiredPerType = {
      if (byTags) {
        val tags = event.getSummary.getValueList.asScala.map(_.getTag)
        _reservoirs.mapValues(r => tags.map(t => r.filter(notExpired, Some(t))).sum)
      } else {
        _reservoirs.mapValues(_.filter(notExpired))
      }
    }
    if (expiredPerType.values.sum > 0) {
      EventAccumulator.logger.warn(
        "Detected out of order event step likely caused by a TensorFlow restart." +
            s"Purging expired events between the previous step " +
            s"(${_mostRecentStep} - timestamp = ${_mostRecentWallTime}) and the current step " +
            s"(${event.getStep} - timestamp = ${event.getWallTime}). " +
            s"Removing ${expiredPerType(ScalarEventType)} scalars, ${expiredPerType(ImageEventType)} images, " +
            s"${expiredPerType(AudioEventType)} audio, ${expiredPerType(HistogramEventType)} histograms, and " +
            s"${expiredPerType(CompressedHistogramEventType)}} compressed histograms.")
    }
  }

  //endregion Purging Methods
}

object EventAccumulator {
  private[EventAccumulator] val logger: Logger = Logger(LoggerFactory.getLogger("Event Accumulator"))

  /** Default size guidance to use. */
  private[events] val DEFAULT_SIZE_GUIDANCE: Map[EventType, Int] = Map(
    ScalarEventType -> 10000,
    ImageEventType -> 4,
    AudioEventType -> 4,
    HistogramEventType -> 1,
    CompressedHistogramEventType -> 500,
    TensorEventType -> 10,
    GraphEventType -> 1,
    MetaGraphEventType -> 1,
    RunMetadataEventType -> 1
  )

  /** Default histogram compression BPS to use. The Normal CDF for standard deviations:
    * (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf) naturally gives bands around the median of width 1 std dev, 2 std dev,
    * 3 std dev, and then the long tail. */
  private[events] val DEFAULT_HISTOGRAM_COMPRESSION_BPS: Seq[Int] = {
    Seq(0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)
  }

  /** Returns an events file reader for the provided path. */
  private[EventAccumulator] def eventLoaderFromPath(path: Path): () => Iterator[Event] = {
    if (Files.isRegularFile(path) && path.getFileName.toString.contains("tfevents")) {
      () => EventFileReader(path).load()
    } else {
      () => DirectoryLoader(path, EventFileReader(_), p => p.getFileName.toString.contains("tfevents")).load()
    }
  }
}
