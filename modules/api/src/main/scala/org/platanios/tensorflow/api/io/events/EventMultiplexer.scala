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

import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.io.{DirectoryLoader, FileIO}
import org.platanios.tensorflow.proto.{GraphDef, MetaGraphDef, RunMetadata, SummaryMetadata}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path, Paths}

import scala.collection.compat._
import scala.collection.mutable

/** An [[EventMultiplexer]] manages access to multiple [[EventAccumulator]]s.
  *
  * Each event accumulator is associated with a `run`, which is a self-contained execution. The event multiplexer
  * provides methods for extracting information about events from multiple `run`s.
  *
  * Example usage for loading specific runs from files:
  * {{{
  *   val m = EventMultiplexer(Map("run1" -> Paths.get("path/to/run1"), "run2" -> Paths.get("path/to/run2")))
  *   m.reload()
  * }}}
  *
  * Example usage for loading a directory where each subdirectory corresponds to a run:
  * {{{
  *   // For example, assume the following directory structure:
  *   //   /parent/directory/path/
  *   //   /parent/directory/path/run1/
  *   //   /parent/directory/path/run1/events.out.tfevents.1001
  *   //   /parent/directory/path/run1/events.out.tfevents.1002
  *   //   /parent/directory/path/run2/
  *   //   /parent/directory/path/run2/events.out.tfevents.9232
  *   //   /parent/directory/path/run3/
  *   //   /parent/directory/path/run3/events.out.tfevents.9232
  *   val m1 = EventMultiplexer().addRunsFromDirectory("/parent/directory/path")
  *   // is equivalent to:
  *   val m2 = EventMultiplexer(Map("run1" -> Paths.get("/parent/directory/path/run1", ...)))
  * }}}
  *
  * @param  initialRunPaths         Optional map containing the initial run name to path mapping.
  * @param  sizeGuidance            Information on how much data each event accumulator should store in memory. The
  *                                 default size guidance tries not to store too much so as to avoid consuming all of
  *                                 the client's memory. The `sizeGuidance` should be a map from event types to integers
  *                                 representing the number of items to keep in memory, per tag for items of that event
  *                                 type. If the size is `0`, then all events are stored. Images, audio, and histograms
  *                                 tend to be very large and thus storing all of them is not recommended.
  * @param  histogramCompressionBps Information on how each event accumulator should compress histogram data for the
  *                                 [[CompressedHistogramEventType]] event type.
  * @param  purgeOrphanedData       Boolean value indicating whether to discard any events that were "orphaned" by a
  *                                 TensorFlow restart.
  *
  * @author Emmanouil Antonios Platanios
  */
case class EventMultiplexer(
    initialRunPaths: Map[String, Path] = Map.empty[String, Path],
    sizeGuidance: Map[EventType, Int] = EventAccumulator.DEFAULT_SIZE_GUIDANCE,
    histogramCompressionBps: Seq[Int] = EventAccumulator.DEFAULT_HISTOGRAM_COMPRESSION_BPS,
    purgeOrphanedData: Boolean = true) {
  private[this] val accumulators: mutable.Map[String, EventAccumulator] = mutable.Map.empty[String, EventAccumulator]
  private[this] val paths       : mutable.Map[String, Path]             = mutable.Map.empty[String, Path]

  private[this] object AccumulatorsLock

  private[this] var reloadCalled: Boolean = false

  // Initialize the multiplexer for the provided run paths.
  initialRunPaths.foreach(p => addRun(p._2, p._1))

  /** Adds a run to this multiplexer.
    *
    * If a name is not provided, then the run name is set to the provided path.
    *
    * If a run with the same name exists and we are already watching the right path, this method does nothing. If we are
    * watching a different path for that name, we replace its event accumulator.
    *
    * If `reload()` has been called, then this method will also reload the newly created event accumulators.
    *
    * @param  path Path for the run.
    * @param  name Name for the run. Defaults to the provided path.
    * @return This event multiplexer after the run has been added to it.
    */
  def addRun(path: Path, name: String = null): EventMultiplexer = {
    val inferredName = if (name != null) name else path.toString
    var accumulator: EventAccumulator = null
    AccumulatorsLock synchronized {
      if (!accumulators.contains(inferredName) || !paths.get(inferredName).contains(path)) {
        if (paths.contains(inferredName) && paths(inferredName) != path)
          EventMultiplexer.logger.warn(s"Replacing path for '$inferredName': '${paths(inferredName)} -> $path.")
        EventMultiplexer.logger.info(s"Constructing an event accumulator for '$path'.")
        accumulator = EventAccumulator(path, sizeGuidance, histogramCompressionBps, purgeOrphanedData)
        accumulators.update(name, accumulator)
        paths.update(name, path)
      }
    }
    if (accumulator != null && reloadCalled)
      accumulator.reload()
    this
  }

  /** Adds runs from a directory and its subdirectories (traversed recursively) to this multiplexer.
    *
    * If the provided directory path does not exist, then this method does nothing. This ensures that it is safe to call
    * it multiple times, even before the directory is created.
    *
    * If the provided path points to a directory, this method loads all event files in the directory (if any exist) and
    * then recursively performs the same on any subdirectories. This means you can call `addRunsFromDirectory` at the
    * root of a tree of event log directories and the event multiplexer will load all events in that tree.
    *
    * If `reload()` has been called, then this method will also reload the newly created event accumulators.
    *
    * @param  directory Path to a directory to load the runs from.
    * @param  name      Optional name for the runs. If a name is provided and the directory contains multiple
    *                   subdirectories, then the name of each sub-run will be the concatenation of the parent name and
    *                   the subdirectory name. If a name is provided and the directory contains event files, then a run
    *                   is added with that name. If a name is not provided, the directory and subdirectory paths are
    *                   used as names.
    * @return This event multiplexer after the runs have been added to it.
    * @throws InvalidArgumentException If the `directory` path exists but does not point to a directory.
    */
  @throws[InvalidArgumentException]
  def addRunsFromDirectory(directory: Path, name: String = null): EventMultiplexer = {
    EventMultiplexer.logger.info(s"Adding runs from directory '$directory'.")
    if (Files.exists(directory) && !Files.isDirectory(directory))
      throw InvalidArgumentException(s"Path '$directory' exists but is not a directory.")
    FileIO.walk(directory)
        .filter(_._3.exists(_.getFileName.toString.contains("tfevents")))
        .map(_._1)
        .foreach(subDirectory => {
          EventMultiplexer.logger.info(s"Adding events from directory '$subDirectory'.")
          val relativePath = directory.relativize(subDirectory)
          val subName = if (name != null) Paths.get(name).resolve(relativePath) else relativePath
          addRun(subDirectory, subName.toString)
        })
    EventMultiplexer.logger.info(s"Done with adding runs from directory '$directory'.")
    this
  }

  /** Reloads all of the managed event accumulators. */
  def reload(): EventMultiplexer = {
    EventMultiplexer.logger.info("Starting a multiplexer reload.")
    reloadCalled = true
    // Build a list so we are safe even if the list of accumulators is modified while we are reloading.
    val accumulators = AccumulatorsLock.synchronized(this.accumulators.toList)
    var namesToDelete = Set.empty[String]
    accumulators.foreach {
      case (name, accumulator) =>
        try {
          accumulator.reload()
        } catch {
          case _: DirectoryLoader.DirectoryDeletedException => namesToDelete += name
          case t: Throwable => EventMultiplexer.logger.error(s"Unable to reload accumulator '$name'.", t)
        }
    }
    AccumulatorsLock synchronized {
      namesToDelete.foreach(name => {
        EventMultiplexer.logger.warn(s"Deleting accumulator '$name'.")
        this.accumulators.remove(name)
        this.paths.remove(name)
      })
    }
    EventMultiplexer.logger.info("Finished the multiplexer reload.")
    this
  }

  /** Returns the timestamp (in seconds) of the first event for the specified run name.
    *
    * If the first event has been loaded (either by this method or by `reload()`, then this method returns immediately.
    * Otherwise, it loads the first event and then returns. Note that this means that calling `reload()` will cause this
    * method to block until `reload()` has finished. */
  def firstEventTimeStamp(run: String): Option[Double] = {
    accumulator(run).map(_.firstEventTimeStamp)
  }

  /** Returns all scalar events associated with the provided run and summary tag. */
  def scalars(run: String, tag: String): Option[List[ScalarEventRecord]] = {
    accumulator(run).map(_.scalars(tag))
  }

  /** Returns all image events associated with the provided run and summary tag. */
  def images(run: String, tag: String): Option[List[ImageEventRecord]] = {
    accumulator(run).map(_.images(tag))
  }

  /** Returns all audio events associated with the provided run and summary tag. */
  def audio(run: String, tag: String): Option[List[AudioEventRecord]] = {
    accumulator(run).map(_.audio(tag))
  }

  /** Returns all histogram events associated with the provided run and summary tag. */
  def histograms(run: String, tag: String): Option[List[HistogramEventRecord]] = {
    accumulator(run).map(_.histograms(tag))
  }

  /** Returns all compressed histogram events associated with the provided run and summary tag. */
  def compressedHistograms(run: String, tag: String): Option[List[CompressedHistogramEventRecord]] = {
    accumulator(run).map(_.compressedHistograms(tag))
  }

  /** Returns all tensor events associated with the provided run and summary tag. */
  def tensors(run: String, tag: String): Option[List[TensorEventRecord]] = {
    accumulator(run).map(_.tensors(tag))
  }

  /** Returns the graph definition for the provided run, if there is one.
    *
    * If the graph is stored directly, the method returns it. If no graph is stored directly, but a meta-graph is stored
    * containing a graph, the method returns that graph. */
  def graph(run: String): Option[GraphDef] = {
    accumulator(run).map(_.graph)
  }

  /** Returns the meta-graph definition for the provided run, if there is one. */
  def metaGraph(run: String): Option[MetaGraphDef] = {
    accumulator(run).map(_.metaGraph)
  }

  /** Returns the run metadata associated with the provided run and summary tag. */
  def runMetadata(run: String, tag: String): Option[RunMetadata] = {
    accumulator(run).map(_.runMetadata(tag))
  }

  /** Returns the summary metadata associated with the provided run and summary tag. */
  def summaryMetadata(run: String, tag: String): Option[SummaryMetadata] = {
    accumulator(run).map(_.summaryMetadata(tag))
  }

  /** Returns a map from runs to a map from tags to content specific to the specified plugin for that run. */
  def pluginTagToContent(pluginName: String): Map[String, Map[String, String]] = {
    runTags.keys
        .map(run => run -> accumulator(run).flatMap(_.pluginTagToContent(pluginName)))
        .filter(_._2.isDefined)
        .toMap
        .view
        .mapValues(_.get)
        .toMap
  }

  /** Returns a map from runs to sequences with paths to all the registered assets for the provided plugin name, for
    * that run.
    *
    * If a plugins directory does not exist in the managed directory, then this method returns an empty list. This
    * maintains compatibility with old log directories that contain no plugin sub-directories.
    */
  def pluginAssets(pluginName: String): Map[String, Seq[Path]] = {
    AccumulatorsLock.synchronized(accumulators.toList).map(a => a._1 -> a._2.pluginAssets(pluginName)).toMap
  }

  /** Retrieves a particular plugin asset (for the provided run) from the managed directory and returns it as a
    * string. */
  def retrievePluginAsset(run: String, pluginName: String, assetName: String): Option[String] = {
    accumulator(run).map(_.retrievePluginAsset(pluginName, assetName))
  }

  /** Returns the event accumulator that corresponds to the provided run name. */
  def accumulator(run: String): Option[EventAccumulator] = AccumulatorsLock synchronized {
    accumulators.get(run)
  }

  /** Returns a map from run names to maps from event types to tags. */
  def runTags: Map[String, Map[EventType, Seq[String]]] = {
    // Build a list to avoid nested locks.
    AccumulatorsLock.synchronized(accumulators.toList).map(a => a._1 -> a._2.tags).toMap
  }

  /** Returns a map from run names to event file paths. */
  def runPaths: Map[String, Path] = AccumulatorsLock synchronized {
    paths.toMap
  }
}

object EventMultiplexer {
  private[EventMultiplexer] val logger: Logger = Logger(LoggerFactory.getLogger("Event Multiplexer"))
}
