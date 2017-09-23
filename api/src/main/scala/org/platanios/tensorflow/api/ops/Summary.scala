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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.types.STRING

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

// TODO: [SUMMARY] [OPS] Add support for creating summary ops.

/** Contains functions for constructing ops related to summaries (e.g., to be used with TensorBoard).
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Summary {
  /** $OpDocSummaryMerge
    *
    * @group SummaryOps
    * @param  summaries   Input summary tensors that can be of any shape, but each must contain serialized `Summary`
    *                     protocol buffers.
    * @param  collections Set of graph collection keys. The created merged summary op will be added to the corresponding
    *                     collections.
    * @param  name        Name for the created op.
    * @return Created op output, which is a [[STRING]] scalar tensor containing the serialized `Summary` protocol buffer
    *         resulting from the merge.
    */
  def merge(
      summaries: Set[Output], collections: Set[Graph.Key[Output]] = Set.empty,
      name: String = "SummariesMerge"): Output = {
    val cleanedName = Summary.sanitizeName(name)
    Op.createWithNameScope(cleanedName, summaries.map(_.op)) {
      val merged = Summary.mergeSummary(summaries.toSeq, cleanedName)
      collections.foreach(k => Op.currentGraph.addToCollection(merged, k))
      merged
    }
  }

  /** $OpDocSummaryMergeAll
    *
    * @group SummaryOps
    * @param  key Graph collection key used to collect the summaries. Defaults to `Graph.Keys.SUMMARIES`.
    * @return Created op output, or `None`, if no summaries could be found in the current graph. The op output is a
    *         [[STRING]] scalar tensor containing the serialized `Summary` protocol buffer resulting from the merge.
    */
  def mergeAll(key: Graph.Key[Output] = Graph.Keys.SUMMARIES, name: String = "AllSummariesMerge"): Option[Output] = {
    val summaries = Op.currentGraph.getCollection(key)
    if (summaries.isEmpty)
      None
    else
      Some(merge(summaries, name = name))
  }
}

/** Contains helper methods for creating summary ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] object Summary extends Summary {
  private[Summary] val logger = Logger(LoggerFactory.getLogger("Ops / Summary"))

  private[this] val INVALID_TAG_CHARACTERS: Regex = "[^-/a-zA-Z0-9_.]".r

  /** Returns the set of all table initializers that have been created in the current graph. */
  def initializers: Set[Op] = Op.currentGraph.tableInitializers

  /** Creates an op that groups the provided table initializers.
    *
    * @param  initializers Table initializers to group.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  def initializer(initializers: Set[Op], name: String = "TablesInitializer"): Op = {
    if (initializers.isEmpty)
      ControlFlow.noOp(name)
    else
      ControlFlow.group(initializers, name)
  }

  /** Clean a tag name by replacing all illegal characters with underscores and stripping any leading slashes. */
  private[Summary] def sanitizeName(name: String): String = {
    if (name != null) {
      // In the past, the first argument to summary ops was a tag, which allowed arbitrary characters. Now we are
      // changing the first argument to be the node name. This has a number of advantages (users of summary ops can now
      // take advantage of the op creation name scope support) but risks breaking existing usage, because a much smaller
      // set of characters are allowed in node names. This function replaces all illegal characters with underscores,
      // and logs a warning. It also strips leading slashes from the name.
      val newName = INVALID_TAG_CHARACTERS.replaceAllIn(name, "_").replaceAll("^/+", "")
      if (newName != name)
        logger.warn(s"Summary name '$name' is invalid; using '$newName' instead.")
      newName
    } else {
      null
    }
  }

  // TODO: !!! [SUMMARIES] Add support for plugin assets.
//  /** This trait allows TensorBoard to serialize assets to disk.
//    *
//    * LifeCycle of a [[PluginAsset]] instance:
//    *
//    *   - Constructed when `getPluginAsset` is called on the class for the first time.
//    *   - Configured by code that follows the calls to `getPluginAsset`.
//    *   - When the containing graph is serialized by the [[SummaryFileWriter]], the writer requests `PluginAsset.assets`
//    *     and the [[PluginAsset]] instance provides its contents to be written to disk.
//    */
//  trait PluginAsset {
//    /** Name of the plugin associated with this asset. */
//    val pluginName: String
//
//    /** Assets contained in this [[PluginAsset]] instance.
//      *
//      * This method will be called by the [[SummaryFileWriter]] when it is time to write the assets to disk.
//      *
//      * @return Map from asset names to asset contents.
//      */
//    val assets: Map[String, String]
//  }

  /** $OpDocSummaryMerge
    *
    * @group SummaryOps
    * @param  inputs Input summary tensors that can be of any shape, but each must contain serialized `Summary` protocol
    *                buffers.
    * @param  name   Name for the created op.
    * @return Created op output, which is a [[STRING]] scalar tensor containing the serialized `Summary` protocol buffer
    *         resulting from the merge.
    */
  private[Summary] def mergeSummary(inputs: Seq[Output], name: String = "MergeSummary"): Output = {
    Op.Builder("MergeSummary", name)
        .addInputList(inputs)
        .build().outputs(0)
  }

  /** @define OpDocSummaryMerge
    * The `merge` op merges summaries.
    *
    * The op creates a [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) protocol
    * buffer that contains the union of all the values in the input summaries.
    *
    * When the op is executed, it throws an [[IllegalArgumentException]] error if multiple values in the summaries to
    * merge use the same tag.
    * @define OpDocSummaryMergeAll
    * The `mergeAll` op merges all summaries in the current graph.
    */
  private[ops] trait Documentation
}
