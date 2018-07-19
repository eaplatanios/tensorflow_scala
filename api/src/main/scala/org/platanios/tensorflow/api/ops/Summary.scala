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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING, UINT8}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import spire.math.UByte

import java.nio.file.Path

import scala.util.matching.Regex

/** Contains functions for constructing ops related to summaries (e.g., to be used with TensorBoard).
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Summary {
  /** $OpDocSummaryTensor
    *
    * @group SummaryOps
    * @param  name        Name for the created summary op.
    * @param  tensor      Tensor to use for the summary.
    * @param  collections Graph collections in which to add the new summary op. Defaults to `Graph.Keys.SUMMARIES`.
    * @param  family      If provided, used as prefix for the summary tag name, which controls the tab name used for
    *                     display on TensorBoard.
    * @return Created op output.
    */
  def tensor(
      name: String,
      tensor: Output,
      collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
      family: String = null
  ): Output = {
    Summary.scoped((scope, tag) => {
      val summary = Summary.tensorSummary(tensor, tag, Tensor(STRING), scope)
      collections.foreach(key => Op.currentGraph.addToCollection(summary, key))
      summary
    }, name, family)
  }

  /** $OpDocSummaryScalar
    *
    * @group SummaryOps
    * @param  name        Name for the created summary op.
    * @param  value       Scalar tensor containing the value to use for the summary.
    * @param  collections Graph collections in which to add the new summary op. Defaults to `Graph.Keys.SUMMARIES`.
    * @param  family      If provided, used as prefix for the summary tag name, which controls the tab name used for
    *                     display on TensorBoard.
    * @return Created op output.
    */
  def scalar(
      name: String,
      value: Output,
      collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
      family: String = null
  ): Output = {
    Summary.scoped((scope, tag) => {
      val summary = Summary.scalarSummary(value, tag, scope)
      collections.foreach(key => Op.currentGraph.addToCollection(summary, key))
      summary
    }, name, family)
  }

  /** $OpDocSummaryHistogram
    *
    * @group SummaryOps
    * @param  name        Name for the created summary op.
    * @param  values      Values to use to build the histogram.
    * @param  collections Graph collections in which to add the new summary op. Defaults to `Graph.Keys.SUMMARIES`.
    * @param  family      If provided, used as prefix for the summary tag name, which controls the tab name used for
    *                     display on TensorBoard.
    * @return Created op output.
    */
  def histogram(
      name: String,
      values: Output,
      collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
      family: String = null
  ): Output = {
    Summary.scoped((scope, tag) => {
      val summary = Summary.histogramSummary(values, tag, scope)
      collections.foreach(key => Op.currentGraph.addToCollection(summary, key))
      summary
    }, name, family)
  }

  /** $OpDocSummaryImage
    *
    * @group SummaryOps
    * @param  name        Name for the created summary op.
    * @param  tensor      Four-dimensional tensor with shape `[batchSize, height, width, channels]` where `channels` is
    *                     1, 3, or 4.
    * @param  badColor    Color to use for pixels with non-finite values. Defaults to red color.
    * @param  maxOutputs  Maximum number of batch elements for which to generate images.
    * @param  collections Graph collections in which to add the new summary op. Defaults to `Graph.Keys.SUMMARIES`.
    * @param  family      If provided, used as prefix for the summary tag name, which controls the tab name used for
    *                     display on TensorBoard.
    * @return Created op output.
    */
  def image(
      name: String,
      tensor: Output,
      badColor: Tensor[DataType] = Tensor(UINT8, UByte(255), UByte(0), UByte(0), UByte(255)),
      maxOutputs: Int = 3,
      collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
      family: String = null
  ): Output = {
    Summary.scoped((scope, tag) => {
      val summary = Summary.imageSummary(tensor, badColor, tag, maxOutputs, scope)
      collections.foreach(key => Op.currentGraph.addToCollection(summary, key))
      summary
    }, name, family)
  }

  /** $OpDocSummaryAudio
    *
    * @group SummaryOps
    * @param  name         Name for the created summary op.
    * @param  tensor       Three-dimensional tensor with shape `[batchSize, frames, channels]` or two-dimensional tensor
    *                      with shape `[batchSize, frames]`.
    * @param  samplingRate Scalar tensor containing the sampling rate of the audio signal, in Hertz.
    * @param  maxOutputs   Maximum number of batch elements for which to generate audio.
    * @param  collections  Graph collections in which to add the new summary op. Defaults to `Graph.Keys.SUMMARIES`.
    * @param  family       If provided, used as prefix for the summary tag name, which controls the tab name used for
    *                      display on TensorBoard.
    * @return Created op output.
    */
  def audio(
      name: String,
      tensor: Output,
      samplingRate: Output,
      maxOutputs: Int = 3,
      collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
      family: String = null
  ): Output = {
    Summary.scoped((scope, tag) => {
      val summary = Summary.audioSummary(tensor, samplingRate, tag, maxOutputs, scope)
      collections.foreach(key => Op.currentGraph.addToCollection(summary, key))
      summary
    }, name, family)
  }

  /** $OpDocSummaryMergeSummaries
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
      summaries: Set[Output],
      collections: Set[Graph.Key[Output]] = Set.empty,
      name: String = "MergeSummaries"
  ): Output = {
    val cleanedName = Summary.sanitizeName(name)
    Op.createWithNameScope(cleanedName, summaries.map(_.op)) {
      val merged = Summary.mergeSummaries(summaries.toSeq, cleanedName)
      collections.foreach(k => Op.currentGraph.addToCollection(merged, k))
      merged
    }
  }

  /** $OpDocSummaryMergeAllSummaries
    *
    * @group SummaryOps
    * @param  key Graph collection key used to collect the summaries. Defaults to `Graph.Keys.SUMMARIES`.
    * @return Created op output, or `None`, if no summaries could be found in the current graph. The op output is a
    *         [[STRING]] scalar tensor containing the serialized `Summary` protocol buffer resulting from the merge.
    */
  def mergeAll(key: Graph.Key[Output] = Graph.Keys.SUMMARIES, name: String = "MergeAllSummaries"): Option[Output] = {
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

  /** Executes `function` and returns its result from within a custom name scope.
    *
    * To ensure that the summary tag name is always unique, we create a name scope based on `name` and use the full
    * scope name in the tag.
    *
    * If `family` is set, then the tag name will be `<family>/<scopeName>`, where `scopeName` is
    * `<outerScope>/<family>/<name>`. This ensures that `family` is always the prefix of the tag (and unmodified), while
    * ensuring that the scope respects the outer scope from where this summary was created.
    *
    * `function` has to take the custom name scope and the created tag as arguments and generate a summary op output.
    *
    * @param  function    Function that takes the custom name scope and tag as arguments and returns the created summary
    *                     op output.
    * @param  name        Name for the created summary op.
    * @param  family      If provided, used as prefix for the summary tag name.
    * @return Created summary op output.
    */
  private[Summary] def scoped(function: (String, String) => Output, name: String, family: String = null): Output = {
    val sanitizedName = sanitizeName(name)
    val sanitizedFamily = sanitizeName(family)
    // Use the family name in the scope to ensure uniqueness of scope/tag.
    val nameScope = if (sanitizedFamily == null) sanitizedName else s"$sanitizedFamily/$sanitizedName"
    Op.createWithNameScope(nameScope) {
      val scope = Op.currentNameScope
      // If a family is provided, we prefix our scope with the family again so it displays in the right tab.
      val tag = if (sanitizedFamily == null) scope.stripSuffix("/") else s"$sanitizedFamily/${scope.stripSuffix("/")}"
      // Note that the tag is not 100% unique if the user explicitly enters a scope with the same name as family, and
      // then later enters it again before the summaries. This is very contrived though and we opt here to let it be a
      // runtime exception if tags do indeed collide.
      function(scope, tag)
    }
  }

  /** $OpDocSummaryTensor
    *
    * @group SummaryOps
    * @param  tensor          Tensor to serialize.
    * @param  tag             Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  summaryMetadata Serialized `SummaryMetadata` protocol buffer containing plugin-related metadata for this
    *                         summary.
    * @param  name            Name for the created op.
    * @return Created op output.
    */
  private[Summary] def tensorSummary(
      tensor: Output,
      tag: Output,
      summaryMetadata: Output,
      name: String = "TensorSummary"
  ): Output = {
    Op.Builder("TensorSummaryV2", name)
        .addInput(tensor)
        .addInput(tag)
        .addInput(summaryMetadata)
        .build().outputs(0)
  }

  /** $OpDocSummaryScalar
    *
    * @group SummaryOps
    * @param  value Value to serialize.
    * @param  tags  Tags to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  private[Summary] def scalarSummary(value: Output, tags: Output, name: String = "ScalarSummary"): Output = {
    Op.Builder("ScalarSummary", name)
        .addInput(tags)
        .addInput(value)
        .build().outputs(0)
  }

  /** $OpDocSummaryHistogram
    *
    * @group SummaryOps
    * @param  values Values to use to build the histogram.
    * @param  tag    Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  private[Summary] def histogramSummary(values: Output, tag: Output, name: String = "HistogramSummary"): Output = {
    Op.Builder("HistogramSummary", name)
        .addInput(tag)
        .addInput(values)
        .build().outputs(0)
  }

  /** $OpDocSummaryImage
    *
    * @group SummaryOps
    * @param  tensor     Four-dimensional tensor with shape `[batchSize, height, width, channels]` where `channels` is
    *                    1, 3, or 4.
    * @param  badColor   Color to use for pixels with non-finite values.
    * @param  tag        Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  maxOutputs Maximum number of batch elements for which to generate images.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  private[Summary] def imageSummary(
      tensor: Output,
      badColor: Tensor[DataType],
      tag: Output,
      maxOutputs: Int = 3,
      name: String = "ImageSummary"
  ): Output = {
    Op.Builder("ImageSummary", name)
        .addInput(tag)
        .addInput(tensor)
        .setAttribute("bad_color", badColor)
        .setAttribute("max_images", maxOutputs)
        .build().outputs(0)
  }

  /** $OpDocSummaryAudio
    *
    * @group SummaryOps
    * @param  tensor       Three-dimensional tensor with shape `[batchSize, frames, channels]` or two-dimensional tensor
    *                      with shape `[batchSize, frames]`.
    * @param  samplingRate Scalar tensor containing the sampling rate of the audio signal, in Hertz.
    * @param  tag          Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  maxOutputs   Maximum number of batch elements for which to generate audio.
    * @param  name         Name for the created op.
    * @return Created op output.
    */
  private[Summary] def audioSummary(
      tensor: Output,
      samplingRate: Output,
      tag: Output,
      maxOutputs: Int = 3,
      name: String = "AudioSummary"
  ): Output = {
    Op.Builder("AudioSummaryV2", name)
        .addInput(tag)
        .addInput(tensor)
        .addInput(samplingRate)
        .setAttribute("max_outputs", maxOutputs)
        .build().outputs(0)
  }

  /** $OpDocSummaryMergeSummaries
    *
    * @group SummaryOps
    * @param  inputs Input summary tensors that can be of any shape, but each must contain serialized `Summary` protocol
    *                buffers.
    * @param  name   Name for the created op.
    * @return Created op output, which is a [[STRING]] scalar tensor containing the serialized `Summary` protocol buffer
    *         resulting from the merge.
    */
  private[Summary] def mergeSummaries(inputs: Seq[Output], name: String = "MergeSummaries"): Output = {
    Op.Builder("MergeSummary", name)
        .addInputList(inputs)
        .build().outputs(0)
  }

  /** $OpDocSummaryWriter
    *
    * @group SummaryOps
    * @param  sharedName Shared name for the summary writer.
    * @param  container  Resource container for the summary writer.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  private[Summary] def summaryWriter(
      sharedName: String = "",
      container: String = "",
      name: String = "SummaryWriter"
  ): Output = {
    Op.Builder("SummaryWriter", name)
        .setAttribute("shared_name", sharedName)
        .setAttribute("container", container)
        .build().outputs(0)
  }

  /** $OpDocSummaryCreateSummaryFileWriter
    *
    * @group SummaryOps
    * @param  writerHandle   Handle to a summary writer resource.
    * @param  workingDir     Directory in which to write the event file.
    * @param  queueCapacity  Maximum number of events pending to be written to disk before a call to `write()` blocks.
    * @param  flushFrequency Specifies how often to flush the written events to disk (in seconds).
    * @param  filenameSuffix Filename suffix to use for the event file.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[Summary] def createSummaryFileWriter(
      writerHandle: Output,
      workingDir: Path,
      queueCapacity: Int = 10,
      flushFrequency: Int = 10,
      filenameSuffix: String = "",
      name: String = "CreateSummaryFileWriter"
  ): Op = {
    Op.Builder("CreateSummaryFileWriter", name)
        .addInput(writerHandle)
        .addInput(workingDir.toString)
        .addInput(queueCapacity)
        .addInput(flushFrequency)
        .addInput(filenameSuffix)
        .build()
  }

  /** $OpDocSummaryFlushSummaryWriter
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def flushSummaryWriter(writerHandle: Output, name: String = "FlushSummaryWriter"): Op = {
    Op.Builder("FlushSummaryWriter", name)
        .addInput(writerHandle)
        .build()
  }

  /** $OpDocSummaryCloseSummaryWriter
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def closeSummaryWriter(writerHandle: Output, name: String = "CloseSummaryWriter"): Op = {
    Op.Builder("CloseSummaryWriter", name)
        .addInput(writerHandle)
        .build()
  }

  /** $OpDocSummaryWriteTensorSummary
    *
    * @group SummaryOps
    * @param  writerHandle    Handle to a summary writer resource.
    * @param  globalStep      [[INT64]] tensor containing the global step to write the summary for.
    * @param  tag             Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  tensor          Tensor to serialize.
    * @param  summaryMetadata Serialized `SummaryMetadata` protocol buffer containing plugin-related metadata for this
    *                         summary.
    * @param  name            Name for the created op.
    * @return Created op.
    */
  private[Summary] def writeTensorSummary(
      writerHandle: Output,
      globalStep: Output,
      tag: Output,
      tensor: Output,
      summaryMetadata: Output,
      name: String = "WriteTensorSummary"
  ): Op = {
    Op.Builder("WriteSummary", name)
        .addInput(writerHandle)
        .addInput(globalStep)
        .addInput(tensor)
        .addInput(tag)
        .addInput(summaryMetadata)
        .build()
  }

  /** $OpDocSummaryWriteScalarSummary
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  globalStep   [[INT64]] tensor containing the global step to write the summary for.
    * @param  value        Value to serialize.
    * @param  tags         Tags to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def writeScalarSummary(
      writerHandle: Output,
      globalStep: Output,
      value: Output,
      tags: Output,
      name: String = "WriteScalarSummary"
  ): Op = {
    Op.Builder("WriteScalarSummary", name)
        .addInput(writerHandle)
        .addInput(globalStep)
        .addInput(tags)
        .addInput(value)
        .build()
  }

  /** $OpDocSummaryWriteHistogramSummary
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  globalStep   [[INT64]] tensor containing the global step to write the summary for.
    * @param  values       Values to use to build the histogram.
    * @param  tag          Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def writeHistogramSummary(
      writerHandle: Output,
      globalStep: Output,
      values: Output,
      tag: Output,
      name: String = "WriteHistogramSummary"
  ): Op = {
    Op.Builder("WriteScalarSummary", name)
        .addInput(writerHandle)
        .addInput(globalStep)
        .addInput(tag)
        .addInput(values)
        .build()
  }

  /** $OpDocSummaryWriteImageSummary
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  globalStep   [[INT64]] tensor containing the global step to write the summary for.
    * @param  tensor       Four-dimensional tensor with shape `[batchSize, height, width, channels]` where `channels` is
    *                      1, 3, or 4.
    * @param  badColor     Color to use for pixels with non-finite values.
    * @param  tag          Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  maxOutputs   Maximum number of batch elements for which to generate images.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def writeImageSummary(
      writerHandle: Output,
      globalStep: Output,
      tensor: Output,
      badColor: Output,
      tag: Output,
      maxOutputs: Int = 3,
      name: String = "WriteImageSummary"
  ): Op = {
    Op.Builder("WriteImageSummary", name)
        .addInput(writerHandle)
        .addInput(globalStep)
        .addInput(tag)
        .addInput(tensor)
        .addInput(badColor)
        .setAttribute("max_images", maxOutputs)
        .build()
  }

  /** $OpDocSummaryWriteAudioSummary
    *
    * @group SummaryOps
    * @param  writerHandle Handle to a summary writer resource.
    * @param  globalStep   [[INT64]] tensor containing the global step to write the summary for.
    * @param  tensor       Three-dimensional tensor with shape `[batchSize, frames, channels]` or two-dimensional tensor
    *                      with shape `[batchSize, frames]`.
    * @param  samplingRate Scalar tensor containing the sampling rate of the audio signal, in Hertz.
    * @param  tag          Tag to use for the created summary. Used for organizing summaries in TensorBoard.
    * @param  maxOutputs   Maximum number of batch elements for which to generate audio.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[Summary] def writeAudioSummary(
      writerHandle: Output,
      globalStep: Output,
      tensor: Output,
      samplingRate: Output,
      tag: Output,
      maxOutputs: Int = 3,
      name: String = "WriteAudioSummary"
  ): Op = {
    Op.Builder("WriteAudioSummary", name)
        .addInput(writerHandle)
        .addInput(globalStep)
        .addInput(tag)
        .addInput(tensor)
        .addInput(samplingRate)
        .setAttribute("max_outputs", maxOutputs)
        .build()
  }

  /** @define OpDocSummaryTensorSummary
    *   The `tensorSummary` op outputs a `Summary` protocol buffer containing a tensor and per-plugin data.
    *
    * @define OpDocSummaryScalarSummary
    *   The `scalarSummary` op outputs a `Summary` protocol buffer containing scalar values.
    *
    *   The input `tags` and `values` must have the same shape. The generated summary contains a summary value for each
    *   tag-value pair in `tags` and `values`.
    *
    * @define OpDocSummaryHistogramSummary
    *   The `histogramSummary` op outputs a `Summary` protocol buffer containing a histogram.
    *
    *   Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard. You can see a
    *   detailed explanation of the TensorBoard histogram dashboard
    *   [here](https://www.tensorflow.org/get_started/tensorboard_histograms).
    *
    *   The generated [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) has one
    *   summary value containing a histogram for the provided values.
    *
    *   This op will throw an [[IllegalArgumentException]] if any of the provided values is not finite.
    *
    * @define OpDocSummaryImageSummary
    *   The `imageSummary` op outputs a `Summary` protocol buffer containing images.
    *
    *   The summary has up to `max_images` summary values containing images. The images are built from the provided
    *   tensor which must be 4-dimensional with shape `[batchSize, height, width, channels]` and where `channels` can
    *   be:
    *
    *     - 1: `tensor` is interpreted as Gray-scale.
    *     - 3: `tensor` is interpreted as RGB.
    *     - 4: `tensor` is interpreted as RGBA.
    *
    *   The images have the same number of channels as the input tensor. For float input, the values are normalized one
    *   image at a time to fit in the range `[0, 255]`. [[UINT8]] values are unchanged. The op uses two different
    *   normalization algorithms:
    *
    *     - If the input values are all positive, they are rescaled so the largest one is 255.
    *     - If any input value is negative, the values are shifted so input value 0.0 is at 127 . hey are then rescaled
    *       so that either the smallest value is 0, or the largest one is 255.
    *
    *   The `tag` argument must be a [[STRING]] scalar tensor. It is used to build the `tag` of the summary values:
    *
    *     - If `maxOutputs` is 1, the summary value tag is `<tag>/image`.
    *     - If `maxOutputs` is greater than 1, the summary value tags are generated sequentially as `<tag>/image/0`,
    *       `<tag>/image/1`, etc.
    *
    *   The `badColor` argument is the color to use in the generated images for non-finite input values. It must be a
    *   [[UINT8]] one-dimensional tensor with length equal to `channels`. Each element must be in the range `[0, 255]`
    *   (it represents the value of a pixel in the output image). Non-finite values in the input tensor are replaced by
    *   this tensor in the output image.
    *
    * @define OpDocSummaryAudioSummary
    *   The `audioSummary` op writes a `Summary` protocol buffer containing audio, to the provided summary writer.
    *
    *   The generated summary contains up to `maxOutputs` summary values containing audio. The audio is built from the
    *   provided tensor which must be three-dimensional with shape `[batchSize, frames, channels]` or two-dimensional
    *   with shape `[batchSize, frames]`. The values are assumed to be in the range `[-1.0, 1.0]` and sampled at the
    *   provided frequency.
    *
    *   The `tag` argument must be a [[STRING]] scalar tensor. It is used to build the `tag` of the summary values:
    *
    *     - If `maxOutputs` is 1, the summary value tag is `<tag>/audio`.
    *     - If `maxOutputs` is greater than 1, the summary value tags are generated sequentially as `<tag>/audio/0`,
    *       `<tag>/audio/1`, etc.
    *
    * @define OpDocSummaryMergeSummaries
    *   The `mergeSummaries` op merges summaries.
    *
    *   The op creates a [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) protocol
    *   buffer that contains the union of all the values in the input summaries.
    *
    *   When the op is executed, it throws an [[IllegalArgumentException]] error if multiple values in the summaries to
    *   merge use the same tag.
    *
    * @define OpDocSummaryMergeAllSummaries
    *   The `mergeAllSummaries` op merges all summaries in the current graph.
    *
    * @define OpDocSummaryWriter
    *   The `summaryWriter` op returns a handle to be used to access a summary writer.
    *
    *   The summary writer is an in-graph resource which can be used by ops to write summaries to event files.
    *
    * @define OpDocSummaryCreateSummaryFileWriter
    *   The `createSummaryFileWriter` op creates a summary file writer accessible by the given resource handle.
    *
    * @define OpDocSummaryFlushSummaryWriter
    *   The `flushSummaryWriter` op flushes the provided summary writer's unwritten events.
    *
    * @define OpDocSummaryCloseSummaryWriter
    *   The `closeSummaryWriter` op flushes and closes the provided summary writer.
    *
    *   The op also removes the summary writer from the resource manager. To reopen it, you have to use another
    *   `createSummaryFileWriter` op.
    *
    * @define OpDocSummaryWriteTensorSummary
    *   The `writeTensorSummary` op writes a `Summary` protocol buffer containing a tensor and per-plugin data, to the
    *   provided summary writer.
    *
    * @define OpDocSummaryWriteScalarSummary
    *   The `writeScalarSummary` op writes a `Summary` protocol buffer containing scalar values, to the provided summary
    *   writer.
    *
    *   The input `tags` and `values` must have the same shape. The generated summary contains a summary value for each
    *   tag-value pair in `tags` and `values`.
    *
    * @define OpDocSummaryWriteHistogramSummary
    *   The `writeHistogramSummary` op writes a `Summary` protocol buffer containing a histogram, to the provided
    *   summary writer.
    *
    *   Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard. You can see a
    *   detailed explanation of the TensorBoard histogram dashboard
    *   [here](https://www.tensorflow.org/get_started/tensorboard_histograms).
    *
    *   The generated [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) has one
    *   summary value containing a histogram for the provided values.
    *
    *   This op will throw an [[IllegalArgumentException]] if any of the provided values is not finite.
    *
    * @define OpDocSummaryWriteImageSummary
    *   The `writeImageSummary` op writes a `Summary` protocol buffer containing images, to the provided summary writer.
    *
    *   The summary has up to `max_images` summary values containing images. The images are built from the provided
    *   tensor which must be 4-dimensional with shape `[batchSize, height, width, channels]` and where `channels` can
    *   be:
    *
    *     - 1: `tensor` is interpreted as Gray-scale.
    *     - 3: `tensor` is interpreted as RGB.
    *     - 4: `tensor` is interpreted as RGBA.
    *
    *   The images have the same number of channels as the input tensor. For float input, the values are normalized one
    *   image at a time to fit in the range `[0, 255]`. [[UINT8]] values are unchanged. The op uses two different
    *   normalization algorithms:
    *
    *     - If the input values are all positive, they are rescaled so the largest one is 255.
    *     - If any input value is negative, the values are shifted so input value 0.0 is at 127 . hey are then rescaled
    *       so that either the smallest value is 0, or the largest one is 255.
    *
    *   The `tag` argument must be a [[STRING]] scalar tensor. It is used to build the `tag` of the summary values:
    *
    *     - If `maxOutputs` is 1, the summary value tag is `<tag>/image`.
    *     - If `maxOutputs` is greater than 1, the summary value tags are generated sequentially as `<tag>/image/0`,
    *       `<tag>/image/1`, etc.
    *
    *   The `badColor` argument is the color to use in the generated images for non-finite input values. It must be a
    *   [[UINT8]] one-dimensional tensor with length equal to `channels`. Each element must be in the range `[0, 255]`
    *   (it represents the value of a pixel in the output image). Non-finite values in the input tensor are replaced by
    *   this tensor in the output image.
    *
    * @define OpDocSummaryWriteAudioSummary
    *   The `writeAudioSummary` op writes a `Summary` protocol buffer containing audio, to the provided summary writer.
    *
    *   The generated summary contains up to `maxOutputs` summary values containing audio. The audio is built from the
    *   provided tensor which must be three-dimensional with shape `[batchSize, frames, channels]` or two-dimensional
    *   with shape `[batchSize, frames]`. The values are assumed to be in the range `[-1.0, 1.0]` and sampled at the
    *   provided frequency.
    *
    *   The `tag` argument must be a [[STRING]] scalar tensor. It is used to build the `tag` of the summary values:
    *
    *     - If `maxOutputs` is 1, the summary value tag is `<tag>/audio`.
    *     - If `maxOutputs` is greater than 1, the summary value tags are generated sequentially as `<tag>/audio/0`,
    *       `<tag>/audio/1`, etc.
    */
  private[ops] trait Documentation
}
