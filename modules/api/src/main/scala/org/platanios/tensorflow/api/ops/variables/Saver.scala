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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.{DeviceSpecification, Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.types.{DataType, TF, IsInt32OrInt64}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.io.FileIO
import org.platanios.tensorflow.api.ops.{Basic, Op, Output, Text, UntypedOp}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.CheckpointStateProto.CheckpointState
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.Proto
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import com.github.ghik.silencer.silent
import com.google.protobuf.TextFormat
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.framework.MetaGraphDef
import org.tensorflow.util.SaverDef
import org.tensorflow.util.SaverDef.CheckpointFormatVersion

import java.nio.file.{Files, Path}
import java.util.UUID
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._
import scala.collection.mutable

/** A saver can save and restore variables and other saveable objects.
  *
  * This class adds ops to save and restore variables to and from *checkpoints*. It also provides convenience methods to
  * run these ops. Checkpoints are binary files in a proprietary format which map variable names to tensor values. The
  * best way to examine the contents of a checkpoint is to load it using a [[Saver]].
  *
  * Savers can automatically number checkpoint filenames. This lets you keep multiple checkpoints at different steps
  * while training a model. For example, you can number the checkpoint filenames with the training step number. To avoid
  * filling up disks, savers manage checkpoint files automatically. For example, they can make sure to keep only the `N`
  * most recent files, or one checkpoint for every `N` hours of training.
  *
  * You may number checkpoint filenames by passing a value to the optional `globalStep` argument of the `save` method.
  * For example:
  * {{{
  *   // Using a slight abuse of notation for paths:
  *   saver.save(session, "my-model", globalStep = 0) ==> filename: "my-model-0"
  *   saver.save(session, "my-model", globalStep = 1000) ==> filename: "my-model-1000"
  * }}}
  *
  * Also, optional arguments to the `Saver` constructor let you control the proliferation of checkpoint files on disk:
  *   - `maxToKeep`: The maximum number of recent checkpoint files to keep. As new files are created, older files are
  *     deleted. If `0`, no checkpoints are deleted from the filesystem but only the last one is kept in the
  *     `checkpoint` file. Defaults to `5` (i.e., only the 5 most recent checkpoint files are kept).
  *   - `keepCheckpointEveryNHours`: In addition to keeping the most recent `maxToKeep` checkpoint files, you might want
  *     to keep one checkpoint file for every `N` hours of training. This can be useful if you want to later analyze how
  *     a model progressed during a long training session. For example, passing `keepCheckpointEveryNHours = 2` ensures
  *     that you keep one checkpoint file for every 2 hours of training. The default value of `10000` hours effectively
  *     disables the feature.
  * Note that you still have to call the `save` method every time you want to save the model. Passing these arguments to
  * the constructor will not save variables automatically for you.
  *
  * An example training program that saves regularly looks like this:
  * {{{
  *   // Using a slight abuse of notation.
  *   // Create a saver.
  *   val saver = tf.Saver(variables)
  *   // Launch the graph and train, saving the model every 1,000 steps.
  *   for (step <- 0 to 1000000) {
  *     session.run(trainOp)
  *     if (step % 1000 == 0) {
  *       // Append the step number to the checkpoint name.
  *       saver.save(session, "my-model", globalStep = step)
  *     }
  *   }
  * }}}
  *
  * In addition to checkpoint files, savers keep a protocol buffer on disk with the list of recent checkpoints. This is
  * used to manage numbered checkpoint files. The `latestCheckpoint` method makes it easy to discover the path to the
  * most recent checkpoint. That protocol buffer is stored in a file next to the checkpoint files, with default name
  * `"checkpoint"` (can be provided using the `checkpointStateFilename` argument of the `save` method).
  *
  * @param  saverDef          [[SaverDef]] object containing all the properties of this saver.
  * @param  saveRelativePaths Boolean value which, if `true`, forces the saver to write relative paths to the checkpoint
  *                           state file. This is needed if the user wants to copy the checkpoint directory and restore
  *                           from the copied directory.
  * @param  padGlobalStep     Boolean value which, if `true`, forces the saver to pad the global step number in the
  *                           checkpoint file paths to some fixed width (`8` by default). This is turned off by default.
  *
  * @author Emmanouil Antonios Platanios
  */
class Saver private(
    saverDef: SaverDef,
    saveRelativePaths: Boolean = false,
    padGlobalStep: Boolean = false
) extends ProtoSerializable {
  val writerVersion: Saver.WriterVersion = saverDef.getVersion match {
    case CheckpointFormatVersion.V1 => Saver.V1
    case CheckpointFormatVersion.V2 => Saver.V2
    case _ => throw new IllegalArgumentException(s"Unsupported writer version '$writerVersion'.")
  }

  Saver.checkSaverDef(saverDef)

  private var lastCheckpoints: mutable.Queue[(Path, Long)] = {
    mutable.Queue.empty[(Path, Long)]
  }

  private var nextCheckpointTime: Float = {
    (System.currentTimeMillis() / 60000) + saverDef.getKeepCheckpointEveryNHours * 3600
  }

  /** Saves the current value of the saveables this saver is responsible for.
    *
    * This method runs the ops responsible for saving variables. It requires a session in which the saver's graph was
    * launched. The variables being saved must also have been initialized.
    *
    * The method returns the path of the newly created checkpoint file. This path can be passed directly to [[restore]].
    *
    * @param  session                 Session to use for saving the variables.
    * @param  savePath                Path to the checkpoint filename. If the saver is `sharded`, this is the prefix of
    *                                 the sharded checkpoint filenames
    * @param  globalStep              If provided, the global step number is appended to `savePath` to create the
    *                                 checkpoint filename.
    * @param  checkpointStateFilename Optional name for the protocol buffer file that will contain / contains the list
    *                                 of the most recent checkpoint filenames. That file, kept in the same directory as
    *                                 the checkpoint files, and it is automatically managed by the saver to keep track
    *                                 of recent checkpoints.
    * @param  metaGraphSuffix         Meta graph filename suffix.
    * @param  writeMetaGraph          Boolean value indicating whether or not to write the graph meta information file.
    * @param  writeCheckpointState    Boolean value indicating whether or not to write the checkpoint state file.
    * @return Path of the newly created checkpoint file, if the save operation was successful; `None`, otherwise. If the
    *         saver is sharded, the filename ends with `"-?????-of-nnnnn"` where `"nnnnn"` is the number of shards
    *         created.
    */
  def save(
      session: Session,
      savePath: Path,
      globalStep: Option[Int] = None,
      checkpointStateFilename: String = "checkpoint",
      metaGraphSuffix: String = "meta",
      writeMetaGraph: Boolean = true,
      writeCheckpointState: Boolean = true
  ): Option[Path] = {
    val absoluteSavePath = savePath.toAbsolutePath
    if (writerVersion != Saver.V2) {
      Saver.logger.warn("===========================================================")
      Saver.logger.warn("TensorFlow's V1 checkpoint format version has been deprecated.")
      Saver.logger.warn("Consider switching to the more efficient V2 format:")
      Saver.logger.warn("   `tf.Saver(writerVersion = tf.Saver.V2)`")
      Saver.logger.warn("V2 is the default checkpoint format version now.")
      Saver.logger.warn("===========================================================")
    }

    if (absoluteSavePath.getFileSystem.getPath(checkpointStateFilename).getNameCount > 1)
      throw new IllegalArgumentException(
        s"The 'checkpointStateFilename' must not contain any path components: $checkpointStateFilename.")
    val checkpointFile = {
      if (globalStep.isDefined) {
        // Optionally zero-pads the step numbers so that they are sorted when listed.
        if (padGlobalStep)
          absoluteSavePath.resolveSibling(f"${absoluteSavePath.getFileName}-${globalStep.get}%08d")
        else
          absoluteSavePath.resolveSibling(s"${absoluteSavePath.getFileName}-${globalStep.get}")
      } else if (absoluteSavePath.getFileName.toString == checkpointStateFilename && !saverDef.getSharded) {
        // Guard against collision between the data file and the checkpoint state file.
        throw new IllegalArgumentException(
          s"The checkpoint state filename ('$checkpointStateFilename') " +
              s"collides with the save path ('$absoluteSavePath').")
      } else {
        absoluteSavePath
      }
    }

    Saver.logger.info(s"Saving parameters to '$checkpointFile'.")

    // Save checkpoint.
    val savePathParent = absoluteSavePath.getParent
    val modelCheckpointPath = {
      try {
        val filenameTensor = session.graph.getOutputByName(
          saverDef.getFilenameTensorName
        ).asInstanceOf[Output[String]]
        val saveTensor = session.graph.getOutputByName(
          saverDef.getSaveTensorName
        ).asInstanceOf[Output[String]]
        val modelCheckpointPath = absoluteSavePath.getFileSystem.getPath(
          session.run(
            feeds = Map(filenameTensor -> checkpointFile.toString.toTensor),
            fetches = saveTensor).scalar)
        if (writeCheckpointState) {
          maybeDeleteOldCheckpoints(modelCheckpointPath, metaGraphSuffix)
          Saver.updateCheckpointStateFile(
            savePathParent, modelCheckpointPath, latestCheckpoints,
            checkpointStateFilename, saveRelativePaths)
        }
        Some(modelCheckpointPath)
      } catch {
        case exception: Exception =>
          if (!Files.isDirectory(savePathParent)) {
            throw new IllegalArgumentException(
              s"The parent directory of '$absoluteSavePath' does not exist, " +
                  s"preventing the saver from running.")
          } else {
            throw exception
          }
      }
    }

    // Save graph meta information.
    if (writeMetaGraph) {
      val metaGraphDef = session.graph.toMetaGraphDef(saverDef = saverDef, clearDevices = false)
      val metaGraphFilename = Saver.metaGraphFilename(checkpointFile, metaGraphSuffix)
      Proto.write(metaGraphFilename.getParent, metaGraphFilename.getFileName.toString, metaGraphDef)
    }

    Saver.logger.info(s"Saved parameters to '$checkpointFile'.")
    modelCheckpointPath
  }

  /** Restores previously saved saveables.
    *
    * This method runs the ops responding for restoring variables. It requires a session in which the saver's graph was
    * launched. The variables to restore do not have to have been initialized, as restoring is itself a way to
    * initialize variables.
    *
    * The `savePath` argument is typically a value previously returned from a [[save]] call, or a call to
    * [[Saver.latestCheckpoint]].
    *
    * @param  session  Session to use for restoring the variables.
    * @param  savePath Path to the checkpoint filename. If the saver is `sharded`, this is the prefix of the sharded
    *                  checkpoint filenames.
    */
  def restore(session: Session, savePath: Path): Unit = {
    Saver.logger.info(s"Restoring parameters from '$savePath'.")
    val filenameTensor = session.graph.getOutputByName(
      saverDef.getFilenameTensorName
    ).asInstanceOf[Output[String]]
    val restoreOp = session.graph.getOpByName(saverDef.getRestoreOpName)
    session.run(
      feeds = Map(filenameTensor -> savePath.toString.toTensor),
      targets = Set(restoreOp))
  }

  /** Returns the sequence of the latest and not-yet-deleted checkpoint filenames, sorted from oldest to newest. You can
    * pass any of the returned values to `restore`. */
  def latestCheckpoints: Seq[Path] = {
    lastCheckpoints.map(_._1)
  }

  /** Recovers the internal saver state (holding the last checkpoints) after a crash.
    *
    * This method searches for the checkpoints pointed to by `checkpoints` (which can also be glob patterns). If the
    * files exist, the method uses their last modification time as the checkpoint stamp.
    *
    * @param  checkpoints Sequence of checkpoint filenames (can also be glob patterns).
    */
  def recoverLastCheckpoints(checkpoints: Seq[Path]): Unit = {
    val times = Saver.checkpointTimes(
      checkpointPrefixes = checkpoints,
      unit = TimeUnit.SECONDS,
      followSymbolicLinks = true)
    lastCheckpoints = mutable.Queue(checkpoints.zip(times).sortBy(_._2): _*)
  }

  /** Deletes old checkpoints, if necessary.
    *
    * We always keep the last `maxToKeep` checkpoints. If `keepCheckpointEveryNHours` was specified, we keep an
    * additional checkpoint every `N` hours. For example, if `N` is `0.5`, an additional checkpoint is kept for every
    * `0.5` hours of training. If `N` is `10`, an additional checkpoint is kept for every `10` hours of training.
    *
    * @param  checkpointFile  Filename (including path) of checkpoint file to save (can also be a glob pattern).
    * @param  metaGraphSuffix Meta graph filename suffix.
    * @throws IllegalArgumentException If an unsupported checkpoint format version is being used by this saver.
    */
  @throws[IllegalArgumentException]
  private def maybeDeleteOldCheckpoints(
      checkpointFile: Path,
      metaGraphSuffix: String = "meta"
  ): Unit = {
    if (saverDef.getMaxToKeep > 0) {
      // Remove first from list if the same name was used before.
      lastCheckpoints = lastCheckpoints.filter(_._1 != checkpointFile)
      // Append new path to list.
      lastCheckpoints.enqueue((checkpointFile, System.currentTimeMillis() / 60000))
      // If more checkpoints than the maximum allowed to keep exist, then remove the oldest.
      if (lastCheckpoints.length > saverDef.getMaxToKeep) {
        val checkpoint = lastCheckpoints.dequeue()
        if (checkpoint._2 > nextCheckpointTime) {
          // Do not delete the file if we keep checkpoints every N hours is set and we have reached N hours of training.
          nextCheckpointTime += saverDef.getKeepCheckpointEveryNHours * 3600
        } else {
          // Otherwise delete the files.
          FileIO.deleteMatchingPaths(Saver.metaGraphFilename(checkpoint._1, metaGraphSuffix))
          writerVersion match {
            case Saver.V1 =>
              // Deprecated checkpoint format using an exact match on the checkpoint filename.
              FileIO.deleteMatchingPaths(checkpoint._1)
            case Saver.V2 =>
              // The V2 format has a metadata file along with some data files.
              val filename = checkpoint._1.getFileName
              FileIO.deleteMatchingPaths(checkpoint._1.resolveSibling(s"$filename.index"))
              FileIO.deleteMatchingPaths(checkpoint._1.resolveSibling(s"$filename.data-?????-of-?????"))
          }
        }
      }
    }
  }

  override def toProto: SaverDef = toProto(null)

  /** Alias for `toSaverDef`. */
  def toProto(exportScope: String = null): SaverDef = {
    toSaverDef(exportScope)
  }

  /** Constructs and returns a [[SaverDef]] object that represents this saver.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[SaverDef]].
    */
  def toSaverDef(exportScope: String = null): SaverDef = {
    if (exportScope == null || exportScope == "") {
      saverDef
    } else if (!saverDef.getFilenameTensorName.startsWith(exportScope) ||
        !saverDef.getSaveTensorName.startsWith(exportScope) ||
        !saverDef.getRestoreOpName.startsWith(exportScope)) {
      null
    } else {
      val saverDefBuilder = SaverDef.newBuilder(saverDef)
      saverDefBuilder.setFilenameTensorName(Op.stripNameScope(exportScope, saverDefBuilder.getFilenameTensorName))
      saverDefBuilder.setSaveTensorName(Op.stripNameScope(exportScope, saverDefBuilder.getSaveTensorName))
      saverDefBuilder.setRestoreOpName(Op.stripNameScope(exportScope, saverDefBuilder.getRestoreOpName))
      saverDefBuilder.build()
    }
  }
}

/** Contains helper functions for managing savers. */
object Saver {
  private[Saver] val logger = Logger(LoggerFactory.getLogger("Variables / Saver"))

  /** Adds save/restore nodes to the graph and creates and returns a [[SaverDef]] proto.
    *
    * @param  saveables                 Objects that need to be saved and loaded. If `null`, then all saveable objects
    *                                   are obtained and used from the graph in the current op creation context.
    * @param  reshape                   Boolean value which, if `true`, allows restoring parameters from a checkpoint
    *                                   where the parameters have a different shape. This is only needed when you try to
    *                                   restore from a Dist-Belief checkpoint, and only some times.
    * @param  sharded                   Boolean value which, if `true`, will shard the checkpoints, one per device that
    *                                   is used by the provided `saveables`.
    * @param  maxToKeep                 Maximum number of checkpoints to keep. As new checkpoints are created, old ones
    *                                   are deleted. If `0`, no checkpoints are deleted from the filesystem but only the
    *                                   last one is kept in the `checkpoint` file. Presently the number is only roughly
    *                                   enforced. For example, in the case of restarts more than `maxToKeep` checkpoints
    *                                   may be kept.
    * @param  keepCheckpointEveryNHours Denotes how often checkpoints should be saved, in hour units. Defaults to
    *                                   10,000 hours.
    * @param  restoreSequentially       Boolean value which, if `true`, causes the restoration of different variables to
    *                                   happen sequentially within each device.
    * @param  filename                  Filename used for the saveable objects saving and loading.
    * @param  builder                   Saver builder to use. Defaults to [[DefaultSaverDefBuilder]].
    * @param  allowEmpty                Boolean value indicating whether to allow for an empty saver (i.e., one with no
    *                                   saveable objects that is effectively a no-op). Defaults to `false`.
    * @param  writerVersion             Checkpoint format version to use.
    * @param  saveRelativePaths         Boolean value which, if `true`, forces the saver to write relative paths to the
    *                                   checkpoint state file. This is needed if the user wants to copy the checkpoint
    *                                   directory and restore from the copied directory.
    * @param  padGlobalStep             Boolean value which, if `true`, forces the saver to pad the global step number
    *                                   in the checkpoint file paths to some fixed width (`8` by default). This is
    *                                   turned off by default.
    * @param  name                      Optional name to use as a prefix when adding ops.
    * @return Created [[SaverDef]] objects.
    * @throws IllegalArgumentException If no saveables are provided or obtained from the current graph and `allowEmpty`
    *                                  is set to `false`.
    */
  @throws[IllegalArgumentException]
  private[api] def apply(
      saveables: Set[Saveable] = null,
      reshape: Boolean = false,
      sharded: Boolean = false,
      maxToKeep: Int = 5,
      keepCheckpointEveryNHours: Float = 10000.0f,
      restoreSequentially: Boolean = false,
      filename: String = "model",
      builder: SaverDefBuilder = DefaultSaverDefBuilder,
      allowEmpty: Boolean = false,
      writerVersion: WriterVersion = V2,
      saveRelativePaths: Boolean = false,
      padGlobalStep: Boolean = false,
      name: String = "Saver"
  ): Saver = {
    val collectedSaveables: Set[Saveable] = {
      if (saveables == null) {
        // TODO: [VARIABLES] Use a better default for this.
        Op.currentGraph.getCollection(Graph.Keys.GLOBAL_VARIABLES)
            .map(new Saveable.VariableSaveable(_))
      } else {
        saveables
      }
    }
    if (collectedSaveables.isEmpty && !allowEmpty)
      throw new IllegalArgumentException("No saveables were provided and 'allowEmpty' is set to 'false'.")
    val saverDef = builder.build(
      saveables = collectedSaveables,
      reshape = reshape,
      sharded = sharded,
      maxToKeep = maxToKeep,
      keepCheckpointEveryNHours = keepCheckpointEveryNHours,
      restoreSequentially = restoreSequentially,
      filename = filename,
      name = name)
    new Saver(saverDef, saveRelativePaths = saveRelativePaths, padGlobalStep = padGlobalStep)
  }

  /** Creates a saver from the provided [[SaverDef]] object.
    *
    * @param  saverDef          Serialized saver object.
    * @param  importScope       Optional prefix that will be prepended to all op names in the saver that is being loaded
    *                           from the provided [[SaverDef]].
    * @param  saveRelativePaths Boolean value which, if `true`, forces the saver to write relative paths to the
    *                           checkpoint state file. This is needed if the user wants to copy the checkpoint directory
    *                           and restore from the copied directory.
    * @param  padGlobalStep     Boolean value which, if `true`, forces the saver to pad the global step number in the
    *                           checkpoint file paths to some fixed width (`8` by default). This is turned off by
    *                           default.
    * @return Constructed [[Saver]].
    */
  def fromSaverDef(
      saverDef: SaverDef,
      importScope: String = null,
      saveRelativePaths: Boolean = false,
      padGlobalStep: Boolean = true
  ): Saver = {
    val newSaverDef = {
      if (importScope != null && importScope != "") {
        val saverDefBuilder = saverDef.toBuilder
        saverDefBuilder.setFilenameTensorName(Op.prependNameScope(importScope, saverDefBuilder.getFilenameTensorName))
        saverDefBuilder.setSaveTensorName(Op.prependNameScope(importScope, saverDefBuilder.getSaveTensorName))
        saverDefBuilder.setRestoreOpName(Op.prependNameScope(importScope, saverDefBuilder.getRestoreOpName))
        saverDefBuilder.build()
      } else {
        saverDef
      }
    }
    new Saver(saverDef = newSaverDef, saveRelativePaths = saveRelativePaths, padGlobalStep = padGlobalStep)
  }

  /** Creates a saver from the provided [[SaverDef]] object.
    *
    * @param  saverDef          Serialized saver object.
    * @param  importScope       Optional prefix that will be prepended to all op names in the saver that is being loaded
    *                           from the provided [[SaverDef]].
    * @param  saveRelativePaths Boolean value which, if `true`, forces the saver to write relative paths to the
    *                           checkpoint state file. This is needed if the user wants to copy the checkpoint directory
    *                           and restore from the copied directory.
    * @param  padGlobalStep     Boolean value which, if `true`, forces the saver to pad the global step number in the
    *                           checkpoint file paths to some fixed width (`8` by default). This is turned off by
    *                           default.
    * @return Constructed [[Saver]].
    */
  def fromProto(
      saverDef: SaverDef,
      importScope: String = null,
      saveRelativePaths: Boolean = false,
      padGlobalStep: Boolean = true
  ): Saver = {
    fromSaverDef(
      saverDef = saverDef,
      importScope = importScope,
      saveRelativePaths = saveRelativePaths,
      padGlobalStep = padGlobalStep)
  }

  /** Imports a serialized representation of a graph and its meta-information into the current graph and returns the
    * saver that is contained in that meta information.
    *
    * This function takes a [[MetaGraphDef]] protocol buffer as input and it adds all the nodes from its `graph_def`
    * field to the current graph. It also recreates the desired collections stored in that protocol buffer.
    *
    * In combination with [[Graph.toMetaGraphDef]], this function can be used to:
    *   - Serialize a graph along with other objects stored in its collections, into a [[MetaGraphDef]].
    *   - Restart training from saved graphs and checkpoints.
    *   - Run inference from saved graphs and checkpoints.
    *
    * @param  metaGraphDef                Serialized representation of the graph and its meta-information, that will be
    *                                     imported into the current graph.
    * @param  importScope                 Optional prefix that will be prepended to all node names in the graph that is
    *                                     being imported to this graph.
    * @param  saveRelativePaths           Boolean value which, if `true`, forces the saver to write relative paths to
    *                                     the checkpoint state file. This is needed if the user wants to copy the
    *                                     checkpoint directory and restore from the copied directory.
    * @param  padGlobalStep               Boolean value which, if `true`, forces the saver to pad the global step number
    *                                     in the checkpoint file paths to some fixed width (`8` by default). This is
    *                                     turned off by default.
    * @param  inputsMap                   Optional inputs mapping. For each
    *                                     `(source_op_name, source_op_output_index) -> destination_op_output` mapping,
    *                                     the importer will  set any imported nodes with input named
    *                                     `source_op_name:source_op_output_index` to have that input replaced with
    *                                     `destination_op_output`. `source_op_name` refers to a node in the graph to be
    *                                     imported, whereas `destination_op_output` references a node already existing
    *                                     in this graph.
    * @param  controlDependenciesMap      Optional control dependencies mapping. For each
    *                                     `source_op_name -> destination_op` mapping, the importer will set any imported
    *                                     ops with control input named `source_op_name` to have that input replaced with
    *                                     `destination_op`. `source_op_name` refers to a node in the graph to be
    *                                     imported, whereas `destination_op` references an op already existing in this
    *                                     graph.
    * @param  controlDependencies         Optional control dependencies set. The importer will make sure that the
    *                                     imported graph has a control dependency on all ops in this set. All such ops,
    *                                     should therefore be defined in this graph.
    * @param  clearDevices                Boolean value indicating whether to clear the device information from the
    *                                     returned node definition.
    * @param  unboundInputsCollectionKey  Collection key for looking up unbound inputs.
    * @param  restoreCollectionsPredicate Function that takes as input a graph collection key and returns a boolean
    *                                     value indicating whether or not to load that collection. Note that the
    *                                     collection specified by `unboundInputsCollectionKey` is never loaded.
    *                                     Defaults to a function that returns `true` for all inputs.
    * @return Constructed saver.
    */
  def fromMetaGraphDef(
      metaGraphDef: MetaGraphDef,
      importScope: String = null,
      saveRelativePaths: Boolean = false,
      padGlobalStep: Boolean = true,
      inputsMap: Map[(String, Int), Output[Any]] = Map.empty,
      controlDependenciesMap: Map[String, UntypedOp] = Map.empty,
      controlDependencies: Set[UntypedOp] = Set.empty,
      clearDevices: Boolean = false,
      unboundInputsCollectionKey: Graph.Key[String] = Graph.Keys.UNBOUND_INPUTS,
      restoreCollectionsPredicate: Graph.Key[_] => Boolean = _ => true
  ): Saver = {
    Op.currentGraph.importMetaGraphDef(
      metaGraphDef = metaGraphDef,
      importScope = importScope,
      inputsMap = inputsMap,
      controlDependenciesMap = controlDependenciesMap,
      controlDependencies = controlDependencies,
      clearDevices = clearDevices,
      unboundInputsCollectionKey = unboundInputsCollectionKey,
      restoreCollectionsPredicate = restoreCollectionsPredicate)
    val saverDef = metaGraphDef.getSaverDef
    if (saverDef == null) {
      Saver()
    } else {
      fromSaverDef(
        saverDef = saverDef,
        importScope = importScope,
        saveRelativePaths = saveRelativePaths,
        padGlobalStep = padGlobalStep)
    }
  }

  /** Creates a new device string based on `device` but using `/CPU:0`. */
  private[variables] def setCPU0(device: String): String = {
    DeviceSpecification.fromString(device)
        .copy(deviceType = "CPU", deviceIndex = 0)
        .toString
  }

  /** Finds and returns the filename of the latest saved checkpoint file.
    *
    * @param  directory           Directory used for saving.
    * @param  checkpointStateFile Optional name for the protocol buffer file that will contain / contains the list of
    *                             the most recent checkpoint filenames. That file, kept in the same directory as the
    *                             checkpoint files, and it is automatically managed by the saver to keep track of recent
    *                             checkpoints.
    * @return Full path to the latest checkpoint, or `None`, if no checkpoint was found.
    */
  def latestCheckpoint(
      directory: Path,
      checkpointStateFile: String = "checkpoint"
  ): Option[Path] = {
    // Pick the latest checkpoint based on the checkpoint state.
    val checkpointState = loadCheckpointState(directory, checkpointStateFile)
    if (checkpointState.isDefined && checkpointState.get.getModelCheckpointPath != null) {
      // Look for either a V2 path or a V1 path, with priority for V2.
      val fileSystem = directory.getFileSystem
      val modelCheckpointPath = fileSystem.getPath(checkpointState.get.getModelCheckpointPath)
      val v2Path = prefixToCheckpointPath(modelCheckpointPath, CheckpointFormatVersion.V2)
      val v1Path = prefixToCheckpointPath(modelCheckpointPath, CheckpointFormatVersion.V1)
      if (FileIO.getMatchingPaths(v2Path).nonEmpty || FileIO.getMatchingPaths(v1Path).nonEmpty)
        Some(modelCheckpointPath)
      else
        throw new IllegalArgumentException(s"Could not match any files for checkpoint '$modelCheckpointPath'.")
    } else {
      None
    }
  }

  /** Checks the provided [[SaverDef]] for validity. */
  @throws[IllegalArgumentException]
  private def checkSaverDef(saverDef: SaverDef): Unit = {
    if (saverDef.getSaveTensorName == null)
      throw new IllegalArgumentException(s"The 'saverDef' must specify a save tensor name: $saverDef.")
    if (saverDef.getRestoreOpName == null)
      throw new IllegalArgumentException(s"The 'saverDef' must specify a restore op name: $saverDef.")
  }

  /** Returns the path to a file for storing the checkpoint state.
    *
    * @param  directory Directory used for saving and restoring checkpoints.
    * @param  filename  File in `directory` that is used to store the checkpoint state.
    * @return Path of the file that contains the checkpoint state.
    */
  private def checkpointPath(
      directory: Path,
      filename: String = "checkpoint"
  ): Path = {
    directory.resolve(filename)
  }

  /** Returns a boolean value indicating whether a V1 or V2 checkpoint exists with the specified prefix.
    *
    * This is the recommended way to check if a checkpoint exists, since it takes into account the naming difference
    * between the V1 and the V2 formats.
    *
    * @param  checkpointPrefix Sequence of checkpoint paths. Typically the results of `Saver.save` or those of
    *                          `Saver.latestCheckpoint`, regardless of sharded/non-sharded or the checkpoint format
    *                          version (i.e., V1/V2).
    * @return `true` if and only if, a checkpoint file referred to by `checkpointPrefix` exists.
    */
  private def checkpointExists(checkpointPrefix: Path): Boolean = {
    // Try V2's metadata file first.
    val pathPattern = prefixToCheckpointPath(checkpointPrefix, CheckpointFormatVersion.V2)
    FileIO.getMatchingPaths(pathPattern).nonEmpty ||
        FileIO.getMatchingPaths(checkpointPrefix).nonEmpty
  }

  /** Generates a checkpoint state.
    *
    * @param  directory               Checkpoints directory.
    * @param  modelCheckpointPath     Checkpoint file path.
    * @param  allModelCheckpointPaths Paths to all not-yet-deleted checkpoints, sorted from oldest to newest. If this is
    *                                 a non-empty list, then the last element must be equal to `checkpointPath`. These
    *                                 paths are also saved in the generated checkpoint state.
    * @return Checkpoint state with `checkpointPath` and `allCheckpointPaths` updated to either absolute paths or
    *         relative paths to `directory`.
    */
  private def checkpointState(
      directory: Path,
      modelCheckpointPath: Path,
      allModelCheckpointPaths: Seq[Path] = Seq.empty
  ): CheckpointState = {
    var checkpointPath = modelCheckpointPath
    var allCheckpointPaths = {
      if (allModelCheckpointPaths.nonEmpty && allModelCheckpointPaths.last != checkpointPath)
        allModelCheckpointPaths :+ checkpointPath
      else
        allModelCheckpointPaths
    }
    if (!directory.isAbsolute) {
      // Relative file paths need to be rewritten to be relative to 'directory'.
      if (!checkpointPath.isAbsolute)
        checkpointPath = directory.relativize(checkpointPath)
      allCheckpointPaths = allCheckpointPaths.map(path => {
        if (!path.isAbsolute)
          directory.relativize(path)
        else
          path
      })
    }
    CheckpointState.newBuilder()
        .setModelCheckpointPath(checkpointPath.toString)
        .addAllAllModelCheckpointPaths(allCheckpointPaths.map(_.toString).asJava)
        .build()
  }

  /** Updates the content of a checkpoint file.
    *
    * @param  directory               Checkpoints directory.
    * @param  modelCheckpointPath     Checkpoint file path.
    * @param  allModelCheckpointPaths Paths to all not-yet-deleted checkpoints, sorted from oldest to newest. If this is
    *                                 a non-empty list, then the last element must be equal to `checkpointPath`. These
    *                                 paths are also saved in the generated checkpoint state.
    * @param  checkpointStateFilename Checkpoint state file name.
    * @param  saveRelativePaths       Boolean value indicating whether to write relative paths to the checkpoint state
    *                                 file.
    */
  private def updateCheckpointStateFile(
      directory: Path,
      modelCheckpointPath: Path,
      allModelCheckpointPaths: Seq[Path] = Seq.empty,
      checkpointStateFilename: String = "checkpoint",
      saveRelativePaths: Boolean = false
  ): Unit = {
    // Writes the "checkpoint" file for the coordinator for later restoration.
    val coordinatorCheckpointStateFilename = directory.resolve(checkpointStateFilename)
    val state = {
      if (saveRelativePaths) {
        val modelCheckpointRelativePath = {
          if (modelCheckpointPath.isAbsolute)
            directory.relativize(modelCheckpointPath)
          else
            modelCheckpointPath
        }
        val allModelCheckpointRelativePaths = allModelCheckpointPaths.map(path => {
          if (path.isAbsolute)
            directory.relativize(path)
          else
            path
        })
        checkpointState(
          directory,
          modelCheckpointRelativePath,
          allModelCheckpointRelativePaths)
      } else {
        checkpointState(
          directory,
          modelCheckpointPath,
          allModelCheckpointPaths)
      }
    }

    if (coordinatorCheckpointStateFilename.toString == state.getModelCheckpointPath)
      throw new IllegalArgumentException(
        s"Save path '$modelCheckpointPath' conflicts with the path used for the checkpoint state. " +
            "Please use a different save path.")

    // Preventing potential read/write race condition by atomically writing to a file.
    FileIO.writeStringToFileAtomic(
      coordinatorCheckpointStateFilename,
      TextFormat.printToString(state))
  }

  /** Loads the checkpoint state stored in the file named `checkpointStateFilename`, in the specified directory.
    *
    * @param  directory           Checkpoints directory.
    * @param  checkpointStateFile Checkpoint state file name.
    * @return Loaded checkpoint state, or `None` if the state could not be loaded.
    * @throws IllegalArgumentException If the checkpoint state does not have its model checkpoint path set.
    */
  @throws[IllegalArgumentException]
  private[api] def loadCheckpointState(
      directory: Path,
      checkpointStateFile: String = "checkpoint"
  ): Option[CheckpointState] = {
    val coordinatorCheckpointStateFilename = directory.resolve(checkpointStateFile)
    // Check that the file exists before opening it to avoid many lines of errors from colossus in the logs.
    if (Files.exists(coordinatorCheckpointStateFilename)) {
      try {
        val loadedLines = Files.readAllLines(coordinatorCheckpointStateFilename)
        val checkpointStateBuilder = CheckpointState.newBuilder()
        TextFormat.merge(loadedLines.asScala.mkString("\n"), checkpointStateBuilder)
        if (checkpointStateBuilder.getModelCheckpointPath == null)
          throw new IllegalArgumentException(s"Invalid checkpoint state loaded from: $directory.")
        // For relative paths, we prepend the directory.
        val modelCheckpointPath = checkpointStateBuilder.getModelCheckpointPath
        if (!directory.getFileSystem.getPath(modelCheckpointPath).isAbsolute) {
          checkpointStateBuilder.setModelCheckpointPath(
            directory.resolve(modelCheckpointPath).toAbsolutePath.toString)
        }
        (0 until checkpointStateBuilder.getAllModelCheckpointPathsCount).foreach(i => {
          val path = checkpointStateBuilder.getAllModelCheckpointPaths(i)
          if (!directory.getFileSystem.getPath(path).isAbsolute) {
            checkpointStateBuilder.setAllModelCheckpointPaths(
              i, directory.resolve(path).toAbsolutePath.toString)
          }
        })
        Some(checkpointStateBuilder.build())
      } catch {
        case t: Throwable =>
          logger.warn(s"Exception thrown while loading the checkpoint state.", t)
          logger.warn(s"Checkpoint '$coordinatorCheckpointStateFilename' ignored.")
          None
      }
    } else {
      None
    }
  }

  /** Returns the times (modification timestamps) of the checkpoints.
    *
    * This function for matching files for the checkpoints pointed to by `checkpointPrefixes`. If the files exist, it
    * collect their last modification times. Both V2 and V1 checkpoints are considered, in that order.
    *
    * Note: This is the recommended way to get the last modification times of checkpoint files, because it takes into
    * account the naming difference between the V1 and the V2 formats.
    *
    * @param  checkpointPrefixes  Sequence of checkpoint paths. Typically the results of `Saver.save` or those of
    *                             `Saver.latestCheckpoint`, regardless of sharded/non-sharded or the checkpoint format
    *                             version (i.e., V1/V2).
    * @param  unit                Time unit in which to return the last modified time. Defaults to [[TimeUnit.SECONDS]].
    * @param  followSymbolicLinks Boolean value indicating whether or not to follow symbolic links. By default, symbolic
    *                             links are followed and the file attribute of the final target of the link is read. If
    *                             `followSymbolicLinks` is set to `false`, then symbolic links are not followed.
    * @return Sequence of last modification times for each one of the provided checkpoints, measured in the units
    *         specified by `unit`.
    */
  private def checkpointTimes(
      checkpointPrefixes: Seq[Path],
      unit: TimeUnit = TimeUnit.SECONDS,
      followSymbolicLinks: Boolean = true
  ): Seq[Long] = {
    def maybeGetTime(pattern: Path): Long = {
      val paths = FileIO.getMatchingPaths(pattern)
      if (paths.nonEmpty) {
        FileIO.getLastModifiedTime(paths.head, unit, followSymbolicLinks)
      } else {
        -1
      }
    }

    val times = checkpointPrefixes.map(prefix => {
      // Try V2's metadata file first.
      val pathPattern = prefixToCheckpointPath(prefix, CheckpointFormatVersion.V2)
      val time = maybeGetTime(pathPattern)
      if (time > 0) {
        time
      } else {
        // Otherwise, try V1, where the prefix is the complete path name.
        maybeGetTime(prefix)
      }
    })

    val badCheckpointPrefix = times.indexWhere(_ < 0)
    if (badCheckpointPrefix != -1) {
      val prefix = checkpointPrefixes(badCheckpointPrefix)
      throw new IllegalArgumentException(
        s"Could not obtain the time for checkpoint with prefix: $prefix.")
    }

    times
  }

  /** Returns the path to a checkpoint file, given the checkpoint prefix.
    *
    * For V1 checkpoints, this function simply returns the prefix itself (the data file). For V2, it returns the path
    * name to the index file.
    *
    * @param  prefix                  Checkpoint prefix.
    * @param  checkpointFormatVersion Checkpoint format version.
    * @return Path to the checkpoint file.
    * @throws IllegalArgumentException If an unsupported checkpoint format version is being used.
    */
  @throws[IllegalArgumentException]
  private def prefixToCheckpointPath(
      prefix: Path,
      checkpointFormatVersion: CheckpointFormatVersion
  ): Path = {
    checkpointFormatVersion match {
      case SaverDef.CheckpointFormatVersion.V1 =>
        // Just the data file.
        prefix
      case SaverDef.CheckpointFormatVersion.V2 =>
        // The index file identifies a checkpoint.
        prefix.resolveSibling(s"${prefix.getFileName}.index")
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported checkpoint format version '$checkpointFormatVersion'.")
    }
  }

  private val SHARDED_CHECKPOINT_FILENAME_REGEX = "-[\\d\\?]+-of-\\d+$".r

  /** Returns the meta graph filename.
    *
    * This function takes into account that checkpoint files may be sharded and returns the appropriate filename for
    * storing the meta graph information.
    *
    * @param  checkpointFile  Checkpoint filename.
    * @param  metaGraphSuffix Meta graph filename suffix.
    * @return Meta graph filename.
    */
  private def metaGraphFilename(
      checkpointFile: Path,
      metaGraphSuffix: String = "meta"
  ): Path = {
    // If the checkpoint filename is sharded, it could be of format
    // "model.ckpt-<step#>-?????-of-<shard#>". For example,
    // "model.ckpt-123456-?????-of-00005", or "model.ckpt-123456-00001-of-00002".
    val filename = checkpointFile.getFileName.toString
    val baseName = SHARDED_CHECKPOINT_FILENAME_REGEX.pattern.matcher(filename)
        .replaceFirst("")
    checkpointFile.resolveSibling(s"$baseName.$metaGraphSuffix")
  }

  /** Checkpoint format version used by a saver.
    *
    * This is simply a wrapper for a [[SaverDef.CheckpointFormatVersion]].
    */
  sealed trait WriterVersion {
    protected val checkpointFormatVersion: CheckpointFormatVersion
  }

  /** Wrapper for [[SaverDef.CheckpointFormatVersion.V1]]. */
  object V1 extends WriterVersion {
    override protected val checkpointFormatVersion: CheckpointFormatVersion = CheckpointFormatVersion.V1
  }

  /** Wrapper for [[SaverDef.CheckpointFormatVersion.V2]]. */
  object V2 extends WriterVersion {
    override protected val checkpointFormatVersion: CheckpointFormatVersion = CheckpointFormatVersion.V2
  }
}

/** A saver builder is used to build [[SaverDef]] objects.
  *
  * Most users shall never have to worry about dealing with saver builders. The [[Saver]] constructor uses
  * [[DefaultSaverDefBuilder]] by default, which should be fine for most applications. */
//noinspection ScalaDeprecation
trait SaverDefBuilder {
  private val checkpointFormatVersion: CheckpointFormatVersion = {
    SaverDef.CheckpointFormatVersion.V2
  }

  /** Creates an op that saves the provided sequence of saveables into a file.
    *
    * Note that this method is intended to be overridden by subclasses that want to generate different types of ops.
    *
    * @param  prefix    String tensor containing a single element. That element corresponds to the prefix of a V2
    *                   checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that is the V1
    *                   checkpoint format is being used (which is deprecated), then this prefix is interpreted as a
    *                   filename instead.
    * @param  saveables Sequence of saveable objects that the created op will save.
    * @param  name      Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If an unsupported checkpoint format version is being used.
    */
  @throws[IllegalArgumentException]
  protected def save(
      prefix: Output[String],
      saveables: Set[Saveable],
      name: String = "Save"
  ): Op[Seq[Output[Any]], Unit] = {
    if (saveables.nonEmpty) {
      val (tensorNames, tensors, slices) =
        saveables.flatMap(_.saveSpecifications)
            .map(s => (s.name, s.value(), s.saveSliceSpecification))
            .toSeq.unzip3[String, Output[Any], String]
      val saveOp = checkpointFormatVersion match {
        case SaverDef.CheckpointFormatVersion.V1 =>
          SaverDefBuilder.saveSlicesOp(prefix, tensorNames, tensors, slices, name): @silent
        case SaverDef.CheckpointFormatVersion.V2 =>
          SaverDefBuilder.saveV2Op(prefix, tensorNames, tensors, slices, name)
        case _ => throw new IllegalArgumentException(
          s"Unsupported checkpoint format version '$checkpointFormatVersion'.")
      }
      saveOp.asInstanceOf[Op[Seq[Output[Any]], Unit]]
    } else {
      ControlFlow.noOp(name).asInstanceOf[Op[Seq[Output[Any]], Unit]]
    }
  }

  /** Creates an op that restores the tensors that constitute `saveable`.
    *
    * Note that this method is intended to be overridden by subclasses that want to generate different types of ops.
    *
    * @param  prefix   Tensor containing a single element. That element corresponds to the prefix of a V2
    *                  checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that is the V1
    *                  checkpoint format is being used (which is deprecated), then this prefix is interpreted as a
    *                  filename instead.
    * @param  saveable Saveable object that the created op will restore.
    * @param  name     Name for the created op.
    * @return Created op outputs (restored tensors that constitute `saveable`).
    */
  protected def restore[T](
      prefix: Output[String],
      saveable: Saveable,
      name: String = "Restore"
  ): Seq[Output[T]] = {
    val (tensorNames, slices, dataTypes) =
      saveable.saveSpecifications
          .map(s => (
              s.name,
              s.saveSliceSpecification,
              s.value().dataType.asInstanceOf[DataType[T]]))
          .unzip3[String, String, DataType[T]]
    SaverDefBuilder.restoreV2Op[T](prefix, tensorNames, slices, dataTypes, name)
  }

  /** Adds ops to save objects that are on the same shard and returns a tensor containing the filename used for the save
    * operation.
    *
    * @param  prefix    Tensor containing a single element. That element corresponds to the prefix of a V2
    *                   checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that is the V1
    *                   checkpoint format is being used (which is deprecated), then this prefix is interpreted as a
    *                   filename instead.
    * @param  saveables Sequence of saveable objects that the created op will save.
    * @param  name      Name for the created op.
    * @return Tensor containing the filename used for the save operation.
    */
  protected def addSaveOps(
      prefix: Output[String],
      saveables: Set[Saveable],
      name: String = "Save"
  ): Output[String] = {
    val saveOp = save(prefix, saveables, name)
    ControlFlow.withControlDependencies(Set(saveOp), prefix)
  }

  /** Adds ops to save sharded (per device) objects.
    *
    * Note that the sharded save procedure for the V2 checkpoint format is different than that for V1. There is a
    * special "merge" step that merges the small metadata produced from each device.
    *
    * @param  prefix            Tensor containing a single element. That element corresponds to the prefix of a
    *                           V2 checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that is
    *                           the V1 checkpoint format is being used (which is deprecated), then this prefix is
    *                           interpreted as a filename instead.
    * @param  saveablesByDevice Sequence of device-saveables pairs, sorted by ascending device name. This is the result
    *                           of the [[SaverDefBuilder.groupByDevice]] method.
    * @return Tensor containing the filename used for the save operation.
    */
  protected def addShardedSaveOps(
      prefix: Output[String],
      saveablesByDevice: Seq[(String, Set[Saveable])]
  ): Output[String] = {
    checkpointFormatVersion match {
      case SaverDef.CheckpointFormatVersion.V1 =>
        val numberOfShards = Tensor(saveablesByDevice.length).toOutput
        val shardedSaves = saveablesByDevice.zipWithIndex.map {
          case ((device, saveables), shard) =>
            Op.createWith(device = Saver.setCPU0(device)) {
              addSaveOps(SaverDefBuilder.shardedFilenameOp(prefix, shard, numberOfShards), saveables)
            }
        }
        // Return the sharded name for the save path.
        Op.createWith(controlDependencies = shardedSaves.map(_.op).toSet) {
          SaverDefBuilder.shardedFilenameSpecificationOp(prefix, numberOfShards)
        }
      case SaverDef.CheckpointFormatVersion.V2 =>
        // Suffix for any well-formed 'prefix', when sharded.
        val _SHARDED_SUFFIX = s"_temp_${UUID.randomUUID().toString}/part": Output[String]
        // Transformations:
        //   - Users pass in "save_path_" in the save and restore methods. E.g., "myckpt".
        //   - 'prefix' gets fed <save_path><_SHARDED_SUFFIX>.
        //
        // For example:
        //   During runtime, a temporary directory is first created, which contains files:
        //     <train dir>/myckpt_temp/part-?????-of-?????{.index, .data-00000-of-00001}
        //
        //   Before the save operation finishes, these files will be (hopefully, automatically) renamed to:
        //     <train dir>/myckpt{.index, .data-?????-of-?????}
        //
        // Users only need to interact with the user-specified prefix, which is "<train dir>/myckpt" in this case. The
        // save and restore operations work with the prefix directly, instead of any physical pathname.
        //
        // On failure and  subsequent restore, an outdated and orphaned temporary directory can be safely removed.
        val temporaryCheckpointPrefix = Text.stringJoin(Seq(prefix, _SHARDED_SUFFIX))
        val (shardedPrefixes, shardedSaves) = saveablesByDevice.zipWithIndex.map {
          case ((device, saveables), shard) =>
            Op.createWith(device = Saver.setCPU0(device)) {
              val prefix = SaverDefBuilder.shardedFilenameOp(
                temporaryCheckpointPrefix, shard, saveablesByDevice.length)
              (prefix, addSaveOps(prefix, saveables))
            }
        }.unzip
        // Co-locates the merge step with the last device.
        Op.createWith(
          controlDependencies = shardedSaves.map(_.op).toSet,
          device = Saver.setCPU0(saveablesByDevice.last._1)
        ) {
          // The V2 format write path consists of a metadata merging step.
          // Once merged, we attempt to delete the temporary directory,
          // "<user-fed prefix>_temp".
          val concatenatedPrefixes = {
            if (shardedPrefixes.length > 1)
              Basic.stack(shardedPrefixes)
            else
              shardedPrefixes.head.reshape(Shape(1))
          }
          val mergeOp = SaverDefBuilder.mergeV2Checkpoints(
            concatenatedPrefixes, prefix, deleteOldDirectories = true)
          // Returns the prefix "<user-fed prefix>" only, without the sharded specification suffix.
          ControlFlow.withControlDependencies(Set(mergeOp), prefix)
        }
      case _ => throw new IllegalArgumentException(
        s"Unsupported checkpoint format version '$checkpointFormatVersion'.")
    }
  }

  /** Adds ops to restore objects that are on the same shard.
    *
    * @param  prefix              Tensor containing a single element. That element corresponds to the prefix of a
    *                             V2 checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that
    *                             is the V1 checkpoint format is being used (which is deprecated), then this prefix is
    *                             interpreted as a filename instead.
    * @param  saveables           Sequence of saveable objects that the created op will restore.
    * @param  reshape             Boolean value indicating whether to reshape loaded tensors to the shape of the
    *                             corresponding saveable object.
    * @param  restoreSequentially Boolean value indicating whether to restore variables objects within a shard.
    * @param  name                Name for the created op.
    * @return Created op.
    */
  protected def addRestoreOps(
      prefix: Output[String],
      saveables: Set[Saveable],
      reshape: Boolean,
      restoreSequentially: Boolean,
      name: String = "Restore"
  ): Op[Seq[Output[Any]], Unit] = {
    var restoreOps = Seq.empty[UntypedOp]
    saveables.foreach(saveable => {
      val restoreControlInputs = if (restoreSequentially) Set(restoreOps.last) else Set.empty[UntypedOp]
      // Load and optionally reshape on the CPU, as string tensors are not available on the GPU.
      // TODO: !!! [GPU] Re-enable restore on GPU when we can support annotating string tensors as "HostMemory" inputs.
      Op.createWith(
        controlDependencies = restoreControlInputs,
        device = Saver.setCPU0(saveable.device)
      ) {
        val shapes = {
          if (reshape) {
            // Compute the shapes and let the restore op decide if and how to do the reshape.
            saveable.saveSpecifications.map(s => {
              val sValue = s.value()
              if (s.value().shape.isFullyDefined)
                sValue.shape.toOutput
              else
                Basic.shape(sValue)(TF.fromDataType(sValue.dataType))
            })
          } else {
            null
          }
        }
        restoreOps :+= saveable.restore(restore(prefix, saveable, name), shapes)
      }
    })

    // Create a no-op that has control dependencies for all the updates.
    ControlFlow.group(restoreOps.toSet).asInstanceOf[Op[Seq[Output[Any]], Unit]]
  }

  /** Adds ops to restore sharded (per device) objects.
    *
    * @param  prefix              String tensor containing a single element. That element corresponds to the prefix of a
    *                             V2 checkpoint. For example, `"/fs/train/ckpt-<step>/tmp/worker<i>-<step>"`. Note that
    *                             is the V1 checkpoint format is being used (which is deprecated), then this prefix is
    *                             interpreted as a filename instead.
    * @param  saveablesByDevice   Sequence of device-saveables pairs to restore, sorted by ascending device name. This
    *                             is the result of the [[SaverDefBuilder.groupByDevice]] method.
    * @param  reshape             Boolean value indicating whether to reshape loaded tensors to the shape of the
    *                             corresponding saveable object.
    * @param  restoreSequentially Boolean value indicating whether to restore variables objects within a shard.
    * @param  name                Name for the created op.
    * @return Created op.
    */
  protected def addShardedRestoreOps(
      prefix: Output[String],
      saveablesByDevice: Seq[(String, Set[Saveable])],
      reshape: Boolean,
      restoreSequentially: Boolean,
      name: String = "Restore"
  ): Op[Seq[Output[Any]], Unit] = {
    val restoreOps = saveablesByDevice.map {
      case (device, saveables) =>
        Op.device(device) {
          addRestoreOps(prefix, saveables, restoreSequentially, reshape, name)
        }
    }
    // Create a no-op that has control dependencies for all the updates.
    ControlFlow.group(restoreOps.map(_.asInstanceOf[UntypedOp]).toSet).asInstanceOf[Op[Seq[Output[Any]], Unit]]
  }

  /** Adds save/restore nodes to the graph and creates and returns a [[SaverDef]] proto.
    *
    * @param  saveables                 Objects that need to be saved and loaded.
    * @param  reshape                   Boolean value which, if `true`, allows restoring parameters from a checkpoint
    *                                   where the parameters have a different shape. This is only needed when you try to
    *                                   restore from a Dist-Belief checkpoint, and only some times.
    * @param  sharded                   Boolean value which, if `true`, will shard the checkpoints, one per device that
    *                                   is used by the provided `saveables`.
    * @param  maxToKeep                 Maximum number of checkpoints to keep. As new checkpoints are created, old ones
    *                                   are deleted. If `0`, no checkpoints are deleted from the filesystem but only the
    *                                   last one is kept in the `checkpoint` file. Presently the number is only roughly
    *                                   enforced. For example, in the case of restarts more than `maxToKeep` checkpoints
    *                                   may be kept.
    * @param  keepCheckpointEveryNHours Denotes how often checkpoints should be saved, in hour units. Defaults to
    *                                   10,000 hours.
    * @param  restoreSequentially       Boolean value which, if `true`, causes the restoration of different variables to
    *                                   happen sequentially within each device.
    * @param  filename                  Filename used for the saveable objects saving and loading.
    * @param  name                      Name scope for the created ops.
    * @return Created [[SaverDef]] object.
    */
  def build(
      saveables: Set[Saveable],
      reshape: Boolean = false,
      sharded: Boolean = false,
      maxToKeep: Int = 5,
      keepCheckpointEveryNHours: Float = 10000.0f,
      restoreSequentially: Boolean = false,
      filename: String = "model",
      name: String = "Saver"
  ): SaverDef = {
    SaverDefBuilder.checkSaveables(saveables)
    val (filenameOutput, saveOutput, restoreOp) = Op.nameScope(name) {
      // Add the constant string tensor for the filename.
      val filenameOutput = filename: Output[String]
      // Add the save ops.
      if (sharded) {
        val saveablesByDevice = SaverDefBuilder.groupByDevice(saveables)
        val saveOutput = addShardedSaveOps(filenameOutput, saveablesByDevice)
        val restoreOp = addShardedRestoreOps(
          filenameOutput, saveablesByDevice, reshape, restoreSequentially)
        (filenameOutput, saveOutput, restoreOp)
      } else {
        val saveOutput = addSaveOps(filenameOutput, saveables)
        val restoreOp = addRestoreOps(
          filenameOutput, saveables, reshape, restoreSequentially)
        (filenameOutput, saveOutput, restoreOp)
      }
    }

    SaverDef.newBuilder()
        .setFilenameTensorName(filenameOutput.name)
        .setSaveTensorName(saveOutput.name)
        .setRestoreOpName(restoreOp.name)
        .setSharded(sharded)
        .setMaxToKeep(maxToKeep)
        .setKeepCheckpointEveryNHours(keepCheckpointEveryNHours)
        .setVersion(checkpointFormatVersion)
        .build()
  }
}

/** Contains helper functions for saver builders. */
//noinspection ScalaDeprecation
object SaverDefBuilder {
  /** Groups the provided saveable objects by device and returns a sequence of device-saveables pairs, sorted by
    * ascending device name.
    *
    * @param  saveables Saveables to group by device.
    * @return Sequence of device-saveables pairs, sorted by ascending device name.
    */
  private def groupByDevice(saveables: Set[Saveable]): Seq[(String, Set[Saveable])] = {
    saveables.groupBy(s => DeviceSpecification.fromString(s.device).toString).toSeq.sortBy(_._1)
  }

  /** Checks that the provided saveable objects are valid. More specifically, this function checks if two or more
    * saveable objects have been provided for the same underlying producer. */
  private def checkSaveables(saveables: Set[Saveable]): Unit = {
    val seenProducers = mutable.Set.empty[UntypedOp]
    saveables.foreach(s => {
      s.producerOps.foreach(producer => {
        if (seenProducers.contains(producer))
          throw new IllegalArgumentException(
            s"The same saveable object has been provided twice or with two different names ('${producer.name}').")
        seenProducers += producer
      })
    })
  }

  /** Creates an op that saves the input tensors to disk.
    *
    * The length of `tensorNames` must match the number of tensors in `tensors`. `tensors(i)` is written to `filename`
    * with name `tensorNames(i)`.
    *
    * Note: The created op uses the old V1 checkpoint format. Please use [[saveV2Op]] for creating an op that uses the
    * newer V2 checkpoint format.
    *
    * @param  filename    String tensor containing a single element. That element corresponds to the filename used for
    *                     the save operation.
    * @param  tensorNames One-dimensional string tensor containing the names of the tensors to be saved.
    * @param  tensors     Tensors to save.
    * @param  name        Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If the length of `tensorNames` does not match the number of tensors in `tensors`.
    */
  @deprecated("The V1 checkpoint format version has been deprecated.", "0.1")
  @throws[IllegalArgumentException]
  private def saveOp[T](
      filename: Output[String],
      tensorNames: Seq[String],
      tensors: Seq[Output[T]],
      name: String = "Save"
  ): Op[(Output[String], Output[String], Seq[Output[T]]), Unit] = {
    if (tensorNames.length != tensors.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of tensors in " +
            s"'tensors' (${tensors.length}).")
    // TODO: [TENSORS] !!! Can we avoid all the tensor reshapes in the future? Maybe have a "withRank" function.
    val tensorNamesInput = tensorNames.toTensor.reshape(Shape(tensorNames.length)).toOutput
    Op.Builder[(Output[String], Output[String], Seq[Output[T]]), Unit](
      opType = "Save",
      name = name,
      input = (filename, tensorNamesInput, tensors)
    ).build()
  }

  /** Creates an op that saves the input tensors to disk.
    *
    * The length of `tensorNames` must match the number of tensors in `tensors`. `tensors(i)` is written to `filename`
    * with name `tensorNames(i)`.
    *
    * This is like [[saveOp]] except that tensors can be listed in the saved file as being slices of a larger tensor.
    * `slices` specifies the shape of the larger tensor and the slice that this tensor covers. `slices` must
    * have as many elements as `tensorNames`.
    *
    * Elements of the `slices` input must either be:
    *   - The empty string, in which case the corresponding tensor is saved normally.
    *   - A string of the form `dim0 dim1 ... dimN-1 slice_spec` where the `dimX` are the dimensions of the larger
    * tensor and `slice_spec` specifies which part is covered by the tensor being saved.
    *
    * `slice_spec` itself is a `:`-separated list, `slice0:slice1:...:sliceN-1`, where each `sliceX` is either:
    *   - The string `-` meaning that the slice covers all indices of this dimension.
    *   - The string `start, length` where `start` and `length` are integer. In this case the slice covers `length`
    * indices starting at `start`.
    *
    * Note: The created op uses the old V1 checkpoint format. Please use [[saveV2Op]] for creating an op that uses the
    * newer V2 checkpoint format.
    *
    * @param  filename    String tensor containing a single element. That element corresponds to the filename used for
    *                     the save operation.
    * @param  tensorNames One-dimensional string tensor containing the names of the tensors to be saved.
    * @param  tensors     Tensors to save.
    * @param  slices      Slice specifications of the tensors to be saved. Empty strings indicate that they are
    *                     non-partitioned tensors. If the caller wishes to save specific  slices of full tensors,
    *                     `slices` should be non-empty strings and correspondingly well-formed.
    * @param  name        Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If the length of `tensorNames` does not match the number of tensors in `tensors`,
    *                                  and the number of strings in `slices`.
    */
  @deprecated("The V1 checkpoint format version has been deprecated.", "0.1")
  @throws[IllegalArgumentException]
  private def saveSlicesOp(
      filename: Output[String],
      tensorNames: Seq[String],
      tensors: Seq[Output[Any]],
      slices: Seq[String],
      name: String = "Save"
  ): Op[(Output[String], Output[String], Output[String], Seq[Output[Any]]), Unit] = {
    if (tensorNames.length != tensors.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of tensors in " +
            s"'tensors' (${tensors.length}).")
    if (tensorNames.length != slices.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of slices in " +
            s"'slices' (${slices.length}).")
    val tensorNamesInput = tensorNames.toTensor.reshape(Shape(tensorNames.length)).toOutput
    val slicesInput = slices.toTensor.reshape(Shape(slices.length)).toOutput
    Op.Builder[(Output[String], Output[String], Output[String], Seq[Output[Any]]), Unit](
      opType = "SaveSlices",
      name = name,
      input = (filename, tensorNamesInput, slicesInput, tensors)
    ).build()
  }

  /** Creates an op that restores a tensor from checkpoint files.
    *
    * The op reads a tensor stored in one or several files. If there are several files (for instance because a tensor
    * was saved as slices), `filenamePattern` may contain wildcard symbols (`*` and `?`) in the filename portion only
    * (i.e., not in the directory portion).
    *
    * If a `filenamePattern` matches several files, `preferredShard` can be used to hint in which file the requested
    * tensor is likely to be found. This op will first open the file at index `preferredShard` in the list of matching
    * files and try to restore tensors from that file. Only if some tensors or tensor slices are not found in that first
    * file, will the op open all the other files. This attribute only affects performance, not correctness. The default
    * value, `-1` means that the files are processed in order.
    *
    * @param  filenamePattern String tensor containing a single element. That element corresponds to the filename
    *                         pattern used for the restore operation.
    * @param  tensorName      Name of the tensor to be restored.
    * @param  preferredShard  Index of the file to open first, if multiple files match the provided `filenamePattern`.
    * @param  name            Name for the created op.
    * @return Created op output.
    */
  @deprecated("The V1 checkpoint format version has been deprecated.", "0.1")
  private def restoreOp[T: TF](
      filenamePattern: Output[String],
      tensorName: String,
      preferredShard: Int = -1,
      name: String = "Restore"
  ): Output[T] = {
    Op.Builder[(Output[String], Output[String]), Output[T]](
      opType = "Restore",
      name = name,
      input = (filenamePattern, tensorName)
    ).setAttribute("preferred_shard", preferredShard)
        .build().output
  }

  /** Creates an op that restores a tensor from checkpoint files.
    *
    * This is like [[restoreOp]] except that the restored tensor can be listed as filling only a slice of a larger
    * tensor. `slice` specifies the shape of the larger tensor and the slice that the restored tensor covers. The
    * `slice` input has the same format as the elements of the `slices` input of [[saveSlicesOp]].
    *
    * @param  filenamePattern Filename pattern used for the restore operation.
    * @param  tensorName      Name of the tensor to be restored.
    * @param  slice           Slice specification to use when restoring the tensor.
    * @param  preferredShard  Index of the file to open first, if multiple files match the provided `filenamePattern`.
    * @param  name            Name for the created op.
    * @return Created op output.
    */
  @deprecated("The V1 checkpoint format version has been deprecated.", "0.1")
  private def restoreSliceOp[T: TF](
      filenamePattern: Output[String],
      tensorName: String,
      slice: String,
      preferredShard: Int = -1,
      name: String = "Restore"
  ): Output[T] = {
    Op.Builder[(Output[String], Output[String], Output[String]), Output[T]](
      opType = "RestoreSlice",
      name = name,
      input = (filenamePattern, tensorName, slice)
    ).setAttribute("preferred_shard", preferredShard)
        .build().output
  }

  /** Creates an op that saves the input tensors to disk.
    *
    * The length of `tensorNames` must match the number of tensors in `tensors`. `tensors(i)` is written to `filename`
    * with name `tensorNames(i)`.
    *
    * `slices` specifies the shape of the larger tensor and the slice that this tensor covers. `slices` must
    * have as many elements as `tensorNames`.
    *
    * Elements of the `slices` input must either be:
    *   - The empty string, in which case the corresponding tensor is saved normally.
    *   - A string of the form `dim0 dim1 ... dimN-1 slice_spec` where the `dimX` are the dimensions of the larger
    * tensor and `slice_spec` specifies which part is covered by the tensor being saved.
    *
    * `slice_spec` itself is a `:`-separated list, `slice0:slice1:...:sliceN-1`, where each `sliceX` is either:
    *   - The string `-` meaning that the slice covers all indices of this dimension.
    *   - The string `start, length` where `start` and `length` are integer. In this case the slice covers `length`
    * indices starting at `start`.
    *
    * @param  prefix      String tensor containing a single element. That element corresponds to the prefix of the V2
    *                     checkpoint to which we write the tensors.
    * @param  tensorNames One-dimensional string tensor containing the names of the tensors to be saved.
    * @param  tensors     Tensors to save.
    * @param  slices      Slice specifications of the tensors to be saved. Empty strings indicate that they are
    *                     non-partitioned tensors. If the caller wishes to save specific  slices of full tensors,
    *                     `slices` should be non-empty strings and correspondingly well-formed.
    * @param  name        Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If the length of `tensorNames` does not match the number of tensors in `tensors`,
    *                                  and the number of strings in `slices`.
    */
  @throws[IllegalArgumentException]
  private def saveV2Op(
      prefix: Output[String],
      tensorNames: Seq[String],
      tensors: Seq[Output[Any]],
      slices: Seq[String],
      name: String = "Save"
  ): Op[(Output[String], Output[String], Output[String], Seq[Output[Any]]), Unit] = {
    if (tensorNames.length != tensors.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of tensors in " +
            s"'tensors' (${tensors.length}).")
    if (tensorNames.length != slices.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of slices in " +
            s"'slices' (${slices.length}).")
    val tensorNamesInput = tensorNames.toTensor.reshape(Shape(tensorNames.length)).toOutput
    val slicesInput = slices.toTensor.reshape(Shape(slices.length)).toOutput
    Op.Builder[(Output[String], Output[String], Output[String], Seq[Output[Any]]), Unit](
      opType = "SaveV2",
      name = name,
      input = (prefix, tensorNamesInput, slicesInput, tensors)
    ).setAttribute("dtypes", tensors.map(_.dataType).toArray)
        .build()
  }

  /** Creates an op that restores tensors from V2 checkpoint files.
    *
    * For backward compatibility with the V1 format, the created op currently allows restoring from a V1 checkpoint as
    * well:
    *   - The op first attempts to find the V2 index file pointed to by `prefix`, and if found proceeds to read it as a
    * V2 checkpoint.
    *   - Otherwise the V1 read path is invoked. Relying on this behavior is not recommended, as the ability to fall
    * back to read V1 might be deprecated and eventually removed.
    *
    * By default, the op restores the named tensors in full. If the caller wishes to restore specific slices of stored
    * tensors, `slices` must contain non-empty strings and correspondingly well-formed (the format is the same as that
    * of the `slices` input of [[saveV2Op]]).
    *
    * Callers must ensure that all the named tensors are indeed stored in the checkpoint.
    *
    * Regarding the V1 read path procedure, `prefix` is treated as a filename pattern. The op then reads a tensor stored
    * in one or several files. If there are several files (for instance because a tensor was saved as slices),
    * `prefix` may contain wildcard symbols (`*` and `?`) in the filename portion only (i.e., not in the directory
    * portion).
    *
    * @param  prefix      String tensor containing a single element. That element corresponds to the prefix of the V2
    *                     checkpoint to which we write the tensors.
    * @param  tensorNames Names of the tensors to be restored.
    * @param  slices      Slice specifications to use when restoring the tensors.
    * @param  dataTypes   Data types of the tensors being restored.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    * @throws IllegalArgumentException If the length of `tensorNames` does not match the number of string in `slices`,
    *                                  and the number of data types in `dataTypes`.
    */
  @throws[IllegalArgumentException]
  private def restoreV2Op[T](
      prefix: Output[String],
      tensorNames: Seq[String],
      slices: Seq[String],
      dataTypes: Seq[DataType[T]],
      name: String = "Restore"
  ): Seq[Output[T]] = {
    if (tensorNames.length != slices.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of slices in " +
            s"'slices' (${slices.length}).")
    if (tensorNames.length != dataTypes.length)
      throw new IllegalArgumentException(
        s"The number of tensor names provided (${tensorNames.length}) does not match the number of data types in " +
            s"'dataTypes' (${dataTypes.length}).")
    val tensorNamesInput = tensorNames.toTensor.reshape(Shape(tensorNames.length)).toOutput
    val slicesInput = slices.toTensor.reshape(Shape(slices.length)).toOutput
    Op.Builder[(Output[String], Output[String], Output[String]), Seq[Output[T]]](
      opType = "RestoreV2",
      name = name,
      input = (prefix, tensorNamesInput, slicesInput)
    ).setAttribute("dtypes", dataTypes.map(_.asInstanceOf[DataType[Any]]).toArray)
        .build().output
  }

  /** Creates an op that merges the metadata files of sharded checkpoints (the op is V2 checkpoint format specific).
    *
    * The result is one logical checkpoint, with one physical metadata file and renamed data files. This op is intended
    * for "grouping" multiple checkpoints in a sharded checkpoint setup.
    *
    * @param  checkpointPrefixes   Prefixes of the V2 checkpoints to merge.
    * @param  destinationPrefix    Desired final prefix. That prefix is allowed to be the same as one of the
    *                              `checkpointPrefixes`.
    * @param  deleteOldDirectories If `true`, the op attempts to recursively delete the directory of each path in the
    *                              input `checkpointPrefixes`. This is useful when those paths are non user-facing
    *                              temporary locations.
    * @param  name                 Name for the created op.
    * @return Created op.
    */
  private def mergeV2Checkpoints(
      checkpointPrefixes: Output[String],
      destinationPrefix: Output[String],
      deleteOldDirectories: Boolean = true,
      name: String = "MergeV2Checkpoints"
  ): Op[(Output[String], Output[String]), Unit] = {
    Op.Builder[(Output[String], Output[String]), Unit](
      opType = "MergeV2Checkpoints",
      name = name,
      input = (checkpointPrefixes, destinationPrefix)
    ).setAttribute("delete_old_dirs", deleteOldDirectories)
        .build()
  }

  /** Creates an op that generates a sharded filename. The filename is `printf` formatted as
    * `%s-%05d-of-%05d, basename, shard, num_shards`.
    *
    * @param  filename       Base filename.
    * @param  shard          Shard index.
    * @param  numberOfShards Total number of shards.
    * @param  name           Created op name.
    * @return Created scalar op output containing the sharded filename.
    */
  private def shardedFilenameOp(
      filename: Output[String],
      shard: Output[Int],
      numberOfShards: Output[Int],
      name: String = "ShardedFilename"
  ): Output[String] = {
    Op.Builder[(Output[String], Output[Int], Output[Int]), Output[String]](
      opType = "ShardedFilename",
      name = name,
      input = (filename, shard, numberOfShards)
    ).build().output
  }

  /** Creates an op that generates a glob pattern matching all sharded file names.
    *
    * @param  filename       String tensor containing a single element. That element corresponds to the base filename.
    * @param  numberOfShards Scalar tensor containing the total number of shards.
    * @param  name           Created op name.
    * @return Created scalar op output containing a filename pattern string.
    */
  private def shardedFilenameSpecificationOp(
      filename: Output[String],
      numberOfShards: Output[Int],
      name: String = "ShardedFilenameSpecification"
  ): Output[String] = {
    Op.Builder[(Output[String], Output[Int]), Output[String]](
      opType = "ShardedFilespec",
      name = name,
      input = (filename, numberOfShards)
    ).build().output
  }
}

/** The default saver builder. */
private[variables] object DefaultSaverDefBuilder extends SaverDefBuilder

/** Class used to describe tensor slices that need to be saved.
  *
  * @param  name                   Name to save `value` under.
  * @param  value                  Value that needs to be saved.
  * @param  saveSliceSpecification Slice specification string used for saving.
  */
case class SaveSpecification private(
    name: String,
    value: () => Output[Any],
    saveSliceSpecification: String)

/** Base class for defining objects that be saved and restored.
  *
  * @param  saveSpecifications Sequence containing a save specification per tensor that needs to be saved.
  */
abstract class Saveable protected(val saveSpecifications: Seq[SaveSpecification]) {
  /** Name to save the object under. */
  val name: String

  /** The "producer" ops that this saveable wraps. For example, a `Variable` op saving its backing tensor. */
  val producerOps: Set[UntypedOp]

  /** Device of this saveable object. All tensors that need to be saved must lie on the same device. */
  def device: String = {
    val device = saveSpecifications.head.value().device
    if (saveSpecifications.exists(_.value().device != device))
      throw new IllegalArgumentException(
        "All tensors being saved under one saveable object must lie on the same device.")
    device
  }

  /** Restores this saveable object from a set of tensors that were loaded from a checkpoint.
    *
    * @param  restoredTensors Tensors that were loaded from a checkpoint.
    * @param  restoredShapes  Shapes that this object should conform to after the restore. If `null`, this argument is
    *                         ignored.
    * @return Op that restores the state of this saveable object.
    */
  private[api] def restore[I: IsInt32OrInt64](
      restoredTensors: Seq[Output[Any]],
      restoredShapes: Seq[Output[I]] = null
  ): UntypedOp
}

private[ops] object Saveable {
  /** Wrapper saveable object that allows variables to be saved. */
  implicit class VariableSaveable(variable: Variable[Any])
      extends Saveable(
        Seq(SaveSpecification(
          if (variable.partitionInformation != null) variable.partitionInformation.fullName else variable.name,
          () => variable.value,
          Option(variable.partitionInformation).map(_.saveSpecString).getOrElse("")))) {
    private val variableDevice: String = {
      variable.device
    }

    override val name: String = {
      if (variable.partitionInformation != null)
        variable.partitionInformation.fullName
      else
        variable.name
    }

    override val producerOps: Set[UntypedOp] = {
      Set(variable.op)
    }

    override private[api] def restore[I: IsInt32OrInt64](
        restoredTensors: Seq[Output[Any]],
        restoredShapes: Seq[Output[I]] = null
    ): UntypedOp = {
      val dataType = restoredTensors.head.dataType
      var restoredTensor = restoredTensors.head
      if (restoredShapes != null) {
        val shapeDataType = restoredShapes.head.dataType
        restoredTensor = Basic.reshape(
          restoredTensors.head,
          restoredShapes.head
        )(TF.fromDataType(dataType), TF.fromDataType(shapeDataType), IsInt32OrInt64[I])
      }
      // Copy the restored tensor to the variable's device.
      restoredTensor = Op.createWith(device = variableDevice) {
        Basic.identity(restoredTensor)(TF.fromDataType(dataType))
      }
      Variable.assign(
        variable.handle,
        restoredTensor
      )(TF.fromDataType(dataType))
    }
  }
}
