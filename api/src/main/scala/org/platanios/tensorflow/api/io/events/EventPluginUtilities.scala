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

package org.platanios.tensorflow.api.io.events

import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, NotFoundException}
import org.platanios.tensorflow.api.io.FileIO

import java.nio.file.Path

/** Contains utilities for managing event plugins (e.g., TensorBoard plugins).
  *
  * @author Emmanouil Antonios Platanios
  */
object EventPluginUtilities {
  private[this] val PLUGINS_DIR: String = "plugins"

  /** Returns the plugin directory for the provided plugin name. */
  def pluginDir(logDir: Path, pluginName: String): Path = {
    logDir.resolve(PLUGINS_DIR).resolve(pluginName)
  }

  /** Returns a sequence with all the plugin directories that have contain registered assets, in `logDir`.
    *
    * If a plugins directory does not exist in `logDir`, then this method returns an empty list. This maintains
    * compatibility with old log directories that contain no plugin sub-directories.
    */
  def listPluginDirs(logDir: Path): Seq[Path] = {
    val pluginsDir = logDir.resolve(PLUGINS_DIR)
    if (!FileIO.isDirectory(pluginsDir)) {
      Seq.empty[Path]
    } else {
      FileIO.listDirectories(pluginsDir).filter(d => FileIO.isDirectory(pluginsDir.resolve(d)))
    }
  }

  /** Returns a sequence with paths to all the registered assets for the provided plugin name, in `logDir`.
    *
    * If a plugins directory does not exist in `logDir`, then this method returns an empty list. This maintains
    * compatibility with old log directories that contain no plugin sub-directories.
    */
  def listPluginAssets(logDir: Path, pluginName: String): Seq[Path] = {
    val pluginsDir = pluginDir(logDir, pluginName)
    if (!FileIO.isDirectory(pluginsDir)) {
      Seq.empty[Path]
    } else {
      FileIO.listDirectories(pluginsDir).filter(d => FileIO.isDirectory(pluginsDir.resolve(d)))
    }
  }

  /** Retrieves a particular plugin asset from `logDir` and returns it as a string. */
  def retrievePluginAsset(logDir: Path, pluginName: String, assetName: String): String = {
    val assetPath = pluginDir(logDir, pluginName).resolve(assetName)
    try {
      val fileIO = FileIO(assetPath, FileIO.READ)
      val content = fileIO.read()
      fileIO.close()
      content
    } catch {
      case _: NotFoundException => throw InvalidArgumentException(s"Asset path '$assetPath' not found.")
      case t: Throwable => throw InvalidArgumentException(s"Could not read asset path '$assetPath'.", t)
    }
  }
}
