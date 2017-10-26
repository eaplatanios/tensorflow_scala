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

package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
case class FileStatistics(length: Long, lastModifiedTime: Long, isDirectory: Boolean)

object FileIO {
  TensorFlow.load()

  @native def fileExists(filename: String): Unit
  @native def deleteFile(filename: String): Unit
  @native def readFileToString(filename: String): String
  @native def writeStringToFile(filename: String, content: String): Unit
  @native def getChildren(filename: String): Array[String]
  @native def getMatchingFiles(filename: String): Array[String]
  @native def mkDir(dirname: String): Unit
  @native def mkDirs(dirname: String): Unit
  @native def copyFile(oldPath: String, newPath: String, overwrite: Boolean): Unit
  @native def renameFile(oldPath: String, newPath: String, overwrite: Boolean): Unit
  @native def deleteRecursively(dirname: String): Unit
  @native def isDirectory(dirname: String): Boolean
  @native def statistics(path: String): FileStatistics

  @native def newBufferedInputStream(filename: String, bufferSize: Long): Long
  @native def readFromBufferedInputStream(handle: Long, numBytes: Long): String
  @native def readLineAsStringFromBufferedInputStream(handle: Long): String
  @native def tellBufferedInputStream(handle: Long): Long
  @native def seekBufferedInputStream(handle: Long, position: Long): Unit
  @native def deleteBufferedInputStream(handle: Long): Unit

  @native def newWritableFile(filename: String, mode: String): Long
  @native def appendToWritableFile(handle: Long, content: String): Unit
  @native def flushWritableFile(handle: Long): Unit
  @native def deleteWritableFile(handle: Long): Unit
}
