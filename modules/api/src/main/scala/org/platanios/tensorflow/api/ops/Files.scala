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

package org.platanios.tensorflow.api.ops

/** Contains functions for constructing ops related to working with files.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Files {
  /** Creates an op that reads and outputs the entire contents of the file pointed to by the input filename.
    *
    * @param  filename Scalar tensor containing the filename.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def readFile(
      filename: Output[String],
      name: String = "ReadFile"
  ): Output[String] = {
    Op.Builder[Output[String], Output[String]](
      opType = "ReadFile",
      name = name,
      input = filename
    ).build().output
  }

  /** Creates an op that writes `contents` to the file pointed to by the input filename.
    *
    * The op creates the file and recursively creates the directory, if it does not already exist.
    *
    * @param  filename Scalar tensor containing the filename.
    * @param  contents Scalar tensor containing the contents to write to the provided file.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def writeFile(
      filename: Output[String],
      contents: Output[String],
      name: String = "WriteFile"
  ): Op[(Output[String], Output[String]), Unit] = {
    Op.Builder[(Output[String], Output[String]), Unit](
      opType = "WriteFile",
      name = name,
      input = (filename, contents)
    ).build()
  }

  /** Creates an op that returns the set of files matching one or more glob patterns.
    *
    * **Note:** The op only supports wildcard characters in the basename portion of the pattern and not in the directory
    * portion.
    *
    * @param  pattern Scalar or vector tensor containing the shell wildcard pattern(s).
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def matchingFiles(
      pattern: Output[String],
      name: String = "MatchingFiles"
  ): Output[String] = {
    Op.Builder[Output[String], Output[String]](
      opType = "MatchingFiles",
      name = name,
      input = pattern
    ).build().output
  }
}

object Files extends Files
