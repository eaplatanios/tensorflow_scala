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

package org.platanios.tensorflow.api.ops.io

import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}

/** Contains helper functions and classes for creating file reading/writing-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
object Files {
  /** Creates an op that reads and outputs the entire contents of the file pointed to by the input filename.
    *
    * @param  filename `STRING` scalar tensor containing the filename.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the file contents.
    */
  private[io] def readFile(filename: Output, name: String = "ReadFile"): Output = {
    Op.Builder(opType = "ReadFile", name = name)
        .addInput(filename)
        .build().outputs(0)
  }

  /** Creates an op that writes `contents` to the file pointed to by the input filename.
    *
    * The op creates the file and recursively creates the directory, if it does not already exist.
    *
    * @param  filename `STRING` scalar tensor containing the filename.
    * @param  contents `STRING` scalar tensor containing the contents to write to the provided file.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the file contents.
    */
  private[io] def writeFile(filename: Output, contents: Output, name: String = "WriteFile"): Op = {
    Op.Builder(opType = "WriteFile", name = name)
        .addInput(filename)
        .addInput(contents)
        .build()
  }

  /** Creates an op that returns the set of files matching one or more glob patterns.
    *
    * **Note:** The op only supports wildcard characters in the basename portion of the pattern and not in the directory
    * portion.
    *
    * @param  pattern `STRING` scalar or vector tensor containing the shell wildcard pattern(s).
    * @param  name    Name for the created op.
    * @return Created op output, which is a `STRING` vector tensor containing the matching filenames.
    */
  private[io] def matchingFiles(pattern: Output, name: String = "MatchingFiles"): Output = {
    Op.Builder(opType = "MatchingFiles", name = name)
        .addInput(pattern)
        .build().outputs(0)
  }

  private[io] object Gradients {
    GradientsRegistry.registerNonDifferentiable("ReadFile")
    GradientsRegistry.registerNonDifferentiable("WriteFile")
    GradientsRegistry.registerNonDifferentiable("MatchingFiles")
  }
}
