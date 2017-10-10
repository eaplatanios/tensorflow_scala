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

/** Groups together documentation related to constructing symbolic ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Documentation
    extends Basic.Documentation
        with Checks.Documentation
        with Clip.Documentation
        with Embedding.Documentation
        with Image.Documentation
        with Logging.Documentation
        with Math.Documentation
        with NN.Documentation
        with Parsing.Documentation
        with Random.Documentation
        with Sparse.Documentation
        with Statistics.Documentation
        with Summary.Documentation
        with Text.Documentation
        with control_flow.ControlFlow.Documentation
