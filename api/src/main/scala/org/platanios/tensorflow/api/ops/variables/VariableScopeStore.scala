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

import org.platanios.tensorflow.api.ops.Op

/** A thread-local score for the current variable scope and scope counts.
  *
  * @author Emmanouil Antonios Platanios
  */
case class VariableScopeStore private[api]() {
  private[api] var scope: VariableScope = VariableScope(CreateNewOnly)

  /** Map with variable scope names as keys and the corresponding use counts as values. */
  private[api] var variableScopeCounts: Map[String, Int] = Map.empty[String, Int]

  private[api] def enterVariableScope(scope: String): Unit = {
    variableScopeCounts += scope -> (variableScopeCounts.getOrElse(scope, 0) + 1)
  }

  private[api] def closeVariableSubScopes(scope: String): Unit = {
    variableScopeCounts.keySet.filter(_.startsWith(s"$scope/")).foreach(variableScopeCounts -= _)
  }

  /** Returns the use count of the provided scope in this variable store.
    *
    * @param  scope Variable scope name.
    * @return Number of usages of the provided variable scope name, in this variable store.
    */
  private[api] def variableScopeCount(scope: String): Int = variableScopeCounts.getOrElse(scope, 0)
}

object VariableScopeStore {
  def current: VariableScopeStore = Op.currentGraph.variableScopeStore.get
}
