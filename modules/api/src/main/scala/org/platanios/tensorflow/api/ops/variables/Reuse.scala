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

package org.platanios.tensorflow.api.ops.variables

/** Enumeration of possible variable reuse options, used by variable scopes and variable stores.
  *
  * The supported options are:
  *   - [[ReuseExistingOnly]]: Reuse existing variables only and throw an exception if no appropriate variable exists.
  *   - [[CreateNewOnly]]: Create new variables only and throw an exception if a variable with the same name exists.
  *   - [[ReuseOrCreateNew]]: Reuse existing variables or create new ones, if no variable with the provided name exists.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Reuse

/** Trait marking the variable reuse modes that allow reusing existing variables. */
sealed trait ReuseAllowed extends Reuse

/** Reuse existing variables only and throw an exception if no appropriate variable exists. */
case object ReuseExistingOnly extends ReuseAllowed

/** Create new variables only and throw an exception if a variable with the same name exists. */
case object CreateNewOnly extends Reuse

/** Reuse existing variables or create new ones, if no variable with the provided name exists. */
case object ReuseOrCreateNew extends ReuseAllowed
