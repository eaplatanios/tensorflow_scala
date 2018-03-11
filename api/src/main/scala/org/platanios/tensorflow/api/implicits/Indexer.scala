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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.core.{Index, IndexerConstructionWithOneNumber}

/** Groups together all implicits related to indexers.
  *
  * @author Emmanouil Antonios Platanios
  */
private[implicits] trait Indexer {
  // TODO: Add begin mask support (not simple).

  implicit def intToIndex(index: Int): Index = Index(index = index)

  implicit def intToIndexerConstruction(n: Int): IndexerConstructionWithOneNumber = {
    IndexerConstructionWithOneNumber(n)
  }
}
