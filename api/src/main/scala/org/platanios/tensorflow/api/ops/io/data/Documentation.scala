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

package org.platanios.tensorflow.api.ops.io.data

/** Collects all documentation macros for the data API.
  *
  * @author Emmanouil Antonios Platanios
  */
private[ops] trait Documentation
    extends BatchDataset.Documentation
        with CacheDataset.Documentation
        with ConcatenatedDataset.Documentation
        with DropDataset.Documentation
        with FilterDataset.Documentation
        with FlatMapDataset.Documentation
        with GroupByWindowDataset.Documentation
        with IgnoreErrorsDataset.Documentation
        with MapDataset.Documentation
        with PaddedBatchDataset.Documentation
        with PrefetchDataset.Documentation
        with RangeDataset.Documentation
        with RepeatDataset.Documentation
        with ShuffleDataset.Documentation
        with TakeDataset.Documentation
        with ZipDataset.Documentation
