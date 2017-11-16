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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.ops.io.data.BatchDataset.BatchDatasetOps
import org.platanios.tensorflow.api.ops.io.data.CacheDataset.CacheDatasetOps
import org.platanios.tensorflow.api.ops.io.data.ConcatenatedDataset.ConcatenatedDatasetOps
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.ops.io.data.DropDataset.DropDatasetOps
import org.platanios.tensorflow.api.ops.io.data.FilterDataset.FilterDatasetOps
import org.platanios.tensorflow.api.ops.io.data.FlatMapDataset.FlatMapDatasetOps
import org.platanios.tensorflow.api.ops.io.data.GroupByWindowDataset.GroupByWindowDatasetOps
import org.platanios.tensorflow.api.ops.io.data.IgnoreErrorsDataset.IgnoreErrorsDatasetOps
import org.platanios.tensorflow.api.ops.io.data.MapDataset.MapDatasetOps
import org.platanios.tensorflow.api.ops.io.data.PaddedBatchDataset.PaddedBatchDatasetOps
import org.platanios.tensorflow.api.ops.io.data.PrefetchDataset.PrefetchDatasetOps
import org.platanios.tensorflow.api.ops.io.data.RepeatDataset.RepeatDatasetOps
import org.platanios.tensorflow.api.ops.io.data.ShuffleDataset.ShuffleDatasetOps
import org.platanios.tensorflow.api.ops.io.data.TakeDataset.TakeDatasetOps
import org.platanios.tensorflow.api.ops.io.data.ZipDataset.ZipDatasetOps

/** Groups together all implicits related to the data API.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Data {
  implicit def datasetToBatchDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): BatchDatasetOps[T, O, D, S] = {
    BatchDatasetOps(dataset)
  }

  implicit def datasetToCacheDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): CacheDatasetOps[T, O, D, S] = {
    CacheDatasetOps(dataset)
  }

  implicit def datasetToConcatenatedDatasetOps[T, O, D, S](
      dataset: Dataset[T, O, D, S]): ConcatenatedDatasetOps[T, O, D, S] = {
    ConcatenatedDatasetOps(dataset)
  }

  implicit def datasetToDropDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): DropDatasetOps[T, O, D, S] = {
    DropDatasetOps(dataset)
  }

  implicit def datasetToFilterDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): FilterDatasetOps[T, O, D, S] = {
    FilterDatasetOps(dataset)
  }

  implicit def datasetToFlatMapDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): FlatMapDatasetOps[T, O, D, S] = {
    FlatMapDatasetOps(dataset)
  }

  implicit def datasetToGroupByWindowDatasetOps[T, O, D, S](
      dataset: Dataset[T, O, D, S]): GroupByWindowDatasetOps[T, O, D, S] = {
    GroupByWindowDatasetOps(dataset)
  }

  implicit def datasetToIgnoreErrorsDatasetOps[T, O, D, S](
      dataset: Dataset[T, O, D, S]): IgnoreErrorsDatasetOps[T, O, D, S] = {
    IgnoreErrorsDatasetOps(dataset)
  }

  implicit def datasetToMapDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): MapDatasetOps[T, O, D, S] = {
    MapDatasetOps(dataset)
  }

  implicit def datasetToPaddedBatchDatasetOps[T, O, D, S](
      dataset: Dataset[T, O, D, S]): PaddedBatchDatasetOps[T, O, D, S] = {
    PaddedBatchDatasetOps(dataset)
  }

  implicit def datasetToPrefetchDatasetOps[T, O, D, S](
      dataset: Dataset[T, O, D, S]): PrefetchDatasetOps[T, O, D, S] = {
    PrefetchDatasetOps(dataset)
  }

  implicit def datasetToRepeatDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): RepeatDatasetOps[T, O, D, S] = {
    RepeatDatasetOps(dataset)
  }

  implicit def datasetToShuffleDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): ShuffleDatasetOps[T, O, D, S] = {
    ShuffleDatasetOps(dataset)
  }

  implicit def datasetToTakeDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): TakeDatasetOps[T, O, D, S] = {
    TakeDatasetOps(dataset)
  }

  implicit def datasetToZipDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]): ZipDatasetOps[T, O, D, S] = {
    ZipDatasetOps(dataset)
  }
}
