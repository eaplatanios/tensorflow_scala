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

package org.platanios.tensorflow.api.utilities

import scala.collection.mutable
import scala.util.Random

/** A key-value store using deterministic reservoir sampling.
  *
  * Items are added with an associated key. Items may be retrieved by the corresponding key, and a list of keys can also
  * be retrieved. If `maxSize` is not zero, then it dictates the maximum number of items that will be stored for each
  * key. Once there are more items for a given key, they are replaced via reservoir sampling, such that each item has an
  * equal probability of being included in the sample.
  *
  * Deterministic means that for any given seed and bucket size, the sequence of values that are kept for any given key
  * will always be the same, and that this is independent of any insertions for other keys. That is:
  *
  * {{{
  *   val reservoirA = ReservoirKVStore(10)
  *   val reservoirB = ReservoirKVStore(10)
  *   (0 until 100).foreach(i => reservoirA.add("key1", i))
  *   (0 until 100).foreach(i => reservoirA.add("key2", i))
  *   (0 until 100).foreach(i => {
  *     reservoirB.add("key1", i)
  *     reservoirB.add("key2", i)
  *   })
  * }}}
  *
  * After executing this code, `reservoirA` and `reservoirB` will be in identical states.
  *
  * For more information on reservoir sampling, refer to [this page](https://en.wikipedia.org/wiki/Reservoir_sampling).
  *
  * Note that, adding items has amortized `O(1)` runtime cost.
  *
  * @param  maxSize        Maximum size of each bucket in this reservoir key-value store.
  * @param  seed           Seed to use for the random number generator used while sampling.
  * @param  alwaysKeepLast Boolean flag indicating whether to always store the last seen item. If set to `true` and the
  *                        last seen item was not sampled to be stored, then it replaces the last item in the
  *                        corresponding bucket.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Reservoir[K, V](maxSize: Int, seed: Long = 0L, alwaysKeepLast: Boolean = true) {
  require(maxSize >= 0, s"'maxSize' (= $maxSize) must be a non-negative integer.")

  // This lock protects the internal items, ensuring that calls to `add()` and `filter()` are thread-safe.
  private[this] object ItemsLock

  private[utilities] val buckets: mutable.Map[K, ReservoirBucket[V]] = mutable.Map.empty[K, ReservoirBucket[V]]

  /** Returns all the keys in the reservoir. */
  def keys: Iterable[K] = buckets.keys

  /** Returns all the items stored for the provided key and throws an exception if the key does not exist. */
  def items(key: K): List[V] = buckets(key).items

  /** Adds a new item to the reservoir with the provided key.
    *
    * If the corresponding reservoir has not yet reached full size, then the new item is guaranteed to be added. If the
    * reservoir is full, then the behavior of this method depends on the value of `alwaysKeepLast`.
    *
    * If `alwaysKeepLast` is set to `true`, then the new item is guaranteed to be added to the reservoir, and either the
    * previous last item will be replaced, or (with low probability) an older item will be replaced.
    *
    * If `alwaysKeepLast` is set to `false`, then the new item may replace an old item with low probability.
    *
    * If `transformFn` is provided, then it will be applied to transform the provided item (lazily, if and only if the
    * item is going to be included in the reservoir).
    *
    * @param  key         Key for the item to add.
    * @param  item        Item to add.
    * @param  transformFn Transform function for the item to add.
    */
  def add(key: K, item: V, transformFn: (V) => V = identity[V]): Unit = ItemsLock synchronized {
    buckets.getOrElseUpdate(key, ReservoirBucket(maxSize, new Random(seed), alwaysKeepLast)).add(item, transformFn)
  }

  /** Filters the items in this reservoir using the provided filtering function.
    *
    * When filtering items from each reservoir bucket, we must update the internal state variable `numItemsSeen`, which
    * is used for determining the rate of replacement in reservoir sampling. Ideally, `numItemsSeen` would contain the
    * exact number of items that have ever been seen by the `add` function of this reservoir, and that satisfy the
    * provided filtering function. However, the reservoir bucket does not have access to all of the items it has seen --
    * it only has access to the subset of items that have survived sampling (i.e., `_items`). Therefore, we estimate
    * `numItemsSeen` by scaling its original value by the same ratio as the ratio of items that were not filtered out
    * and that are currently stored in this reservoir bucket.
    *
    * @param  filterFn Filtering function that returns `true` for the items to be kept in the reservoir.
    * @param  key      Optional key for which to filter the values. If `None` (the default), then the values for all
    *                  keys in the reservoir are filtered.
    * @return Number of items filtered from this reservoir.
    */
  def filter(filterFn: (V) => Boolean, key: Option[K] = None): Int = ItemsLock synchronized {
    if (key.isDefined)
      buckets.get(key.get).map(_.filter(filterFn)).getOrElse(0)
    else
      buckets.values.map(_.filter(filterFn)).sum
  }
}

/** Container for items coming from a stream, that implements reservoir sampling so that its size never exceeds
  * `maxSize`.
  *
  * @param  maxSize        Maximum size of this bucket.
  * @param  random         Random number generator to use while sampling.
  * @param  alwaysKeepLast Boolean flag indicating whether to always store the last seen item. If set to `true` and the
  *                        last seen item was not sampled to be stored, then it replaces the last item in this bucket.
  */
case class ReservoirBucket[T](maxSize: Int, random: Random = new Random(0), alwaysKeepLast: Boolean = true) {
  require(maxSize >= 0, s"'maxSize' (= $maxSize) must be a non-negative integer.")

  // This lock protects the internal items, ensuring that calls to `add()` and `filter()` are thread-safe.
  private[this] object ItemsLock

  private[this] var _items: List[T] = List.empty[T]

  private[utilities] var numItemsSeen: Int = 0

  /** Returns all the items stored in this bucket. */
  def items: List[T] = _items

  /** Adds an item to this reservoir bucket, replacing an old item, if necessary.
    *
    * If `alwaysKeepLast` is `true`, then the new item is guaranteed to be added to the bucket, and to be the last
    * element in the bucket. If the bucket has reached capacity, then an old item will be replaced. With probability
    * `maxSize / numItemsSeen` a random item in the bucket will be popped out and the new item will be appended to the
    * end. With probability `1 - maxSize / numItemsSeen`, the last item in the bucket will be replaced.
    *
    * If `alwaysKeepLast` is `false`, then with probability `1 - maxSize / numItemsSeen` the new item may not be added
    * to the reservoir at all.
    *
    * Since the `O(n)` replacements occur with `O(1/numItemsSeen)` likelihood, the amortized runtime cost is `O(1)`.
    *
    * @param  item        Item to add.
    * @param  transformFn A function used to transform the item before addition, if the item will be kept in the
    *                     reservoir.
    */
  def add(item: T, transformFn: (T) => T = identity[T]): Unit = ItemsLock synchronized {
    if (_items.size < maxSize || maxSize == 0) {
      _items :+= transformFn(item)
    } else {
      val r = random.nextInt(numItemsSeen)
      if (r < maxSize) {
        _items = _items.patch(r, Nil, 1)
        _items :+= transformFn(item)
      } else if (alwaysKeepLast) {
        _items = _items.updated(_items.size - 1, transformFn(item))
      }
    }
    numItemsSeen += 1
  }

  /** Filters the items in this reservoir using the provided filtering function.
    *
    * When filtering items from the reservoir bucket, we must update the internal state variable `numItemsSeen`, which
    * is used for determining the rate of replacement in reservoir sampling. Ideally, `numItemsSeen` would contain the
    * exact number of items that have ever been seen by the `add` function of this reservoir, and that satisfy the
    * provided filtering function. However, the reservoir bucket does not have access to all of the items it has seen --
    * it only has access to the subset of items that have survived sampling (i.e., `_items`). Therefore, we estimate
    * `numItemsSeen` by scaling its original value by the same ratio as the ratio of items that were not filtered out
    * and that are currently stored in this reservoir bucket.
    *
    * @param  filterFn Filtering function that returns `true` for the items to be kept in the reservoir.
    * @return Number of items filtered from this reservoir bucket.
    */
  def filter(filterFn: (T) => Boolean): Int = ItemsLock synchronized {
    val sizeBefore = _items.size
    _items = _items.filter(filterFn)
    val numFiltered = sizeBefore - _items.size
    // Estimate a correction for the number of items seen.
    val proportionRemaining = if (sizeBefore > 0) _items.size.toFloat / sizeBefore else 0.0f
    numItemsSeen = Math.round(proportionRemaining * numItemsSeen)
    numFiltered
  }
}
