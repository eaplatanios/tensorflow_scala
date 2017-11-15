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

package org.platanios.tensorflow.data.nmt.utilities

import org.platanios.tensorflow.api._

/** Contains utilities for dealing with Neural Machine Translation (NMT) datasets.
  *
  * @author Emmanouil Antonios Platanios
  */
object Datasets {
  type NMTInferDataset = tf.data.Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)]
  type NMTTrainDataset = tf.data.Dataset[
      (Tensor, Tensor, Tensor, Tensor, Tensor),
      (Output, Output, Output, Output, Output),
      (DataType, DataType, DataType, DataType, DataType),
      (Shape, Shape, Shape, Shape, Shape)]

  def createInferDataset(
      srcDataset: tf.data.Dataset[Tensor, Output, DataType, Shape],
      srcVocabularyTable: tf.LookupTable,
      batchSize: Int,
      endSequenceToken: String = Vocabulary.END_SEQUENCE_TOKEN,
      srcReverse: Boolean = false,
      srcMaxLength: Int = -1
  ): NMTInferDataset = {
    val srcEosId = srcVocabularyTable.lookup(tf.constant(endSequenceToken)).cast(INT32)

    srcDataset
        .map(o => tf.stringSplit(o).values)
        // Crop based on the maximum allowed sequence length.
        .transform(d => if (srcMaxLength != -1) d.map(dd => dd(0 :: srcMaxLength)) else d)
        // Reverse the source sequence if necessary.
        .transform(d => if (srcReverse) d.map(dd => tf.reverse(dd, axes = 0)) else d)
        // Convert the word strings to IDs. Word strings that are not in the vocabulary
        // get the lookup table's default value.
        .map(d => tf.cast(srcVocabularyTable.lookup(d), INT32))
        // Add sequence lengths.
        .map(d => (d, tf.size(d, INT32)))
        .dynamicPaddedBatch(
          batchSize, 
          // The first entry represents the source line rows, which are unknown-length vectors. 
          // The last entry is the source row size, which is a scalar.
          (Shape(-1), Shape.scalar()), 
          // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
          // not generally need to do this since later on we will be masking out calculations past 
          // the true sequence.
          (srcEosId, tf.zeros(INT32, Shape.scalar())))
  }

  def createTrainDataset(
      srcDataset: tf.data.Dataset[Tensor, Output, DataType, Shape],
      tgtDataset: tf.data.Dataset[Tensor, Output, DataType, Shape],
      srcVocabularyTable: tf.LookupTable,
      tgtVocabularyTable: tf.LookupTable,
      batchSize: Int,
      beginSequenceToken: String = Vocabulary.BEGIN_SEQUENCE_TOKEN,
      endSequenceToken: String = Vocabulary.END_SEQUENCE_TOKEN,
      srcReverse: Boolean = false,
      randomSeed: Option[Int] = None,
      numBuckets: Int = 1,
      srcMaxLength: Int = -1,
      tgtMaxLength: Int = -1,
      numParallelCalls: Int = 4,
      outputBufferSize: Long = -1L,
      dropCount: Int = 0,
      numShards: Int = 1,
      shardIndex: Int = 0
  ): NMTTrainDataset = {
    val bufferSize = if (outputBufferSize == -1L) 1000 * batchSize else outputBufferSize
    val srcEosId = srcVocabularyTable.lookup(tf.constant(endSequenceToken)).cast(INT32)
    val tgtBosId = tgtVocabularyTable.lookup(tf.constant(beginSequenceToken)).cast(INT32)
    val tgtEosId = tgtVocabularyTable.lookup(tf.constant(endSequenceToken)).cast(INT32)

    val batchingFn = (dataset: NMTTrainDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first three entries are the source and target line rows, which are unknown-length vectors. 
        // The last two entries are the source and target row sizes, which are scalars.
        (Shape(-1), Shape(-1), Shape(-1), Shape.scalar(), Shape.scalar()),
        // We pad the source and target sequences with 'endSequenceToken' tokens. Though notice that we do not 
        // generally need to do this since later on we will be masking out calculations past the true sequence.
        (srcEosId, tgtEosId, tgtEosId, tf.zeros(INT32, Shape.scalar()), tf.zeros(INT32, Shape.scalar())))
    }

    val datasetBeforeBucketing =
      srcDataset.zip(tgtDataset)
          .shard(numShards, shardIndex)
          .drop(dropCount)
          .shuffle(bufferSize, randomSeed)
          // Tokenize by splitting on white spaces.
          .map(d => (tf.stringSplit(d._1).values, tf.stringSplit(d._2).values))
          .prefetch(bufferSize)
          // Filter zero length input sequences and sequences exceeding the maximum length.
          .filter(d => tf.logicalAnd(tf.size(d._1) > 0, tf.size(d._2) > 0))
          // Crop based on the maximum allowed sequence lengths.
          .transform(d => {
            if (srcMaxLength != -1 && tgtMaxLength != -1)
              d.map(dd => (dd._1(0 :: srcMaxLength), dd._2(0 :: tgtMaxLength)), numParallelCalls).prefetch(bufferSize)
            else if (srcMaxLength != -1)
              d.map(dd => (dd._1(0 :: srcMaxLength), dd._2), numParallelCalls).prefetch(bufferSize)
            else if (tgtMaxLength != -1)
              d.map(dd => (dd._1, dd._2(0 :: tgtMaxLength)), numParallelCalls).prefetch(bufferSize)
            else
              d
          })
          // Reverse the source sequence if necessary.
          .transform(d => {
            if (srcReverse)
              d.map(dd => (tf.reverse(dd._1, axes = 0), dd._2)).prefetch(bufferSize)
            else
              d
          })
          // Convert the word strings to IDs. Word strings that are not in the vocabulary
          // get the lookup table's default value.
          .map(d => (
              tf.cast(srcVocabularyTable.lookup(d._1), INT32),
              tf.cast(tgtVocabularyTable.lookup(d._2), INT32)),
            numParallelCalls).prefetch(bufferSize)
          // Create a target input prefixed with 'beginSequenceToken'
          // and a target output suffixed with 'endSequenceToken'.
          .map(d => (
              d._1,
              tf.concatenate(Seq(tgtBosId.expandDims(0), d._2), axis = 0),
              tf.concatenate(Seq(d._2, tgtEosId.expandDims(0)), axis = 0)),
            numParallelCalls).prefetch(bufferSize)
          // Add sequence lengths.
          .map(d => (d._1, d._2, d._3, tf.size(d._1, INT32), tf.size(d._2, INT32)), numParallelCalls)
          .prefetch(bufferSize)

    if (numBuckets == 1) {
      batchingFn(datasetBeforeBucketing)
    } else {
      // Calculate the bucket width by using the maximum source sequence length, if provided. Pairs with length
      // [0, bucketWidth) go to bucket 0, length [bucketWidth, 2 * bucketWidth) go to bucket 1, etc. Pairs with length
      // over ((numBuckets - 1) * bucketWidth) all go into the last bucket.
      val bucketWidth = if (srcMaxLength != -1) (srcMaxLength + numBuckets - 1) / numBuckets else 10

      def keyFn(element: (Output, Output, Output, Output, Output)): Output = {
        // Bucket sequence  pairs based on the length of their source sequence and target sequence.
        val bucketId = tf.maximum(
          tf.truncateDivide(element._4, bucketWidth),
          tf.truncateDivide(element._5, bucketWidth))
        tf.minimum(numBuckets, bucketId).cast(INT64)
      }

      def reduceFn(pair: (Output, NMTTrainDataset)): NMTTrainDataset = {
        batchingFn(pair._2)
      }

      datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
    }
  }
}
