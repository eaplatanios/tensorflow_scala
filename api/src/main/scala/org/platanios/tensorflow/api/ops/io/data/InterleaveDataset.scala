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

import org.platanios.tensorflow.api.implicits.helpers.OutputToTensor
import org.platanios.tensorflow.api.ops.{Function, Op, Output}
import org.platanios.tensorflow.api.types.INT64

/** Dataset that wraps the application of the `interleave` op.
  *
  * $OpDocDatasetInterleave
  *
  * @param  inputDataset Input dataset.
  * @param  function     Mapping function.
  * @param  cycleLength  Number of elements from the input dataset that will be processed concurrently.
  * @param  blockLength  Number of consecutive elements to produce from each input element before cycling to another
  *                      input element.
  * @param  name         Name for this dataset.
  * @tparam T            Tensor type (i.e., nested structure of tensors).
  * @tparam O            Output type (i.e., nested structure of symbolic tensors).
  * @tparam D            Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S            Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class InterleaveDataset[T, O, D, S, RT, RO, RD, RS](
    inputDataset: Dataset[T, O, D, S],
    function: (O) => Dataset[RT, RO, RD, RS],
    cycleLength: Output,
    blockLength: Output = 1,
    override val name: String = "InterleaveDataset"
)(implicit
    evOToT: OutputToTensor.Aux[O, T] = inputDataset.evOToT,
    evData: Data.Aux[T, O, D, S] = inputDataset.evData,
    evFunctionInput: Function.ArgType[O] = inputDataset.evFunctionInput,
    evROToRT: OutputToTensor.Aux[RO, RT],
    evRData: Data.Aux[RT, RO, RD, RS],
    evFunctionOutput: Function.ArgType[RO]
) extends Dataset[RT, RO, RD, RS](name)(evROToRT, evRData, evFunctionOutput) {
  private[this] lazy val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes)
  }

  override def createHandle(): Output = {
    Op.Builder(opType = "InterleaveDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedFunction.extraInputs)
        .addInput(cycleLength.cast(INT64))
        .addInput(blockLength.cast(INT64))
        .setAttribute("f", instantiatedFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: RD = instantiatedFunction.dummyOutputs.outputDataTypes
  override def outputShapes: RS = instantiatedFunction.dummyOutputs.outputShapes
}

/** Dataset that wraps the application of the `parallelInterleave` op.
  *
  * $OpDocDatasetParallelInterleave
  *
  * @param  inputDataset          Input dataset.
  * @param  function              Mapping function.
  * @param  cycleLength           Number of elements from the input dataset that will be processed concurrently.
  * @param  blockLength           Number of consecutive elements to produce from each input element before cycling to
  *                               another input element.
  * @param  sloppy                If `false`, elements are produced in deterministic order. Otherwise, the
  *                               implementation is allowed, for the sake of expediency, to produce elements in a
  *                               non-deterministic order.
  * @param  bufferOutputElements  Number of elements each iterator being interleaved should buffer (similar to the
  *                               `prefetch(...)` transformation for each interleaved iterator).
  * @param  prefetchInputElements Number of input elements to transform to iterators before they are needed for
  *                               interleaving.
  * @param  name                  Name for this dataset.
  * @tparam T                     Tensor type (i.e., nested structure of tensors).
  * @tparam O                     Output type (i.e., nested structure of symbolic tensors).
  * @tparam D                     Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S                     Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
case class ParallelInterleaveDataset[T, O, D, S, RT, RO, RD, RS](
    inputDataset: Dataset[T, O, D, S],
    function: (O) => Dataset[RT, RO, RD, RS],
    cycleLength: Output,
    blockLength: Output = 1,
    sloppy: Boolean = false,
    bufferOutputElements: Output = null,
    prefetchInputElements: Output = null,
    override val name: String = "ParallelInterleaveDataset"
)(implicit
    evOToT: OutputToTensor.Aux[O, T] = inputDataset.evOToT,
    evData: Data.Aux[T, O, D, S] = inputDataset.evData,
    evFunctionInput: Function.ArgType[O] = inputDataset.evFunctionInput,
    evROToRT: OutputToTensor.Aux[RO, RT],
    evRData: Data.Aux[RT, RO, RD, RS],
    evFunctionOutput: Function.ArgType[RO]
) extends Dataset[RT, RO, RD, RS](name)(evROToRT, evRData, evFunctionOutput) {
  private[this] lazy val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(
      inputDataset.flattenedOutputDataTypes, inputDataset.flattenedOutputShapes)
  }

  override def createHandle(): Output = {
    val bufferOutputElements = if (this.bufferOutputElements == null) 2 * blockLength else this.bufferOutputElements
    val prefetchInputElements = if (this.prefetchInputElements == null) 2 * cycleLength else this.prefetchInputElements
    Op.Builder(opType = "ParallelInterleaveDataset", name = name)
        .addInput(Op.createWithNameScope(name)(inputDataset.createHandle()))
        .addInputList(instantiatedFunction.extraInputs)
        .addInput(cycleLength.cast(INT64))
        .addInput(blockLength.cast(INT64))
        .addInput(sloppy)
        .addInput(bufferOutputElements.cast(INT64))
        .addInput(prefetchInputElements.cast(INT64))
        .setAttribute("f", instantiatedFunction)
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: RD = instantiatedFunction.dummyOutputs.outputDataTypes
  override def outputShapes: RS = instantiatedFunction.dummyOutputs.outputShapes
}

object InterleaveDataset {
  case class InterleaveDatasetOps[T, O, D, S](dataset: Dataset[T, O, D, S]) {
    /** $OpDocDatasetInterleave
      *
      * @param  function    Mapping function.
      * @param  cycleLength Number of elements from this dataset that will be processed concurrently.
      * @param  blockLength Number of consecutive elements to produce from each input element before cycling to another
      *                     input element.
      * @param  name        Name for the created dataset.
      * @return Created dataset.
      */
    def interleave[RT, RO, RD, RS](
        function: (O) => Dataset[RT, RO, RD, RS],
        cycleLength: Output,
        blockLength: Output = 1,
        name: String = "Interleave"
    )(implicit
        evROToRT: OutputToTensor.Aux[RO, RT],
        evRData: Data.Aux[RT, RO, RD, RS],
        evFunctionOutput: Function.ArgType[RO]
    ): Dataset[RT, RO, RD, RS] = {
      Op.createWithNameScope(dataset.name) {
        InterleaveDataset(dataset, function, cycleLength, blockLength, name)
      }
    }

    /** $OpDocDatasetParallelInterleave
      *
      * @param  function              Mapping function.
      * @param  cycleLength           Number of elements from this dataset that will be processed concurrently.
      * @param  blockLength           Number of consecutive elements to produce from each input element before cycling
      *                               to another input element.
      * @param  sloppy                If `false`, elements are produced in deterministic order. Otherwise, the
      *                               implementation is allowed, for the sake of expediency, to produce elements in a
      *                               non-deterministic order.
      * @param  bufferOutputElements  Number of elements each iterator being interleaved should buffer (similar to the
      *                               `prefetch(...)` transformation for each interleaved iterator).
      * @param  prefetchInputElements Number of input elements to transform to iterators before they are needed for
      *                               interleaving.
      * @param  name                  Name for the created dataset.
      * @return Created dataset.
      */
    def parallelInterleave[RT, RO, RD, RS](
        function: (O) => Dataset[RT, RO, RD, RS],
        cycleLength: Output,
        blockLength: Output = 1,
        sloppy: Boolean = false,
        bufferOutputElements: Output = null,
        prefetchInputElements: Output = null,
        name: String = "ParallelInterleave"
    )(implicit
        evROToRT: OutputToTensor.Aux[RO, RT],
        evRData: Data.Aux[RT, RO, RD, RS],
        evFunctionOutput: Function.ArgType[RO]
    ): Dataset[RT, RO, RD, RS] = {
      Op.createWithNameScope(dataset.name) {
        ParallelInterleaveDataset(
          dataset, function, cycleLength, blockLength, sloppy, bufferOutputElements, prefetchInputElements, name)
      }
    }
  }

  /** @define OpDocDatasetInterleave
    *       The dataset `interleave` op creates a new dataset by a mapping function over its input and interleaving the
    *       result.
    *
    *       For example, you can use `Dataset.interleave()` to process many input files concurrently:
    *       {{{
    *              // Preprocess 4 files concurrently, and interleave blocks of 16 records from each file.
    *              val filenames = Tensor("/var/data/file1.txt", "/var/data/file2.txt", ...)
    *              val dataset = TensorSlicesDataset(filenames).interleave(
    *                d => TextLinesDataset(d).map(parseFn), cycleLength = 4, blockLength = 16)
    *       }}}
    *
    *       The `cycleLength` and `blockLength` arguments control the order in which elements are produced. `cycleLength`
    *       controls the number of input elements that are processed concurrently. If you set `cycleLength` to `1`, this
    *       transformation will handle one input element at a time, and produce identical results to `flatMap(...)`. In
    *       general, this transformation will apply `function` to `cycleLength` input elements, open iterators on the
    *       returned dataset objects, and cycle through them producing `blockLength` consecutive elements from each
    *       iterator, and consuming the next input element each time it reaches the end of an iterator.
    *
    *       For example:
    *       {{{
    *              // The following examples use `{ ... }` to represent the contents of a dataset,
    *              // and new lines indicate "block" boundaries.
    *              a = { 1, 2, 3, 4, 5 }
    *              a.interleave(d => TensorDataset(d).repeat(6), cycleLength = 2, blockLength = 4) == {
    *                1, 1, 1, 1,
    *                2, 2, 2, 2,
    *                1, 1,
    *                2, 2,
    *                3, 3, 3, 3,
    *                4, 4, 4, 4,
    *                3, 3,
    *                4, 4,
    *                5, 5, 5, 5,
    *                5, 5
    *              }
    *       }}}
    *
    *       Note that the order of elements yielded by this transformation is deterministic, as long as `function` is a
    *       pure function. If `function` contains any stateful operations, the order in which that state is accessed is
    *       undefined.
    *
    * @define OpDocDatasetParallelInterleave
    *       The dataset `parallelInterleave` op is a parallel version of the `interleave` op.
    *
    *       This op maps `function` across its input to produce nested datasets, and outputs their elements interleaved.
    *       Unlike the `interleave` op, it gets elements from `cycleLength` nested datasets in parallel, which increases
    *       the throughput, especially in the presence of stragglers. Furthermore, the `sloppy` argument can be used to
    *       improve performance, by relaxing the requirement that the outputs are produced in a deterministic order, and
    *       allowing the implementation to skip over nested datasets whose elements are not readily available when
    *       requested.
    *
    *       Note that, if `sloppy` is `true`, the order of produced elements is not deterministic.
    */
  private[data] trait Documentation
}
