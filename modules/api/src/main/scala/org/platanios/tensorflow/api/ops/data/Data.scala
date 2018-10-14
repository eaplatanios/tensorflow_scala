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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.core.types.{INT64, STRING, Variant}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.io.{CompressionType, NoCompression}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.language.postfixOps

// TODO: [DATA] Separate into readers and transformations.
// TODO: [DATA] paddedBatchAndDropRemainder
// TODO: [DATA] denseToSparseBatch
// TODO: [DATA] listFiles

/**
  * @author Emmanouil Antonios Platanios
  */
trait Data extends Experimental {
  //region Dataset Constructors

  /** Creates a dataset with a single element.
    *
    * @param  data Data representing the single element.
    * @param  name Name for this dataset.
    * @tparam T Symbolic tensor type of the element (symbolic equivalent of `V`).
    * @tparam V Value tensor type of the element.
    * @tparam D Data type of the element.
    * @tparam S Shape of the element.
    * @return Created dataset.
    */
  def datasetFromTensors[T, V, D, S](
      data: V,
      name: String = "TensorDataset"
  )(implicit evStructureT: NestedStructure.Aux[T, V, D, S]): Dataset[T] = {
    val datasetName = name
    new Dataset[T] {
      override val name: String = datasetName

      protected val providedEvStructureT: NestedStructure.Aux[T, V, D, S] = evStructureT

      override def createHandle[VV, DD, SS]()(implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): Output[Variant] = {
        val outputs = Op.nameScope(s"$name/TensorToOutput") {
          providedEvStructureT.outputFromTensor(data)
        }
        val flatOutputs = evStructureT.outputs(outputs)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[VV, DD, SS](implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): DD = {
        providedEvStructureT.dataTypeFromTensor(data).asInstanceOf[DD]
      }

      override def outputShapes[VV, DD, SS](implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): SS = {
        providedEvStructureT.shapeFromTensor(data).asInstanceOf[SS]
      }
    }
  }

  /** Creates a dataset with a single element.
    *
    * @param  data   Data representing the single element.
    * @param  name   Name for this dataset.
    * @tparam T Tensor type of the element.
    * @return Created dataset.
    */
  def datasetFromOutputs[T](
      data: T,
      name: String = "TensorDataset"
  ): Dataset[T] = {
    val datasetName = name
    new Dataset[T] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evT: NestedStructure.Aux[T, V, D, S]): Output[Variant] = {
        val flatOutputs = evT.outputs(data)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): D = {
        evT.dataTypeFromOutput(data)
      }

      override def outputShapes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): S = {
        evT.shapeFromOutput(data)
      }
    }
  }

  /** Creates a dataset with slices from the nested structure of tensors (i.e., a [[NestedStructure]]-supported type).
    * The slices are taken along the first axis of each tensor in the nested structure.
    *
    * @param  data Data representing the elements of this dataset.
    * @param  name Name for this dataset.
    * @tparam T Symbolic tensor type of the element (symbolic equivalent of `V`).
    * @tparam V Value tensor type of the element.
    * @tparam D Data type of the element.
    * @tparam S Shape of the element.
    * @return Created dataset.
    */
  def datasetFromTensorSlices[T, V, D, S](
      data: V,
      name: String = "TensorSlicesDataset"
  )(implicit evStructureT: NestedStructure.Aux[T, V, D, S]): Dataset[T] = {
    val datasetName = name
    new Dataset[T] {
      override val name: String = datasetName

      protected val providedEvStructureT: NestedStructure.Aux[T, V, D, S] = evStructureT

      override def createHandle[VV, DD, SS]()(implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): Output[Variant] = {
        val outputs = Op.nameScope(s"$name/TensorToOutput") {
          providedEvStructureT.outputFromTensor(data)
        }
        val flatOutputs = evStructureT.outputs(outputs)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorSliceDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[VV, DD, SS](implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): DD = {
        providedEvStructureT.dataTypeFromTensor(data).asInstanceOf[DD]
      }

      override def outputShapes[VV, DD, SS](implicit evStructureT: NestedStructure.Aux[T, VV, DD, SS]): SS = {
        val flatShapes = providedEvStructureT.shapes(providedEvStructureT.shapeFromTensor(data))
        evStructureT.decodeShapeFromDataType(
          outputDataTypes,
          flatShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))._1
      }
    }
  }

  /** Creates a dataset with slices from the nested structure of tensors (i.e., a [[NestedStructure]]-supported type).
    * The slices are taken along the first axis of each tensor in the nested structure.
    *
    * @param  data   Data representing the elements of this dataset.
    * @param  name   Name for this dataset.
    * @tparam T Tensor type of the element.
    * @return Created dataset.
    */
  def datasetFromOutputSlices[T](
      data: T,
      name: String = "TensorSlicesDataset"
  ): Dataset[T] = {
    val datasetName = name
    new Dataset[T] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evT: NestedStructure.Aux[T, V, D, S]): Output[Variant] = {
        val flatOutputs = evT.outputs(data)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorSliceDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): D = {
        evT.dataTypeFromOutput(data)
      }

      override def outputShapes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): S = {
        val flatShapes = evT.shapes(evT.shapeFromOutput(data))
        evT.decodeShapeFromDataType(
          outputDataTypes,
          flatShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))._1
      }
    }
  }

  // TODO: [DATA] Add dynamic version (i.e., passing in `Output`s) for the `fromRange`, `fromFixedLengthRecordFiles`, and `fromTextFiles` datasets.

  /** Creates a new dataset that contains a range of values.
    *
    * @param  start Starting value of the number sequence.
    * @param  limit Ending value (exclusive) of the number sequence.
    * @param  delta Difference between consecutive numbers in the sequence.
    * @param  name  Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromRange(
      start: Long,
      limit: Long,
      delta: Long = 1L,
      name: String = "RangeDataset"
  ): Dataset[Output[Long]] = {
    val datasetName = name
    new Dataset[Output[Long]] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): Output[Variant] = {
        Op.Builder[(Output[Long], Output[Long], Output[Long]), Output[Variant]](
          opType = "RangeDataset",
          name = name,
          input = (
              Basic.constant(start, name = s"$name/Start"),
              Basic.constant(limit, name = s"$name/Limit"),
              Basic.constant(delta, name = s"$name/Delta"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): D = {
        INT64.asInstanceOf[D]
      }

      override def outputShapes[V, D, S](implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): S = {
        Shape().asInstanceOf[S]
      }
    }
  }

  /** Creates a new dataset that contains pseudorandom integers.
    *
    * @param  seed Optional random seed, used to generate a random seed pair for the random number generator, when
    *              combined with the graph-level seed.
    * @param  name Name for this dataset.
    * @return Created dataset.
    */
  def randomDataset(
      seed: Option[Int] = None,
      name: String = "RandomDataset"
  ): Dataset[Output[Long]] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val datasetName = name
    new Dataset[Output[Long]] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): Output[Variant] = {
        Op.Builder[(Output[Long], Output[Long]), Output[Variant]](
          opType = "RandomDataset",
          name = name,
          input = (
              Basic.constant(graphSeed.getOrElse(0L), name = s"$name/Seed1"),
              Basic.constant(opSeed.getOrElse(0L), name = s"$name/Seed2"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): D = {
        INT64.asInstanceOf[D]
      }

      override def outputShapes[V, D, S](implicit evOutputLong: NestedStructure.Aux[Output[Long], V, D, S]): S = {
        Shape().asInstanceOf[S]
      }
    }
  }

  /** Creates a dataset with elements read from binary files.
    *
    * @param  filenames      Names of the files to be read.
    * @param  recordNumBytes Number of bytes in the record.
    * @param  headerNumBytes Number of bytes in the header (i.e., the number of bytes to skip at the start of a file).
    * @param  footerNumBytes Number of bytes in the footer (i.e., the number of bytes to skip at the end of a file).
    * @param  bufferSize     Number of bytes to buffer while reading from files.
    * @param  name           Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromFixedLengthRecordFiles(
      filenames: Seq[String],
      recordNumBytes: Long,
      headerNumBytes: Long,
      footerNumBytes: Long,
      bufferSize: Long = 256 * 1024,
      name: String = "FixedLengthRecordDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): Output[Variant] = {
        Op.Builder[(Output[String], Output[Long], Output[Long], Output[Long], Output[Long]), Output[Variant]](
          opType = "FixedLengthRecordDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(recordNumBytes, name = s"$name/RecordNumBytes"),
              Basic.constant(headerNumBytes, name = s"$name/HeaderNumBytes"),
              Basic.constant(footerNumBytes, name = s"$name/FooterNumBytes"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): D = {
        STRING.asInstanceOf[D]
      }

      override def outputShapes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): S = {
        Shape().asInstanceOf[S]
      }
    }
  }

  /** Creates a dataset with elements read from text files (each line in each file corresponds to an element).
    *
    * **Note:** New-line characters are stripped from the output.
    *
    * @param  filenames       Names of the files to be read.
    * @param  compressionType Compression type for the files.
    * @param  bufferSize      Number of bytes to buffer while reading from files.
    * @param  name            Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromTextFiles(
      filenames: Seq[String],
      compressionType: CompressionType = NoCompression,
      bufferSize: Long = 256 * 1024,
      name: String = "TextLinesDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): Output[Variant] = {
        Op.Builder[(Output[String], Output[String], Output[Long]), Output[Variant]](
          opType = "TextLineDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(compressionType.name, name = s"$name/CompressionType"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): D = {
        STRING.asInstanceOf[D]
      }

      override def outputShapes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): S = {
        Shape().asInstanceOf[S]
      }
    }
  }

  /** Creates a dataset with elements read from files that contain TensorFlow records.
    *
    * @param  filenames       Names of the files to be read.
    * @param  compressionType Compression type for the files.
    * @param  bufferSize      Number of bytes to buffer while reading from files.
    * @param  name            Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromTFRecordFiles(
      filenames: Seq[String],
      compressionType: CompressionType = NoCompression,
      bufferSize: Long = 256 * 1024,
      name: String = "TFRecordsDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name: String = datasetName

      override def createHandle[V, D, S]()(implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): Output[Variant] = {
        Op.Builder[(Output[String], Output[String], Output[Long]), Output[Variant]](
          opType = "TFRecordDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(compressionType.name, name = s"$name/CompressionType"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): D = {
        STRING.asInstanceOf[D]
      }

      override def outputShapes[V, D, S](implicit evOutputString: NestedStructure.Aux[Output[String], V, D, S]): S = {
        Shape().asInstanceOf[S]
      }
    }
  }

  /** Stores outstanding iterators created from a Scala iterable.
    *
    * This class keeps track of potentially multiple iterators that may have been created from an iterable, e.g., in the
    * case that the dataset is repeated, or nested within a parallel computation.
    *
    * @param  generator Function that generates an iterable containing dataset elements.
    */
  private case class GeneratorState[T](generator: () => Iterable[T]) {
    private val _nextId   = new AtomicLong(0)
    private val iterators = mutable.Map.empty[Long, scala.Iterator[T]]

    private[Data] def nextId: Long = {
      _nextId.getAndIncrement()
    }

    private[Data] def getIterator(id: Long): scala.Iterator[T] = {
      iterators.getOrElseUpdate(id, generator().iterator)
    }

    private[Data] def deleteIterator(id: Long): Unit = {
      iterators.remove(id)
    }
  }

  /** Creates a dataset whose elements are generated by Scala iterables over dataset elements.
    *
    * The `generator` argument must be a function that takes no arguments and returns an iterable over dataset
    * elements. The elements contained in that iterable must be compatible with the provided `outputDataType` and
    * `outputShape` arguments.
    *
    * For example:
    * {{{
    *   // TODO: [DATA] !!! Improve this example with variable shapes -- like in the Python API.
    *   val generator = () => Range(0, 10).map(Tensor(_))
    *   val dataset = Dataset.fromGenerator(generator, outputDataType = Int, outputShape = Shape.scalar())
    *   val value = dataset.createOneShotIterator().next()
    *   session.run(value) ==> 0
    *   session.run(value) ==> 1
    * }}}
    *
    * @param  generator      Function that takes no arguments and returns an [[Iterable]] over dataset elements.
    * @param  outputDataType Output data type structure for the tensor structure of the generated [[Iterable]] elements.
    * @param  outputShape    Output shape structure for the tensor structure of the generated [[Iterable]] elements.
    * @tparam T Symbolic tensor type of the element (symbolic equivalent of `V`).
    * @tparam V Value tensor type of the element.
    * @tparam D Data type of the element.
    * @tparam S Shape of the element.
    * @return Constructed dataset.
    */
  def datasetFromGenerator[T, V, D, S](
      generator: () => Iterable[V],
      outputDataType: D,
      outputShape: S = null
  )(implicit
      evStructure: NestedStructure.Aux[T, V, D, S]
  ): Dataset[T] = {
    val outputShapeWithDefault: S = {
      if (outputShape != null) {
        outputShape
      } else {
        evStructure.decodeShapeFromDataType(
          outputDataType,
          Seq.fill(evStructure.sizeFromDataType(outputDataType))(Shape.unknown()))._1
      }
    }

    val flatDataTypes = evStructure.dataTypes(outputDataType)
    val flatShapes = evStructure.shapes(outputShapeWithDefault)
    val generatorState = GeneratorState(generator)

    /** Creates an op that generates the next element from iterator with ID, `iteratorId`.
      *
      * We map this function across an infinite repetition of the `iteratorId`, and throw an `OutOfRange` to terminate
      * the iteration.
      *
      * @param  iteratorId Scalar tensor whose value uniquely identifies the iterator in the internal
      *                    generator state, from which to generate an element.
      * @return Created op outputs structured according to the output data type of this dataset.
      */
    def generatorMapFn(iteratorId: Output[Long]): T = {
      /** Scala callback function that will be called to invoke the iterator. */
      @throws[OutOfRangeException]
      def generatorScalaCallback(iteratorId: Tensor[Long]): Seq[Tensor[Any]] = {
        val iterator = generatorState.getIterator(iteratorId.scalar)
        val element = {
          if (iterator.hasNext)
            iterator.next()
          else
            throw OutOfRangeException("The iterator does not contain any more elements.")
        }
        val flatTensors = evStructure.tensors(element)
        // Additional type and shape checking to ensure that the components of the generated element match the
        // output data types and output shapes arguments.
        (flatTensors, flatDataTypes, flatShapes).zipped.foreach((tensor, dataType, shape) => {
          if (tensor.dataType != dataType)
            throw InvalidDataTypeException(
              s"The generator yielded an element of type ${tensor.dataType} " +
                  s"where an element of type $dataType was expected.")
          if (!tensor.shape.isCompatibleWith(shape))
            throw InvalidShapeException(
              s"The generator yielded an element with shape ${tensor.shape} " +
                  s"where an element with shape $shape was expected.")
        })
        flatTensors
      }

      val flatValues = Callback.callback(generatorScalaCallback, iteratorId, flatDataTypes, stateful = true)
      // The Scala callback op drops the inferred shapes, so we add them back in here.
      if (outputShape != null) {
        flatValues.zip(flatShapes).foreach(p => {
          if (!p._1.shape.isCompatibleWith(p._2)) {
            throw new IllegalArgumentException(
              s"Generator output shape ${p._1.shape} is not compatible with provided shape ${p._2}.")
          } else {
            p._1.setShape(p._2)
          }
        })
      }
      evStructure.decodeOutputFromDataType(outputDataType, flatValues)._1
    }

    /** Associates each traversal of the provided `generator` with a unique iterator ID. */
    def flatMapFn(iteratorId: Output[Long]): Dataset[T] = {
      // First, generate an infinite dataset containing the iterator ID repeated forever. Then, map using the
      // `generatorMapFn`, which gets the next element from the iterator with the relevant ID, and throws an
      // IndexOutOfBoundsException when that iterator contains no more elements.
      datasetFromOutputs(iteratorId).repeat().map(generatorMapFn)
    }

    // A single-element dataset that, each time it is evaluated, contains a freshly-generated and unique (for the
    // returned dataset) INT64 ID that will be used to identify the appropriate Scala state, which is encapsulated in
    // the internal generator state, and captured in the provided callback function. The ID disambiguates between
    // multiple concurrently existing iterators.
    val idDataset = datasetFromTensors(Tensor(0L)).map((_: Output[Long]) => {
      Callback.callback((_: Unit) => {
        Tensor(generatorState.nextId)
      }, (), INT64, stateful = true)
    })

    // A dataset that contains all of the elements generated by a single iterator created from the provided generator,
    // identified by the iterator ID contained in `idDataset`. Lifting the iteration into a `flatMap` here enables
    // multiple repetitions and/or nested versions of the returned dataset to be created, because it forces the
    // generation of a new ID for each version.
    idDataset.flatMap(flatMapFn)
  }

  //endregion Dataset Constructors
}

object Data extends Data
