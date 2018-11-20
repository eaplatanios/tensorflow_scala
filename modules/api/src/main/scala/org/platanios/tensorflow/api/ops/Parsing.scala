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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor}

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.immutable.ListMap
import scala.collection.mutable

/** Contains functions for constructing ops related to parsing data.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Parsing {
  /** $OpDocParsingEncodeTensor
    *
    * @group ParsingOps
    * @param  tensor Tensor to encode.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def encodeTensor[T: TF](
      tensor: Output[T],
      name: String = "EncodeTensor"
  ): Output[String] = {
    Op.Builder[Output[T], Output[String]](
      opType = "SerializeTensor",
      name = name,
      input = tensor
    ).build().output
  }

  /** $OpDocParsingDecodeTensor
    *
    * @group ParsingOps
    * @param  data Tensor containing a serialized `TensorProto` proto.
    * @param  name Name for the created op.
    * @tparam T Data type of the serialized tensor. The provided data type must match the data type of the
    *           serialized tensor and no implicit conversion will take place.
    * @return Created op output.
    */
  def decodeTensor[T: TF](
      data: Output[String],
      name: String = "DecodeTensor"
  ): Output[T] = {
    Op.Builder[Output[String], Output[T]](
      opType = "ParseTensor",
      name = name,
      input = data
    ).setAttribute("out_type", TF[T].dataType)
        .build().output
  }

  /** $OpDocParsingDecodeRaw
    *
    * @group ParsingOps
    * @param  bytes        Tensor interpreted as raw bytes. All the elements must have the same length.
    * @param  littleEndian Boolean value indicating whether the input `bytes` are stored in little-endian order. Ignored
    *                      for `dataType` values that are stored in a single byte, like [[UINT8]].
    * @param  name         Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def decodeRaw[T: TF](
      bytes: Output[String],
      littleEndian: Boolean = true,
      name: String = "DecodeRaw"
  ): Output[T] = {
    Op.Builder[Output[String], Output[T]](
      opType = "DecodeRaw",
      name = name,
      input = bytes
    ).setAttribute("out_type", TF[T].dataType)
        .setAttribute("little_endian", littleEndian)
        .build().output
  }

  // TODO: [OPS|TYPES|PARSING] Use tuples instead of sequences for parsing CSVs.

  /** $OpDocParsingDecodeCSV
    *
    * @group ParsingOps
    * @param  records            Tensor where each string is a record/row in the csv and all records should
    *                            have the same format.
    * @param  recordDefaults     One tensor per column of the input record, with either a scalar default value for that
    *                            column or empty if the column is required.
    * @param  dataTypes          Output tensor data types.
    * @param  delimiter          Delimiter used to separate fields in a record.
    * @param  useQuoteDelimiters If `false`, the op treats double quotation marks as regular characters inside the
    *                            string fields (ignoring RFC 4180, Section 2, Bullet 5).
    * @param  name               Name for the created op.
    * @return Created op outputs.
    */
  def decodeCSV[T: TF](
      records: Output[String],
      recordDefaults: Seq[Output[T]],
      dataTypes: Seq[DataType[T]],
      delimiter: String = ",",
      useQuoteDelimiters: Boolean = true,
      name: String = "DecodeCSV"
  ): Seq[Output[T]] = {
    Op.Builder[(Output[String], Seq[Output[T]]), Seq[Output[T]]](
      opType = "DecodeCSV",
      name = name,
      input = (records, recordDefaults)
    ).setAttribute("OUT_TYPE", dataTypes.map(_.asInstanceOf[DataType[Any]]).toArray[DataType[Any]])
        .setAttribute("field_delim", delimiter)
        .setAttribute("use_quote_delim", useQuoteDelimiters)
        .build().output
  }

  /** $OpDocParsingStringToNumber
    *
    * @group ParsingOps
    * @param  data Tensor containing string representations of numbers.
    * @param  name Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def stringToNumber[T: TF](
      data: Output[String],
      name: String = "StringToNumber"
  ): Output[T] = {
    Op.Builder[Output[String], Output[T]](
      opType = "StringToNumber",
      name = name,
      input = data
    ).setAttribute("out_type", TF[T].dataType)
        .build().output
  }

  /** $OpDocParsingDecodeJSONExample
    *
    * @group ParsingOps
    * @param  jsonExamples Tensor where each string is a JSON object serialized according to the JSON mapping
    *                      of the `Example` proto.
    * @param  name         Name for the created op.
    * @return Created op output.
    */
  def decodeJSONExample(
      jsonExamples: Output[String],
      name: String = "DecodeJSONExample"
  ): Output[String] = {
    Op.Builder[Output[String], Output[String]](
      opType = "DecodeJSONExample",
      name = name,
      input = jsonExamples
    ).build().output
  }

  // TODO: [OPS] [PARSING] Document the following two methods.
  // TODO: [OPS] [PARSING] Add support for parsing `SequenceExample`s.

  @throws[InvalidArgumentException]
  def parseExample[T, R](
      serialized: Output[String],
      features: T,
      debugNames: Output[String] = "",
      name: String = "ParseExample"
  )(implicit ev: Parsing.Features.Aux[T, R]): R = {
    Op.nameScope(name) {
      val rawParameters = ev.toRawParameters(features)
      if (rawParameters.sparseKeys.isEmpty && rawParameters.denseKeys.isEmpty)
        throw InvalidArgumentException("Must provide at least one dense or sparse feature.")

      val output = Op.Builder[(Output[String], Output[String], Seq[Output[String]], Seq[Output[String]], Seq[Output[Any]]), Seq[Output[Any]]](
        opType = "ParseExample",
        name = name,
        input = (
            serialized,
            debugNames,
            rawParameters.sparseKeys.map(key => Output.constant[String](key, name = s"SparseKeys/$key")),
            rawParameters.denseKeys.map(key => Output.constant[String](key, name = s"DenseKeys/$key")),
            rawParameters.denseDefaults.toSeq.map(default => {

              Output.constant(
                value = default._2,
                name = s"Padding${Op.normalizeNameScope(default._1)}")(TF.fromDataType(default._2.dataType))
            }))
      ).setAttribute("sparse_types", rawParameters.sparseTypes.toArray)
          .setAttribute("Tdense", rawParameters.denseTypes.toArray)
          .setAttribute("dense_shapes", rawParameters.denseShapes.toArray)
          .build().output

      val (sparseIndices, sparseValues, sparseShapes, denseValues) = {
        val numSparse = rawParameters.sparseKeys.size
        (output.take(numSparse).map(_.asInstanceOf[Output[Long]]),
            output.slice(numSparse, 2 * numSparse),
            output.slice(2 * numSparse, 3 * numSparse).map(_.asInstanceOf[Output[Long]]),
            output.drop(3 * numSparse))
      }

      val sparseComposedValues = (sparseIndices, sparseValues, sparseShapes).zipped.map(SparseOutput(_, _, _))
      val sparseParsed = rawParameters.sparseKeys.zip(sparseComposedValues).toMap
      val denseParsed = rawParameters.denseKeys.zip(denseValues).toMap
      ev.fromParsed(features, sparseParsed, denseParsed)
    }
  }

  @throws[InvalidArgumentException]
  def parseSingleExample[T, R](
      serialized: Output[String],
      features: T,
      name: String = "ParseSingleExample"
  )(implicit ev: Parsing.Features.Aux[T, R]): R = {
    Op.nameScope(name) {
      val rawParameters = ev.toRawParameters(features)
      if (rawParameters.sparseKeys.isEmpty && rawParameters.denseKeys.isEmpty)
        throw InvalidArgumentException("Must provide at least one dense or sparse feature.")

      val output = Op.Builder[(Output[String], Seq[Output[Any]]), Seq[Output[Any]]](
        opType = "ParseSingleExample",
        name = name,
        input = (
            serialized,
            rawParameters.denseDefaults.toSeq.map(default => {
              Output.constant(
                value = default._2,
                name = s"Padding${Op.normalizeNameScope(default._1)}")(TF.fromDataType(default._2.dataType))
            }))
      ).setAttribute("num_sparse", rawParameters.sparseKeys.size)
          .setAttribute("sparse_keys", rawParameters.sparseKeys.toArray)
          .setAttribute("dense_keys", rawParameters.denseKeys.toArray)
          .setAttribute("sparse_types", rawParameters.sparseTypes.toArray)
          .setAttribute("Tdense", rawParameters.denseTypes.toArray)
          .setAttribute("dense_shapes", rawParameters.denseShapes.toArray)
          .build().output

      val (sparseIndices, sparseValues, sparseShapes, denseValues) = {
        val numSparse = rawParameters.sparseKeys.size
        (output.take(numSparse).map(_.asInstanceOf[Output[Long]]),
            output.slice(numSparse, 2 * numSparse),
            output.slice(2 * numSparse, 3 * numSparse).map(_.asInstanceOf[Output[Long]]),
            output.drop(3 * numSparse))
      }
      val sparseComposedValues = (sparseIndices, sparseValues, sparseShapes).zipped.map(SparseOutput(_, _, _))
      val sparseParsed = rawParameters.sparseKeys.zip(sparseComposedValues).toMap
      val denseParsed = rawParameters.denseKeys.zip(denseValues).toMap
      ev.fromParsed(features, sparseParsed, denseParsed)
    }
  }
}

object Parsing extends Parsing {
  sealed trait Feature

  /** Configuration for parsing a fixed-length input feature.
    *
    * To treat sparse input as dense, provide a `defaultValue`. Otherwise, the parsing functions will fail on any
    * examples missing this feature.
    *
    * @param  shape        Shape of the input feature.
    * @param  defaultValue Value to be used if an example is missing this feature. It must match the specified `shape`.
    * @tparam T Data type of the input feature.
    */
  case class FixedLengthFeature[T: TF](
      key: String,
      shape: Shape,
      defaultValue: Option[Tensor[T]] = None
  ) extends Feature {
    require(defaultValue.isEmpty || defaultValue.get.shape == shape,
      s"The default value shape (${defaultValue.get.shape}) does not match the expected $shape.")
  }

  /** Configuration for parsing a variable-length input feature.
    *
    * @param  dataType Data type of the input feature.
    */
  case class VariableLengthFeature[T: TF](
      key: String,
      dataType: DataType[T]
  ) extends Feature

  /** Configuration for parsing a sparse input feature.
    *
    * '''NOTE:''' Preferably use [[VariableLengthFeature]] (possibly in combination with a `SequenceExample`) in order
    * to parse [[SparseTensor]]s instead of [[SparseFeature]] due to its simplicity.
    *
    * Closely mimicking the [[SparseTensor]] that will be obtained by parsing an `Example` with a [[SparseFeature]]
    * configuration, a [[SparseFeature]] contains:
    *
    *   - An `indexKey`, which is a sequence of names -- one for each dimension in the resulting [[SparseTensor]], whose
    *     `indices(i)(j)` indicate the position of the `i`-th value in the `j`-th dimension will be equal to the `i`-th
    *     value in the feature with key names `indexKey(j)` in the `Example`.
    *   - A `valueKey`, which is the name of a key for a `Feature` in the `Example`, whose parsed tensor will be the
    *     resulting [[SparseTensor]]'s values.
    *   - A `dataType`, which is the resulting sparse tensor's data type.
    *   - A `size`, which is a sequence of integers, matching in length the `indexKey`, and corresponding to the
    *     resulting sparse tensor's dense shape.
    *
    * For example, we can represent the following 2D sparse tensor:
    * {{{
    *   SparseTensor(
    *     indices = Tensor(Tensor(3, 1), Tensor(20, 0)),
    *     values = Tensor(0.5, -1.0),
    *     denseShape = Tensor(100, 3))
    * }}}
    * with an `Example` input proto:
    * {{{
    *   features {
    *     feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
    *     feature { key: "ix0" value { int64_list { value: [ 3, 20 ] } } }
    *     feature { key: "ix1" value { int64_list { value: [ 1, 0 ] } } }
    *   }
    * }}}
    * and a `SparseFeature` configuration with two index keys:
    * {{{
    *   SparseFeature[Float](
    *     indexKey = Seq("ix0", "ix1"),
    *     valueKey = "val",
    *     size = Seq(100, 3))
    * }}}
    *
    * @param  indexKey      Sequence of string names of index features. For each key, the underlying feature's type must
    *                       be [[INT64]] and its length must always match the rank of the `valueKey` feature's value.
    * @param  valueKey      Name of the value feature. The underlying feature's type must be `dataType` and its rank
    *                       must always match that of all the `indexKey`s' features.
    * @param  size          Sequence of integers specifying the dense shape of the sparse tensor. The length of this
    *                       sequence must be equal to the length of the `indexKey` sequence. For each entry `i`, all
    *                       values in the `indexKey(i)`-th feature must be in the interval `[0, size(i))`.
    * @param  alreadySorted Boolean value specifying whether the values in `valueKey` are already sorted by their index
    *                       position. If so, we skip sorting.
    * @tparam T Data type of the `valueKey` feature.
    */
  case class SparseFeature[T: TF](
      indexKeys: Seq[String],
      valueKey: String,
      size: Seq[Long],
      alreadySorted: Boolean = false
  ) extends Feature

  sealed trait Features[T] {
    type Result

    @throws[InvalidArgumentException]
    def toRawParameters(features: T): Features.RawParameters

    def fromParsed(
        features: T,
        sparseParsed: Map[String, SparseOutput[Any]],
        denseParsed: Map[String, Output[Any]]
    ): Result
  }

  object Features {
    type Aux[T, R] = Features[T] {
      type Result = R
    }

    implicit def fromFixedLengthFeature[T: TF]: Features.Aux[FixedLengthFeature[T], Output[T]] = {
      new Features[FixedLengthFeature[T]] {
        override type Result = Output[T]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: FixedLengthFeature[T]): RawParameters = {
          prepareRawParameters(Seq(RawParameters(
            denseKeys = Seq(features.key),
            denseTypes = Seq(TF[T].dataType),
            denseShapes = Seq(features.shape),
            denseDefaults = features.defaultValue
                .map(v => ListMap(features.key -> v.asInstanceOf[Tensor[Any]]))
                .getOrElse(ListMap.empty))))
        }

        override def fromParsed(
            features: FixedLengthFeature[T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): Output[T] = {
          denseParsed(features.key).asInstanceOf[Output[T]]
        }
      }
    }

    implicit def fromVariableLengthFeature[T: TF]: Features.Aux[VariableLengthFeature[T], SparseOutput[T]] = {
      new Features[VariableLengthFeature[T]] {
        override type Result = SparseOutput[T]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: VariableLengthFeature[T]): RawParameters = {
          prepareRawParameters(Seq(RawParameters(
            sparseKeys = Seq(features.key),
            sparseTypes = Seq(TF[T].dataType))))
        }

        override def fromParsed(
            features: VariableLengthFeature[T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): SparseOutput[T] = {
          sparseParsed(features.key).asInstanceOf[SparseOutput[T]]
        }
      }
    }

    implicit def fromSparseFeature[T: TF]: Features.Aux[SparseFeature[T], SparseOutput[T]] = {
      new Features[SparseFeature[T]] {
        override type Result = SparseOutput[T]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: SparseFeature[T]): RawParameters = {
          val sparseKeys = mutable.ArrayBuffer.empty[String]
          val sparseTypes = mutable.ArrayBuffer.empty[DataType[Any]]
          features.indexKeys.foreach(indexKey => {
            val index = sparseKeys.indexOf(indexKey)
            if (index == -1) {
              sparseKeys.append(indexKey)
              sparseTypes.append(INT64)
            }
          })
          val valueDataType = TF[T].dataType
          val valueKeyIndex = sparseKeys.indexOf(features.valueKey)
          if (valueKeyIndex > -1) {
            val dataType = sparseTypes(valueKeyIndex)
            if (dataType != valueDataType) {
              throw InvalidArgumentException(
                s"Conflicting type '$dataType' vs '$valueDataType' for feature ${features.valueKey}.")
            }
          } else {
            sparseKeys.append(features.valueKey)
            sparseTypes.append(valueDataType)
          }
          prepareRawParameters(Seq(RawParameters(
            sparseKeys = sparseKeys,
            sparseTypes = sparseTypes)))
        }

        override def fromParsed(
            features: SparseFeature[T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): SparseOutput[T] = {
          Sparse.merge[T, Long](
            sparseIndices = features.indexKeys.map(k => sparseParsed(k).asInstanceOf[SparseOutput[Long]]),
            sparseValues = sparseParsed(features.valueKey).asInstanceOf[SparseOutput[T]],
            depths = features.size.map(s => s: Tensor[Long]),
            alreadySorted = features.alreadySorted,
            name = s"${features.valueKey}/SparseMerge")
        }
      }
    }

    implicit def fromOption[T, R](implicit ev: Features.Aux[T, R]): Features.Aux[Option[T], Option[R]] = {
      new Features[Option[T]] {
        override type Result = Option[R]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: Option[T]): RawParameters = {
          features.map(ev.toRawParameters).getOrElse(prepareRawParameters(Seq(RawParameters())))
        }

        override def fromParsed(
            features: Option[T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): Option[R] = {
          features.map(f => ev.fromParsed(f, sparseParsed, denseParsed))
        }
      }
    }

    implicit def fromSeq[T, R](implicit ev: Features.Aux[T, R]): Features.Aux[Seq[T], Seq[R]] = {
      new Features[Seq[T]] {
        override type Result = Seq[R]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: Seq[T]): RawParameters = {
          prepareRawParameters(features.map(ev.toRawParameters))
        }

        override def fromParsed(
            features: Seq[T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): Seq[R] = {
          features.map(f => ev.fromParsed(f, sparseParsed, denseParsed))
        }
      }
    }

    implicit def fromMap[K, T, R](implicit ev: Features.Aux[T, R]): Features.Aux[Map[K, T], Map[K, R]] = {
      new Features[Map[K, T]] {
        override type Result = Map[K, R]

        @throws[InvalidArgumentException]
        override def toRawParameters(features: Map[K, T]): RawParameters = {
          prepareRawParameters(features.values.toSeq.map(ev.toRawParameters))
        }

        override def fromParsed(
            features: Map[K, T],
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): Map[K, R] = {
          features.mapValues(f => ev.fromParsed(f, sparseParsed, denseParsed))
        }
      }
    }

    implicit val fromHNil: Features.Aux[HNil, HNil] = {
      new Features[HNil] {
        override type Result = HNil

        @throws[InvalidArgumentException]
        override def toRawParameters(features: HNil): RawParameters = {
          prepareRawParameters(Seq(RawParameters()))
        }

        override def fromParsed(
            features: HNil,
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): HNil = {
          HNil
        }
      }
    }

    implicit def fromHList[HT, HR, TT <: HList, TR <: HList](implicit
        evH: Strict[Features.Aux[HT, HR]],
        evT: Strict[Features.Aux[TT, TR]]
    ): Features.Aux[HT :: TT, HR :: TR] = {
      new Features[HT :: TT] {
        override type Result = HR :: TR

        @throws[InvalidArgumentException]
        override def toRawParameters(features: HT :: TT): RawParameters = {
          prepareRawParameters(Seq(evH.value.toRawParameters(features.head), evT.value.toRawParameters(features.tail)))
        }

        override def fromParsed(
            features: HT :: TT,
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): HR :: TR = {
          evH.value.fromParsed(features.head, sparseParsed, denseParsed) ::
              evT.value.fromParsed(features.tail, sparseParsed, denseParsed)
        }
      }
    }

    implicit def fromProduct[PT <: Product, PR <: Product, HT <: HList, HR <: HList](implicit
        genT: Generic.Aux[PT, HT],
        evT: Strict[Features.Aux[HT, HR]],
        tuplerR: Tupler.Aux[HR, PR]
    ): Features.Aux[PT, PR] = {
      new Features[PT] {
        override type Result = PR

        @throws[InvalidArgumentException]
        override def toRawParameters(features: PT): RawParameters = {
          evT.value.toRawParameters(genT.to(features))
        }

        override def fromParsed(
            features: PT,
            sparseParsed: Map[String, SparseOutput[Any]],
            denseParsed: Map[String, Output[Any]]
        ): PR = {
          tuplerR(evT.value.fromParsed(genT.to(features), sparseParsed, denseParsed))
        }
      }
    }

    /** Represents the raw parameters parsed from a set features that we pass to the parsing ops.
      * Note that we use a [[ListMap]] for `denseDefaults` because ignoring the ordering would cause graph equality to
      * fail in some tests. */
    private[Parsing] case class RawParameters(
        sparseKeys: Seq[String] = Seq.empty,
        sparseTypes: Seq[DataType[Any]] = Seq.empty,
        denseKeys: Seq[String] = Seq.empty,
        denseTypes: Seq[DataType[Any]] = Seq.empty,
        denseShapes: Seq[Shape] = Seq.empty,
        denseDefaults: ListMap[String, Tensor[Any]] = ListMap.empty)

    @throws[InvalidArgumentException]
    private[Parsing] def prepareRawParameters(
        rawParameters: Seq[RawParameters]
    ): RawParameters = {
      val sparseKeys = mutable.ListBuffer.empty[String]
      val sparseTypes = mutable.ListBuffer.empty[DataType[Any]]
      val denseKeys = mutable.ListBuffer.empty[String]
      val denseTypes = mutable.ListBuffer.empty[DataType[Any]]
      val denseShapes = mutable.ListBuffer.empty[Shape]
      val denseDefaults = mutable.ListMap.empty[String, Tensor[Any]]

      rawParameters.foreach(parameters => {
        // Process sparse keys and types.
        parameters.sparseKeys.indices.foreach(keyIndex => {
          val key = parameters.sparseKeys(keyIndex)
          val dataType = parameters.sparseTypes(keyIndex)
          val index = sparseKeys.indexOf(key)
          if (index > -1) {
            if (sparseTypes(index) != dataType) {
              throw InvalidArgumentException(
                s"Found mismatching data types ('${sparseTypes(index)}' vs '$dataType'), for the same key ('$key').")
            }
          } else {
            sparseKeys.append(key)
            sparseTypes.append(dataType)
          }
        })

        // Process dense keys, types, shapes, and defaults.
        parameters.denseKeys.indices.foreach(keyIndex => {
          val key = parameters.denseKeys(keyIndex)
          val dataType = parameters.denseTypes(keyIndex)
          val shape = parameters.denseShapes(keyIndex)
          val default = parameters.denseDefaults.get(key)
          val index = denseKeys.indexOf(key)
          if (index > -1) {
            if (denseTypes(index) != dataType) {
              throw InvalidArgumentException(
                s"Found mismatching data types ('${denseTypes(index)}' vs '$dataType'), for the same key ('$key').")
            }
            if (!denseShapes(index).isCompatibleWith(shape)) {
              throw InvalidArgumentException(
                s"Found incompatible shapes ('${denseShapes(index)}' vs '$shape'), for the same key ('$key').")
            }
            default.foreach(d => {
              if (denseDefaults.getOrElseUpdate(key, d) != d)
                throw InvalidArgumentException(s"Mismatching default values for the same key ('$key').")
            })
          } else {
            denseKeys.append(key)
            denseTypes.append(dataType)
            denseShapes.append(shape)
            default.foreach(d => denseDefaults.put(key, d))
          }
        })
      })

      val intersection = sparseKeys.intersect(denseKeys)
      if (intersection.nonEmpty)
        throw InvalidArgumentException(s"Dense keys and sparse keys must not intersect. Intersection: $intersection.")

      // Process the default dense values.
      var processedDenseDefaults = ListMap.empty[String, Tensor[Any]]
      for ((key, index) <- denseKeys.zipWithIndex) {
        val dataType = denseTypes(index)
        val shape = denseShapes(index)
        var default = denseDefaults.get(key)
        if (shape.rank > 0 && shape(0) == -1) {
          // For a variable stride dense shape, the default value should be a scalar padding value.
          default match {
            case None =>
              default = Some(Tensor.zeros(dataType, Shape()))
            case Some(value) =>
              // Reshape to a scalar to ensure the user gets an error if they provide a tensor that is not intended to
              // used as a padding value (i.e., containing zero or more than 2 elements).
              default = Some(value.reshape(Shape()))
          }
        } else if (default.isEmpty) {
          default = Some(Tensor.empty(dataType))
        }
        processedDenseDefaults += key -> default.get
      }

      RawParameters(
        sparseKeys = sparseKeys,
        sparseTypes = sparseTypes,
        denseKeys = denseKeys,
        denseTypes = denseTypes,
        denseShapes = denseShapes,
        denseDefaults = processedDenseDefaults)
    }
  }

  /** @define OpDocParsingEncodeTensor
    *   The `encodeTensor` op transforms a tensor into a serialized `TensorProto` proto.
    *
    * @define OpDocParsingDecodeTensor
    *   The `decodeTensor` op transforms a serialized `TensorProto` proto into a tensor.
    *
    * @define OpDocParsingDecodeRaw
    *   The `decodeRaw` op reinterprets the bytes of a string as a vector of numbers.
    *
    * @define OpDocParsingDecodeCSV
    *   The `decodeCSV` op converts CSV records to tensors. Each column maps to one tensor.
    *
    *   The [RFC 4180](https://tools.ietf.org/html/rfc4180) format is expected for the CSV records. Note that we allow
    *   leading and trailing spaces with integer or floating-point fields.
    *
    * @define OpDocParsingStringToNumber
    *   The `stringToNumber` op converts each string in the input tensor to the specified numeric type,
    *
    *   '''NOTE:''' Int overflow results in an error while [[FLOAT32]] overflow results in a rounded value.
    *
    * @define OpDocParsingDecodeJSONExample
    *   The `decodeJSONExample` op converts JSON-encoded `Example` records to binary protocol buffer strings.
    *
    *   The op translates a tensor containing `Example` records, encoded using the
    *   [standard JSON mapping](https://developers.google.com/protocol-buffers/docs/proto3#json), into a tensor
    *   containing the same records encoded as binary protocol buffers. The resulting tensor can then be fed to any of
    *   the other `Example`-parsing ops.
    */
  private[ops] trait Documentation
}
