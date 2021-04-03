package org.platanios.tensorflow.api.implicits.helpers

import org.platanios.tensorflow.api.TF
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops._

import shapeless._
import shapeless.labelled.{FieldType, field}
import shapeless.ops.hlist.Tupler

import scala.collection.compat._

trait PlaceholderSupport[T] {
  type D
  type S

  def dataType(shape: S): D
  def placeholder(shape: S): T
}

object PlaceholderSupport {
  def apply[T](implicit ev: PlaceholderSupport[T]): PlaceholderSupport.Aux[T, ev.D, ev.S] = {
    ev.asInstanceOf[PlaceholderSupport.Aux[T, ev.D, ev.S]]
  }

  type Aux[T, DD, SS] = PlaceholderSupport[T] {
    type D = DD
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit, Unit] = {
    new PlaceholderSupport[Unit] {
      override type D = Unit
      override type S = Unit

      override def dataType(shape: Unit): Unit = ()
      override def placeholder(shape: Unit): Unit = ()
    }
  }

  implicit def fromOutput[T: TF]: Aux[Output[T], DataType[T], Shape] = {
    new PlaceholderSupport[Output[T]] {
      override type D = DataType[T]
      override type S = Shape

      override def dataType(shape: Shape): DataType[T] = TF[T].dataType
      override def placeholder(shape: Shape): Output[T] = Basic.placeholder[T](shape)
    }
  }

  implicit def fromOutputIndexedSlices[T: TF]: Aux[OutputIndexedSlices[T], IndexedSlicesDataType[T], SparseShape] = {
    new PlaceholderSupport[OutputIndexedSlices[T]] {
      override type D = IndexedSlicesDataType[T]
      override type S = SparseShape

      override def dataType(shape: SparseShape): IndexedSlicesDataType[T] = (INT32, TF[T].dataType, INT32)

      override def placeholder(shape: SparseShape): OutputIndexedSlices[T] = {
        OutputIndexedSlices(
          Basic.placeholder[Int](shape._1, name = "IndicesPlaceholder"),
          Basic.placeholder[T](shape._2, name = "ValuesPlaceholder"),
          Basic.placeholder[Int](shape._3, name = "DenseShapePlaceholder"),
        )
      }
    }
  }

  implicit def fromSparseOutput[T: TF]: Aux[SparseOutput[T], SparseDataType[T], SparseShape] = {
    new PlaceholderSupport[SparseOutput[T]] {
      override type D = SparseDataType[T]
      override type S = SparseShape

      override def dataType(shape: SparseShape): SparseDataType[T] = (INT64, TF[T].dataType, INT64)

      override def placeholder(shape: SparseShape): SparseOutput[T] = {
        SparseOutput(
          Basic.placeholder[Long](shape._1, name = "IndicesPlaceholder"),
          Basic.placeholder[T](shape._2, name = "ValuesPlaceholder"),
          Basic.placeholder[Long](shape._3, name = "DenseShapePlaceholder"),
        )
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], DataType[Float], Shape] = {
    new PlaceholderSupport[TensorArray[T]] {
      override type D = DataType[Float]
      override type S = Shape

      override def dataType(shape: Shape): DataType[Float] = FLOAT32
      override def placeholder(shape: Shape): TensorArray[T] = ???
    }
  }

  implicit def fromDataset[T]: Aux[Dataset[T], DataType[Variant], Shape] = {
    new PlaceholderSupport[Dataset[T]] {
      override type D = DataType[Variant]
      override type S = Shape

      override def dataType(shape: Shape): DataType[Variant] = VARIANT
      override def placeholder(shape: Shape): Dataset[T] = ???
    }
  }

  implicit def fromOption[T](implicit
      ev: PlaceholderSupport[T]
  ): PlaceholderSupport.Aux[Option[T], Option[ev.D], Option[ev.S]] = {
    new PlaceholderSupport[Option[T]] {
      override type D = Option[ev.D]
      override type S = Option[ev.S]

      override def dataType(shape: Option[ev.S]): Option[ev.D] = shape.map(ev.dataType(_))
      override def placeholder(shape: Option[ev.S]): Option[T] = shape.map(ev.placeholder(_))
    }
  }

  implicit def fromSeq[T](implicit
      ev: PlaceholderSupport[T]
  ): PlaceholderSupport.Aux[Seq[T], Seq[ev.D], Seq[ev.S]] = {
    new PlaceholderSupport[Seq[T]] {
      override type D = Seq[ev.D]
      override type S = Seq[ev.S]

      override def dataType(shape: Seq[ev.S]): Seq[ev.D] = shape.map(ev.dataType(_))

      override def placeholder(shape: Seq[ev.S]): Seq[T] = shape.zipWithIndex.map {
        case (shape, index) => Op.nameScope(s"Element$index")(ev.placeholder(shape))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: PlaceholderSupport[T]
  ): PlaceholderSupport.Aux[Map[K, T], Map[K, ev.D], Map[K, ev.S]] = {
    new PlaceholderSupport[Map[K, T]] {
      override type D = Map[K, ev.D]
      override type S = Map[K, ev.S]

      override def dataType(shape: Map[K, ev.S]): Map[K, ev.D] = shape.view.mapValues(ev.dataType(_)).toMap

      override def placeholder(shape: Map[K, ev.S]): Map[K, T] = shape.zipWithIndex.map {
        case ((key, shape), index) => key -> Op.nameScope(s"Element$index")(ev.placeholder(shape))
      }.toMap
    }
  }

  implicit val fromHNil: PlaceholderSupport.Aux[HNil, HNil, HNil] = {
    new PlaceholderSupport[HNil] {
      override type D = HNil
      override type S = HNil

      override def dataType(shape: HNil): HNil = HNil
      override def placeholder(shape: HNil): HNil = HNil
    }
  }

  implicit def fromHList[KHT <: Symbol, HT, HD, HS, TT <: HList, TD <: HList, TS <: HList](implicit
      witnessKHT: Witness.Aux[KHT],
      evH: Strict[PlaceholderSupport.Aux[HT, HD, HS]],
      evT: Strict[PlaceholderSupport.Aux[TT, TD, TS]],
  ): PlaceholderSupport.Aux[FieldType[KHT, HT] :: TT, HD :: TD, HS :: TS] = {
    new PlaceholderSupport[FieldType[KHT, HT] :: TT] {
      override type D = HD :: TD
      override type S = HS :: TS

      override def dataType(shape: HS :: TS): HD :: TD = {
        evH.value.dataType(shape.head) :: evT.value.dataType(shape.tail)
      }

      override def placeholder(shape: HS :: TS): FieldType[KHT, HT] :: TT = {
        Op.nameScope(witnessKHT.value.name)(field[KHT](evH.value.placeholder(shape.head))) ::
            evT.value.placeholder(shape.tail)
      }
    }
  }

  implicit def fromKnownProduct[PT <: Product, PD, PS, HT <: HList, HD <: HList, HS <: HList](implicit
      genT: LabelledGeneric.Aux[PT, HT],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS],
      evT: Strict[PlaceholderSupport.Aux[HT, HD, HS]],
  ): PlaceholderSupport.Aux[PT, PD, PS] = {
    new PlaceholderSupport[PT] {
      override type D = PD
      override type S = PS

      override def dataType(shape: PS): PD = genD.from(evT.value.dataType(genS.to(shape)))
      override def placeholder(shape: PS): PT = genT.from(evT.value.placeholder(genS.to(shape)))
    }
  }
}

trait PlaceholderSupportLowPriorityImplicits {
  implicit def fromProduct[PT <: Product, PD, PS, HT <: HList, HD <: HList, HS <: HList](implicit
      genT: LabelledGeneric.Aux[PT, HT],
      evT: Strict[PlaceholderSupport.Aux[HT, HD, HS]],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS],
  ): PlaceholderSupport.Aux[PT, PD, PS] = {
    new PlaceholderSupport[PT] {
      override type D = PD
      override type S = PS

      override def dataType(shape: PS): PD = genD.from(evT.value.dataType(genS.to(shape)))
      override def placeholder(shape: PS): PT = genT.from(evT.value.placeholder(genS.to(shape)))
    }
  }
}
