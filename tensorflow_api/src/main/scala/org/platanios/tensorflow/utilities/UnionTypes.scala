package org.platanios.tensorflow.utilities

import scala.language.higherKinds

/**
  *
  * @author Emmanouil Antonios Platanios
  */
//trait UnionTypes {
//  type ![A] = A => Nothing
//  type !![A] = ![![A]]
//
//  trait Disjunction {
//    self =>
//    type D
//    type t[S] = Disjunction {type D = self.D with ![S]}
//  }
//
//  type t[T] = {
//    type t[S] = (Disjunction {type D = ![T]})#t[S]
//  }
//
//  type or[T <: Disjunction] = ![T#D]
//
//  type Contains[S, T <: Disjunction] = !![S] <:< or[T]
//  type ∈[S, T <: Disjunction] = Contains[S, T]
//
//  //  sealed trait Union[T] {
//  //    val value: Any
//  //  }
//}

//object UnionTypes extends UnionTypes

object UnionTypes {
  sealed trait ¬[-A]

  sealed trait TSet {
    type Compound[A]
    type Map[F[_]] <: TSet
  }

  sealed trait ∅ extends TSet {
    type Compound[A] = A
    type Map[F[_]] = ∅
  }

  // Note that this type is left-associative for the sake of concision.
  sealed trait ∨[T <: TSet, H] extends TSet {
    // Given a type of the form `∅ ∨ A ∨ B ∨ ...` and parameter `X`, we want to produce the type
    // `¬[A] with ¬[B] with ... <:< ¬[X]`.
    type Member[X] = T#Map[¬]#Compound[¬[H]] <:< ¬[X]

    // This could be generalized as a fold, but for concision we leave it as is.
    type Compound[A] = T#Compound[H with A]

    type Map[F[_]] = T#Map[F] ∨ F[H]
  }
}
