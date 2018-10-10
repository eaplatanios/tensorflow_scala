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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.types.{TF, IsFloat16OrFloat32OrFloat64}
import org.platanios.tensorflow.api.ops.{NN, Op, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

import shapeless._

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

import scala.reflect.ClassTag

/** RNN cell that applies dropout to the provided RNN cell.
  *
  * Note that currently, a different dropout mask is used for each time step in an RNN (i.e., not using the variational
  * recurrent dropout method described in
  * ["A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"](https://arxiv.org/abs/1512.05287).
  *
  * Note also that for LSTM cells, no dropout is applied to the memory tensor of the state. It is only applied to the
  * state tensor.
  *
  * @param  cell                  RNN cell on which to perform dropout.
  * @param  inputKeepProbability  Keep probability for the input of the RNN cell.
  * @param  outputKeepProbability Keep probability for the output of the RNN cell.
  * @param  stateKeepProbability  Keep probability for the output state of the RNN cell.
  * @param  seed                  Optional random seed, used to generate a random seed pair for the random number
  *                               generator, when combined with the graph-level seed.
  * @param  name                  Name prefix used for all new ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class DropoutWrapper[O, OS, S, SS] protected (
    val cell: RNNCell[O, OS, S, SS],
    val inputKeepProbability: Output[Float] = 1.0f,
    val outputKeepProbability: Output[Float] = 1.0f,
    val stateKeepProbability: Output[Float] = 1.0f,
    val seed: Option[Int] = None,
    val name: String = "DropoutWrapper"
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS],
    evODropout: DropoutWrapper.Supported[O],
    evSDropout: DropoutWrapper.Supported[S]
) extends RNNCell[O, OS, S, SS]()(evO, evS) {
  override def outputShape: OS = cell.outputShape
  override def stateShape: SS = cell.stateShape
  override def forward(input: Tuple[O, S]): Tuple[O, S] = {
    Op.nameScope(name) {
      val dropoutInput = evODropout.dropout(input.output, inputKeepProbability, "input", seed)._1
      val nextTuple = cell(Tuple(dropoutInput, input.state))
      val nextState = evSDropout.dropout(nextTuple.state, stateKeepProbability, "state", seed)._1
      val nextOutput = evODropout.dropout(nextTuple.output, stateKeepProbability, "output", seed)._1
      Tuple(nextOutput, nextState)
    }
  }
}

object DropoutWrapper {
  def apply[O, OS, S, SS](
      cell: RNNCell[O, OS, S, SS],
      inputKeepProbability: Output[Float] = 1.0f,
      outputKeepProbability: Output[Float] = 1.0f,
      stateKeepProbability: Output[Float] = 1.0f,
      seed: Option[Int] = None,
      name: String = "DropoutWrapper"
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS],
      evODropout: DropoutWrapper.Supported[O],
      evSDropout: DropoutWrapper.Supported[S]
  ): DropoutWrapper[O, OS, S, SS] = {
    new DropoutWrapper(
      cell, inputKeepProbability, outputKeepProbability,
      stateKeepProbability, seed, name)
  }

  private def generateSeed(
      saltPrefix: String,
      seed: Option[Int],
      index: Int
  ): Option[Int] = {
    seed.map(s => {
      val md5 = MessageDigest.getInstance("MD5")
          .digest(s"$s${saltPrefix}_$index".getBytes(StandardCharsets.UTF_8))
      ByteBuffer.wrap(md5.take(8)).getInt() & 0x7fffffff
    })
  }

  trait Supported[T] {
    def dropout(
        value: T,
        keepProbability: Output[Float],
        saltPrefix: String,
        seed: Option[Int],
        index: Int = 0
    ): (T, Int)
  }

  object Supported {
    implicit val fromUnit: Supported[Unit] = {
      new Supported[Unit] {
        override def dropout(
            value: Unit,
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Unit, Int) = {
          ((), index)
        }
      }
    }

    implicit def fromOutput[T: TF : IsFloat16OrFloat32OrFloat64]: Supported[Output[T]] = {
      new Supported[Output[T]] {
        override def dropout(
            value: Output[T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Output[T], Int) = {
          (NN.dynamicDropout(
            value,
            keepProbability.castTo[T],
            seed = generateSeed(saltPrefix, seed, index)
          ), index + 1)
        }
      }
    }

    implicit def fromTensorArray[T: TF : IsFloat16OrFloat32OrFloat64]: Supported[TensorArray[T]] = {
      new Supported[TensorArray[T]] {
        override def dropout(
            value: TensorArray[T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (TensorArray[T], Int) = {
          (value, index)
        }
      }
    }

    implicit def fromOption[T](implicit ev: Supported[T]): Supported[Option[T]] = {
      new Supported[Option[T]] {
        override def dropout(
            value: Option[T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Option[T], Int) = {
          // TODO: Make this atomic for parallel sequences.
          var currentIndex = index
          (value.map({ v =>
            val (dropoutV, dropoutIndex) = ev.dropout(
              v, keepProbability, saltPrefix, seed, currentIndex)
            currentIndex = dropoutIndex
            dropoutV
          }), currentIndex)
        }
      }
    }

    implicit def fromArray[T: ClassTag](implicit ev: Supported[T]): Supported[Array[T]] = {
      new Supported[Array[T]] {
        override def dropout(
            value: Array[T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Array[T], Int) = {
          var currentIndex = index
          (value.map({ v =>
            val (dropoutV, dropoutIndex) = ev.dropout(
              v, keepProbability, saltPrefix, seed, currentIndex)
            currentIndex = dropoutIndex
            dropoutV
          }), currentIndex)
        }
      }
    }

    implicit def fromSeq[T](implicit ev: Supported[T]): Supported[Seq[T]] = {
      new Supported[Seq[T]] {
        override def dropout(
            value: Seq[T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Seq[T], Int) = {
          // TODO: Make this atomic for parallel sequences.
          var currentIndex = index
          (value.map({ v =>
            val (dropoutV, dropoutIndex) = ev.dropout(
              v, keepProbability, saltPrefix, seed, currentIndex)
            currentIndex = dropoutIndex
            dropoutV
          }), currentIndex)
        }
      }
    }

    implicit def fromMap[T, MK](implicit ev: Supported[T]): Supported[Map[MK, T]] = {
      new Supported[Map[MK, T]] {
        override def dropout(
            value: Map[MK, T],
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (Map[MK, T], Int) = {
          // TODO: Make this atomic for parallel maps.
          var currentIndex = index
          (value.mapValues({ v =>
            val (dropoutV, dropoutIndex) = ev.dropout(
              v, keepProbability, saltPrefix, seed, currentIndex)
            currentIndex = dropoutIndex
            dropoutV
          }), currentIndex)
        }
      }
    }

    implicit val fromHNil: Supported[HNil] = {
      new Supported[HNil] {
        override def dropout(
            value: HNil,
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (HNil, Int) = {
          (HNil, index)
        }
      }
    }

    implicit def fromHList[H, T <: HList](implicit
        evH: Strict[Supported[H]],
        evT: Supported[T]
    ): Supported[H :: T] = {
      new Supported[H :: T] {
        override def dropout(
            value: H :: T,
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int
        ): (H :: T, Int) = {
          val dropoutH = evH.value.dropout(value.head, keepProbability, saltPrefix, seed, index)
          val dropoutT = evT.dropout(value.tail, keepProbability, saltPrefix, seed, dropoutH._2)
          (dropoutH._1 :: dropoutT._1, dropoutT._2)
        }
      }
    }

    implicit def fromCoproduct[H, T <: Coproduct](implicit
        evH: Strict[Supported[H]],
        evT: Supported[T]
    ): Supported[H :+: T] = {
      new Supported[H :+: T] {
        override def dropout(
            value: H :+: T,
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int
        ): (H :+: T, Int) = {
          value match {
            case Inl(h) =>
              val (result, i) = evH.value.dropout(h, keepProbability, saltPrefix, seed, index)
              (Inl(result), i)
            case Inr(t) =>
              val (result, i) = evT.dropout(t, keepProbability, saltPrefix, seed, index)
              (Inr(result), i)
          }
        }
      }
    }

    implicit def fromProduct[P <: Product, L <: HList](implicit
        genP: Generic.Aux[P, L],
        evL: Strict[Supported[L]]
    ): Supported[P] = {
      new Supported[P] {
        override def dropout(
            value: P,
            keepProbability: Output[Float],
            saltPrefix: String,
            seed: Option[Int],
            index: Int = 0
        ): (P, Int) = {
          val tuple = evL.value.dropout(genP.to(value), keepProbability, saltPrefix, seed, index)
          (genP.from(tuple._1), tuple._2)
        }
      }
    }
  }
}
