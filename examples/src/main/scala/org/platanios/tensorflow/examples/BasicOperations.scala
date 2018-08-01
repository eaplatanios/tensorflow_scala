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

package org.platanios.tensorflow.examples

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, ZerosInitializer}
import org.platanios.tensorflow.api.tensors.Tensor
import org.slf4j.LoggerFactory

//Python Ver:
//import tensorflow as tf
//from tensorflow.python.framework import ops
//sess = tf.Session()


/**
  * Created by Mageswaran.D (mageswaran1989@gmail.com) on 24/04/18.
  */
object BasicOperations {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / BasicOperations"))


  def main(args: Array[String]): Unit = {

    val session = Session()

    logger.info(s"\n\n This example covers most of the ops defined in org.platanios.tensorflow.api.ops")


    logger.info(s"\n\n tf.zeros: \n${ session.run(fetches = tf.zeros(INT32, Shape(3,3))).summarize()}")

    val x = tf.placeholder(FLOAT64, shape = Shape(3,3))

    val data = Tensor(Tensor(Tensor(1.0), Tensor(2.0), Tensor(3.0)),
      Tensor(Tensor(4.0), Tensor(5.0), Tensor(6.0)),
      Tensor(Tensor(7.0), Tensor(8.0), Tensor(9.0)))

    session.run(targets = tf.globalVariablesInitializer())

    val y = tf.identity(x)

    val feed_dict = Map(x -> data)

    logger.info(s"\n\n tf.placeholder: \n${session.run(feeds = feed_dict, fetches = y).summarize()}")


    logger.info(s"\n\n tf.zerosLike: \n${ session.run(fetches = tf.zerosLike(x)).summarize()}")

    logger.info(s"\n\n tf.ones: \n${ session.run(fetches = tf.ones(INT32, Shape(3,3))).summarize()}")

    logger.info(s"\n\n tf.onesLike: \n${ session.run(fetches = tf.onesLike(x)).summarize()}")

    logger.info(s"\n\n tf.rank: \n${ session.run(fetches = tf.rank(x)).summarize()}")

    logger.info(s"\n\n tf.size: \n${ session.run(fetches = tf.size(x)).summarize()}")

    logger.info(s"\n\n tf.size: \n${ session.run(fetches = tf.shape(x)).summarize()}")

    logger.info(s"\n\n tf.expandDims: \n${ session.run(fetches = tf.ones(INT32, Shape(3,3)).expandDims(0)).summarize()}")
    logger.info(s"\n\n tf.expandDims: \n${ session.run(fetches = tf.ones(INT32, Shape(3,3)).expandDims(1)).summarize()}")

    logger.info(s"\n\n tf.squeeze: \n${ session.run(fetches = tf.ones(INT32, Shape(1,3,3)).squeeze())}")

    logger.info(s"\n\n tf.stack: \n${ session.run(feeds = feed_dict, fetches = tf.stack(Array(x,x))).summarize()}")
    logger.info(s"\n\n tf.stack: \n${ session.run(feeds = feed_dict, fetches = tf.stack(Array(x,x), axis = 1)).summarize()}")

//    logger.info(s"\n\n tf.parallelStack: \n${ session.run(feeds = feed_dict, fetches = tf.parallelStack(
//      Array(tf.constant(List(1,2,3)),
//        tf.constant(List(1,2,3)),
//        tf.constant(List(1,2,3))))).summarize()}")

    logger.info(s"\n\n tf.unstack: \n${ session.run(feeds = feed_dict, fetches = tf.unstack(x, axis = 0)).summarize()}")

    logger.info(s"\n\n tf.concatenate: \n${ session.run(feeds = feed_dict, fetches = tf.concatenate(Array(x,x), axis = 0)).summarize()}")
    logger.info(s"\n\n tf.concatenate: \n${ session.run(feeds = feed_dict, fetches = tf.concatenate(Array(x,x), axis = 1)).summarize()}")

//    logger.info(s"\n\n tf.meshGrid: \n${ session.run(feeds = feed_dict, fetches = tf.meshGrid(Tensors(Tensors(2)),Tensors(Tensors(2)))).summarize()}")

  }
}
