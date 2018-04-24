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
import org.slf4j.LoggerFactory

//Python Ver:
//import tensorflow as tf
//from tensorflow.python.framework import ops
//sess = tf.Session()


/**
  * Created by Mageswaran.D (mageswaran1989@gmail.com) on 24/04/18.
  */
object Tensors {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Tensors"))


  def main(args: Array[String]): Unit = {

    // PyVer:
    // sess = tf.Session()

    //Get graph handle with the tf.Session()
    val session = Session()

    // PyVer:
    // my_tensor = tf.zeros([1,20])
    // sess.run(my_tensor)

    //TensorFlow has built in function to create tensors for use in variables.
    // For example, we can create a zero filled tensor of predefined shape using the tf.zeros() function as follows.
    val myTensor = tf.zeros(INT32, Shape(3, 4))

    //We can evaluate tensors with calling a run() method on our session.
    logger.info(s"myTensor info: ${ session.run(fetches = myTensor).toString}")
    logger.info(s"Output of myTensor: ${ session.run(fetches = myTensor).summarize()}")


    // PyVer:
    // my_var = tf.Variable(tf.zeros([1,20]))
    // sess.run(my_var.initializer)
    // sess.run(my_var)

    //TensorFlow algorithms need to know which objects are variables and which are constants.
    val myVar = tf.variable("myVar", FLOAT32, Shape(3, 4), initializer = ZerosInitializer)

    //Note that you can not run sess.run(my_var), this would result in an error.
    // Because TensorFlow operates with computational graphs, we have to create a variable
    // intialization operation in order to evaluate variables.
    // we can initialize one variable at a time by calling the variable method my_var.initializer.
    session.run(targets = myVar.initializer)
    logger.info(s"Output of myVar: ${session.run(fetches = myVar.value).summarize()}")


    val fillVar = tf.variable("fillVar", FLOAT32,  Shape(4,4), initializer=ConstantInitializer(-1))
    session.run(targets = fillVar.initializer)
    logger.info(s"Output of fillVar: ${session.run(fetches = fillVar.value).summarize()}")

    //PyVer:
    //rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
    //runif_var = tf.random_uniform([row_dim, col_dim], minval=0, maxval=4)

    //print(sess.run(rnorm_var))
    //print(sess.run(runif_var))

    //Random number Tensors
    val randomNormalizedTensor = tf.randomNormal(shape=Shape(3,3))
    logger.info(s"randomNormalizedTensor info: ${session.run(fetches = randomNormalizedTensor).toString}")
    logger.info(s"Output of randomNormalizedTensor: ${session.run(fetches = randomNormalizedTensor).summarize()}")

  }
}
