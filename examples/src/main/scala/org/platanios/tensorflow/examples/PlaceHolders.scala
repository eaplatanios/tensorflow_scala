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
import org.platanios.tensorflow.api.tensors.Tensor
import org.slf4j.LoggerFactory

//Python Ver:
//import tensorflow as tf
//from tensorflow.python.framework import ops
//sess = tf.Session()



/**
  * Created by Mageswaran.D (mageswaran1989@gmail.com) on 24/04/18.
  */
object PlaceHolders {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / PlaceHolders"))

  def main(args: Array[String]): Unit = {

    //PyVer:
    // sess = tf.Session()
    //x = tf.placeholder(tf.float32, shape=(4, 4))

    //Get graph handle with the tf.Session()
    //# Input data to placeholder, note that 'rand_array' and 'x' are the same shape.
    //  rand_array = np.random.rand(4, 4)

    //# Create a Tensor to perform an operation (here, y will be equal to x, a 4x4 matrix)
    //y = tf.identity(x)

    //# Print the output, feeding the value of x into the computational graph
    //print(sess.run(y, feed_dict={x: rand_array}))

    val session = Session()

    val x = tf.placeholder(FLOAT64, shape = Shape(3,3))

    val y = tf.identity(x)

    val data = Tensor(Tensor(Tensor(2.0), Tensor(0.0), Tensor(5.0)),
                        Tensor(Tensor(1.0), Tensor(4.0), Tensor(7.0)),
                        Tensor(Tensor(56.0), Tensor(-2.0), Tensor(-9.0)))

    session.run(targets = tf.globalVariablesInitializer())

    val feed_dict = Map(x -> data)

    logger.info(session.run(feeds = feed_dict, fetches = y).summarize())
  }
}
