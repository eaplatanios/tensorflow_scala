/* Copyright 2017, Mageswaran.D <mageswaran1989@gmail.com>. All Rights Reserved.
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
  * Created by Mageswaran.D <mageswaran1989@gmail.com> on 22/8/17.
  */
object ActivationFunctions {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / ActivationFunctions"))

  def main(args: Array[String]): Unit = {

    val session = tf.Session()

    logger.info(s"relu: : ${session.run(fetches = tf.relu(Array(1,-1, 2,-2, 3, -3, 9))).summarize()}")

    logger.info(s"relu6: : ${session.run(fetches = tf.relu6(Array(1,-1, 2,-2, 3, -3, 9))).summarize()}")

    //TODO sigmoid, tanh, softsign, softplus

    logger.info(s"elu: : ${session.run(fetches = tf.elu(Array(1,-1, 2,-2, 3, -3, 9))).summarize()}")

  }
}