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
object Matrices {

  private[this] val logger = Logger(LoggerFactory.getLogger("Examples / Matrices"))

  def main(args: Array[String]): Unit = {

    val session = Session()

    //PyVer:
    //identity_matrix = tf.diag([1.0,1.0,1.0])
    //print(sess.run(identity_matrix))
    val identityMatrix = tf.diag(Array(1,2,3,4,5))
    logger.info(s"\n identityMatrix: \n${session.run(fetches = identityMatrix).summarize()}")

    //A = tf.truncated_normal([2,3])
    //print(sess.run(A))
    //TODO

    //B = tf.fill([2,3], 5.0)
    //print(sess.run(B))
    val fill = tf.fill(dataType= INT32, shape = Shape(3,3))(5)
    logger.info(s"\n fill: \n${session.run(fetches = fill).summarize()}")


    //C = tf.random_uniform([3,2])
    //print(sess.run(C))
    val randMatrix = tf.randomUniform(shape = Shape(3,2))
    logger.info(s" \n random_uniform: \n${session.run(fetches = randMatrix).summarize()}")

    //Values will differ since this is a fresh run!
    logger.info(s"\n transpose: \n${session.run(fetches = tf.transpose(randMatrix)).summarize()}")

    //D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
    //print(sess.run(D))
    //TODO

    val A = tf.fill(shape = Shape(3,3))(value=3)
    val B = tf.fill(shape = Shape(3,3))(value=3)

    logger.info(s"\n A + B: \n${session.run(fetches = A+B).summarize()}")
    logger.info(s"\n A - B: \n${session.run(fetches = A-B).summarize()}")

    logger.info(s"\n A * B (elementwise product): \n${session.run(fetches = A*B).summarize()}")
    logger.info(s"\n A * B (dot product): \n${session.run(fetches = tf.matmul(A,B)).summarize()}")

//    logger.info(s"matrix_determinant: ${session.run(fetches = tf.matrixDeterminant(A)).summarize()}") TODO

//    logger.info(s"matrixInverse: ${session.run(fetches = tf.matrixInverse(A)).summarize()}") TODO
  }
}
