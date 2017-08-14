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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.learn.{Hook, Layer, TrainingState}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
//case class Logger(quantities: Seq[Layer[_, _]], name: String = "LoggerHook") extends Hook {
//  private[this] val logger = Logger(LoggerFactory.getLogger(name))
//
//  logger.info(f"${"Iteration"}%10s | ${"Train Loss"}%13s | ${"Test Accuracy"}%13s")
//
//  override def initialize[SI, OI, ST, OT, I, T](state: TrainingState[SI, OI, ST, OT, I, T]): Unit = ()
//  override def call[SI, OI, ST, OT, I, T](state: TrainingState[SI, OI, ST, OT, I, T]): Boolean = ???
//}
