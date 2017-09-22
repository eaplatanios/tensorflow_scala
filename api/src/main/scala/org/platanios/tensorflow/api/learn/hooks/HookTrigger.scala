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

/** Determines when hooks should be triggered.
  *
  * @author Emmanouil Antonios Platanios
  */
trait HookTrigger {
  /** Returns a copy of this hook trigger that is also reset. */
  def copy(): HookTrigger

  /** Resets the internal state of this trigger (e.g., step counter or timer). */
  def reset(): Unit

  /** Returns `true` if the hook should be triggered for the specified step. */
  def shouldTriggerForStep(step: Int): Boolean

  /** Updates the last triggered step and time.
    *
    * @param  step Current step.
    * @return A tuple `(elapsedTime, elapsedSteps)`, where `elapsedTime` is the number of seconds between the current
    *         trigger and the last one, and `elapsedSteps` is the number of steps between the current trigger and the
    *         last one. Both values will be set to `None` on the first trigger.
    */
  def updateLastTrigger(step: Int): Option[(Double, Int)]

  /** Returns the last triggered time step or `None`, if never triggered. */
  def lastTriggerStep(): Option[Int]
}

/** Hook trigger that never actually triggers. */
case object NoHookTrigger extends HookTrigger {
  /** Returns a copy of this hook trigger that is also reset. */
  override def copy(): HookTrigger = this

  /** Resets the internal state of this trigger (e.g., step counter or timer). */
  override def reset(): Unit = ()

  /** Returns `true` if the hook should be triggered for the specified step. */
  override def shouldTriggerForStep(step: Int): Boolean = false

  /** Updates the last triggered step and time.
    *
    * @param  step Current step.
    * @return A tuple `(elapsedTime, elapsedSteps)`, where `elapsedTime` is the number of seconds between the current
    *         trigger and the last one, and `elapsedSteps` is the number of steps between the current trigger and the
    *         last one. Both values will be set to `None` on the first trigger.
    */
  override def updateLastTrigger(step: Int): Option[(Double, Int)] = None

  /** Returns the last triggered time step or `None`, if never triggered. */
  override def lastTriggerStep(): Option[Int] = None
}

/** Hook trigger that triggers at most once every `numSteps` steps.
  *
  * @param  numSteps   Triggering step frequency.
  */
case class StepHookTrigger(numSteps: Int) extends HookTrigger {
  require(numSteps >= 0, s"'numSteps' (= $numSteps) must be a non-negative number.")

  /** Returns a copy of this hook trigger that is also reset. */
  override def copy(): HookTrigger = StepHookTrigger(numSteps)

  private[this] var _lastTrigger: Option[(Double, Int)] = None

  /** Resets the internal state of this trigger (e.g., step counter or timer). */
  override def reset(): Unit = _lastTrigger = None

  /** Returns `true` if the hook should be triggered for the specified step. */
  override def shouldTriggerForStep(step: Int): Boolean = _lastTrigger match {
    case None => true
    case Some((_, s)) if s == step => false
    case Some((_, s)) => step >= s + numSteps
  }

  /** Updates the last triggered step and time.
    *
    * @param  step Current step.
    * @return A tuple `(elapsedTime, elapsedSteps)`, where `elapsedTime` is the number of seconds between the current
    *         trigger and the last one, and `elapsedSteps` is the number of steps between the current trigger and the
    *         last one. Both values will be set to `None` on the first trigger.
    */
  override def updateLastTrigger(step: Int): Option[(Double, Int)] = {
    val currentTime = System.currentTimeMillis().toDouble / 1000.0
    val elapsed = _lastTrigger.map(t => (currentTime - t._1, step - t._2))
    _lastTrigger = Some((currentTime, step))
    elapsed
  }

  /** Returns the last triggered time step or `None`, if never triggered. */
  override def lastTriggerStep(): Option[Int] = _lastTrigger.map(_._2)
}

/** Hook trigger that triggers at most once every `numSeconds` seconds.
  *
  * @param  numSeconds Triggering time frequency.
  */
case class TimeHookTrigger(numSeconds: Double) extends HookTrigger {
  require(numSeconds >= 0, s"'numSeconds' (= $numSeconds) must be a non-negative number.")

  /** Returns a copy of this hook trigger that is also reset. */
  override def copy(): HookTrigger = TimeHookTrigger(numSeconds)

  private[this] var _lastTrigger: Option[(Double, Int)] = None

  /** Resets the internal state of this trigger (e.g., step counter or timer). */
  override def reset(): Unit = _lastTrigger = None

  /** Returns `true` if the hook should be triggered for the specified step. */
  override def shouldTriggerForStep(step: Int): Boolean = _lastTrigger match {
    case None => true
    case Some((_, s)) if s == step => false
    case Some((t, _)) => (System.currentTimeMillis().toDouble / 1000.0) >= t + numSeconds
  }

  /** Updates the last triggered step and time.
    *
    * @param  step Current step.
    * @return A tuple `(elapsedTime, elapsedSteps)`, where `elapsedTime` is the number of seconds between the current
    *         trigger and the last one, and `elapsedSteps` is the number of steps between the current trigger and the
    *         last one. Both values will be set to `None` on the first trigger.
    */
  override def updateLastTrigger(step: Int): Option[(Double, Int)] = {
    val currentTime = System.currentTimeMillis().toDouble / 1000.0
    val elapsed = _lastTrigger.map(t => (currentTime - t._1, step - t._2))
    _lastTrigger = Some((currentTime, step))
    elapsed
  }

  /** Returns the last triggered time step or `None`, if never triggered. */
  override def lastTriggerStep(): Option[Int] = _lastTrigger.map(_._2)
}
