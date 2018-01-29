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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.lookup.Lookup
import org.platanios.tensorflow.api.ops.variables.{Saver, Variable}

/** Structure used to create or gather pieces commonly needed to train a model.
  *
  * When you build a model for training you usually need ops to initialize variables, a [[Saver]] to checkpoint them, an
  * op to collect summaries for the visualizer, and so on.
  *
  * Various libraries built on top of the core TensorFlow library take care of creating some or all of these pieces and
  * storing them in well known collections in the graph. The [[SessionScaffold]] class helps pick these pieces from
  * graph collections, create them, and/or also add them to graph collections if needed.
  *
  * If you call the scaffold constructor without any arguments, it will pick pieces from the graph collections, creating
  * default ones if needed, when `SessionScaffold.build()` is called. You can pass arguments to the constructor to
  * provide your own pieces. Pieces that you pass to the constructor are not added to the graph collections.
  *
  * @param  readyOp             [[Output]] used to verify that the variables are initialized. Picked from and stored
  *                             into the `READY_OP` graph collection by default.
  * @param  readyForLocalInitOp [[Output]] used to verify that global state has been initialized and it is fine to
  *                             execute `localInitOp`. Picked from and stored into the `READY_FOR_LOCAL_INIT_OP` graph
  *                             collection by default.
  * @param  initOp              [[Op]] used to initialize the variables. Picked from and stored into the `INIT_OP` graph
  *                             collection by default.
  * @param  initFeedMap         Feed map that will be used when executing `initOp`.
  * @param  initFunction        Function to run after the init op to perform additional initializations.
  * @param  localInitOp         [[Op]] used to initialize the local variables. Picked from and stored into the
  *                             `LOCAL_INIT_OP` graph collection by default.
  * @param  localInitFunction   Function to run after the local init op to perform additional initializations.
  * @param  summaryOp           [[Output]] used to merge the summaries in the graph. Picked from and stored into the
  *                             `SUMMARY_OP` graph collection by default.
  * @param  saver               [[Saver]] object taking care of saving the variables. Picked from and stored into the
  *                             `SAVERS` graph collection by default.
  * @author Emmanouil Antonios Platanios
  */
case class SessionScaffold(
    readyOp: Option[Output] = None,
    readyForLocalInitOp: Option[Output] = None,
    initOp: Option[Op] = None,
    initFeedMap: FeedMap = FeedMap.empty,
    initFunction: Option[(Session, BuiltSessionScaffold) => Unit] = None,
    localInitOp: Option[Op] = None,
    localInitFunction: Option[(Session, BuiltSessionScaffold) => Unit] = None,
    summaryOp: Option[Output] = None,
    saver: Option[Saver] = None) {
  /** Creates any necessary operations, freezes the graph, and returns a new session scaffold that is built. */
  def build(): BuiltSessionScaffold = {
    val _readyOp = readyOp.getOrElse(getItemOrElse("ready_op", Graph.Keys.READY_OP, () => {
      Basic.concatenate(Seq(
        Variable.uninitializedVariables(),
        Resource.uninitializedResources()))
    }))
    val _readyForLocalInitOp = readyForLocalInitOp.getOrElse(getItemOrElse(
      "ready_for_local_init_op", Graph.Keys.READY_FOR_LOCAL_INIT_OP, () => {
        Variable.uninitializedVariables(Variable.globalVariables)
      }))
    val _initOp = initOp.getOrElse(getItemOrElse("init_op", Graph.Keys.INIT_OP, () => {
      ControlFlow.group(Set(
        Variable.initializer(Variable.globalVariables),
        Resource.initializer(Resource.sharedResources)))
    }))
    val _localInitOp = localInitOp.getOrElse(getItemOrElse("local_init_op", Graph.Keys.LOCAL_INIT_OP, () => {
      ControlFlow.group(Set(
        Variable.initializer(Variable.localVariables),
        Lookup.initializer(Lookup.initializers)))
    }))
    val _summaryOp = summaryOp.getOrElse(getItemOrElse(
      "summary_op", Graph.Keys.SUMMARY_OP, () => Summary.mergeAll().orNull))
    val _saver = saver.getOrElse(getItemOrElse(
      "saver", Graph.Keys.SAVERS, () => Saver(sharded = true, allowEmpty = true)))
    BuiltSessionScaffold(
      _readyOp, _readyForLocalInitOp, _initOp, initFeedMap, initFunction, _localInitOp, localInitFunction,
      Option(_summaryOp), Option(_saver))
  }

  /** Gets the specified item (by `name`) from a current graph collection, or creates it using `default`, if it cannot
    * be found.
    *
    * @param  name          Item name.
    * @param  collectionKey Collection key for that item.
    * @param  default       Function providing a default value for the item (potentially constructing that value).
    * @return Obtained or created item value.
    */
  private[this] def getItemOrElse[K](name: String, collectionKey: Graph.Key[K], default: () => K): K = {
    val collection = Op.currentGraph.getCollection(collectionKey)
    if (collection.size > 1) {
      throw new IllegalStateException(
        s"There exist more than one items in collection '${collectionKey.name}'. Please indicate which one to use by " +
            s"passing it to the 'SessionScaffold' constructor as: SessionScaffold($name = <item to use>).")
    } else if (collection.size == 1) {
      collection.head
    } else {
      val op = default()
      if (op != null)
        Op.currentGraph.addToCollection(op, collectionKey)
      op
    }
  }
}

/** Built session scaffold. */
case class BuiltSessionScaffold private[learn](
    readyOp: Output,
    readyForLocalInitOp: Output,
    initOp: Op,
    initFeedMap: FeedMap,
    initFunction: Option[(Session, BuiltSessionScaffold) => Unit],
    localInitOp: Op,
    localInitFunction: Option[(Session, BuiltSessionScaffold) => Unit],
    summaryOp: Option[Output],
    saver: Option[Saver] = None) {
  private[learn] val internalInitFunction: Option[(Session) => Unit] = {
    initFunction.map(f => (session: Session) => f(session, this))
  }

  private[learn] val internalLocalInitFunction: Option[(Session) => Unit] = {
    localInitFunction.map(f => (session: Session) => f(session, this))
  }
}
