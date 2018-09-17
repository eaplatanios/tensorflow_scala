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

package org.platanios.tensorflow.api.ops.training.distribute.strategies

import org.platanios.tensorflow.api.core.{DeviceSpecification, Graph, Shape}
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.values._
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output, OutputLike}
import org.platanios.tensorflow.api.types.{DataType, FLOAT32}

/** Represents a list of devices with a state and a compute distribution policy.
  *
  * The intent is that you can write an algorithm in a stylized way and it will be usable with a variety of different
  * `DistributionStrategy` implementations. Each descendant will implement a different strategy for distributing the
  * algorithm across multiple devices/machines. Furthermore, these changes can be hidden inside the specific layers and
  * other library classes that need special treatment to run in a distributed setting, so that most users' model
  * definition code can run unchanged.
  *
  * First let's introduce a few high-level concepts:
  *
  *   - ''Data parallelism'' is where we run multiple copies of the model on different slices of the input data. This is
  *     in contrast to ''model parallelism'' where we divide up a single copy of a model across multiple devices. Note
  *     that we only support data parallelism at this time, but plan to add support for model parallelism in the future.
  *   - A ''tower'' is one copy of the model, running on one slice of the input data.
  *   - ''Synchronous'', or more commonly ''sync'', training is when the updates from each tower are aggregated together
  *     before updating the model variables. This is in contrast to ''asynchronous'', or ''async'' training where each
  *     tower updates the model variables independently.
  *   - Furthermore you might run your computation on multiple devices on one machine (or "host"), or on multiple
  *     machines/hosts. If you are running on multiple machines, you might have a single master host that drives
  *     computation across all of them, or you might have multiple clients driving the computation asynchronously.
  *
  * To distribute an algorithm, we might use some of these ingredients:
  *
  *   - ''Parameter servers'': These are hosts that hold a single copy of parameters/variables. All towers that want to
  *     operate on a variable retrieve it at the beginning of a step and send an update to be applied at the end of the
  *     step. Can support either sync or async training.
  *   - ''Mirrored variables'': These are variables that are copied to multiple devices, where we keep the copies in
  *     sync by applying the same updates to every copy. Normally would only be used with sync training.
  *   - ''Reductions'': A reduction is a method of aggregating multiple values into one value, like "sum" or "mean". If
  *     doing sync training, we will perform a reduction on the gradients to a parameter from each tower before applying
  *     the update. `Allreduce` is an algorithm for performing a reduction on values from multiple devices and making
  *     the result available on all of those devices.
  *   - In the future we will have support for TensorFlows' partitioned variables, where a single variable is split
  *     across multiple devices.
  *
  * We have a few approaches we want to support:
  *
  *   - Code written (as if) with no knowledge of class `DistributionStrategy`. This code should work as before, even if
  *     some of the layers, etc., used by that code are written to be distribution-aware. This is done by having a
  *     default `DistributionStrategy` that gives ordinary behavior, and by default being in a single tower context.
  *   - Ordinary model code that you want to run using a specific `DistributionStrategy`. This can be as simple as:
  *     {{{
  *       d.scope {
  *         val iterator = d.distributeDataset(dataset)
  *         val towerTrainOps = d.forEachTower(towerFn, iterator.next())
  *         val trainOp = tf.group(d.unwrap(towerTrainOps))
  *       }
  *     }}}
  *     This takes an ordinary `dataset` and `towerFn` and runs it distributed using a particular `DistributionStrategy`
  *     in `d`. Any variables created in `towerFn` are created using `d`'s policy, and library functions called by
  *     `towerFn` can use the `currentTowerContext` API to get enhanced behavior in this case. Note that in the future
  *     we will add support for initializable dataset iterators, at which point this example code will change.
  *   - If you want to write a distributed algorithm, you may use any of the `DistributionStrategy` APIs inside a
  *     `d.scope` block of code.
  *
  * Lower-level concepts:
  *
  *   - ''Wrapped values'': In order to represent values parallel across devices (either towers or the devices
  *     associated with a particular value), we wrap them in a `PerDevice` or `Mirrored` object that contains a map from
  *     device to values. `PerDevice` is used when the value may be different across devices, and `Mirrored` is used
  *     when the value is the same across devices.
  *   - ''Unwrapping and merging'': Consider calling a function `fn` on multiple devices, like `forEachTower(fn, w)`
  *     with an argument `w` that is a wrapped value. This means that `w` will have a map taking tower device `d0` to
  *     `w0`, tower device `d1` to `w1`, etc. `forEachTower()` unwraps `w` before calling `fn`, and so it calls `fn(w0)`
  *     on `d0`, `fn(w1)` on `d1`, etc. It then merges the return values from `fn()`, which can possibly result in
  *     wrapped values. For example, let's say `fn()` returns a tuple with three components: `(x, a, v0)` from tower 0,
  *     `(x, b, v1)` from tower 1, etc. If the first component is the same object `x` for every tower, then the first
  *     component of the merged result will also be `x`. If the second component is different (`a`, `b`, ...) for each
  *     tower, then the merged value will have a wrapped map from tower device to the different values. If the third
  *     component is the members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to `v1`, etc.), then the merged
  *     result will be that mirrored variable (i.e., `v`).
  *   - ''Tower context vs. cross-tower context'': Tower context is when we are in some function that is being called
  *     once for each tower. Otherwise, we are in a cross-tower context, which is useful for calling
  *     `DistributionStrategy` methods which operate across towers (like `reduce()`). By default you start in a tower
  *     context (the default "single tower context") and then some methods can switch you back and forth,
  *     as described below.
  *   - ''Worker devices vs. parameter devices'': Most tower computations will happen on worker devices. Since we do not
  *     yet support model parallelism, there will be one worker device per tower. When using parameter servers (see
  *     above), the set of devices holding variables may be different, otherwise the parameter devices might match the
  *     worker devices.
  *   - Non-slot devices are some subset of the parameter devices where we put all the non-slot variables. We need to
  *     ensure that all non-slot variables are allocated on the same device, or mirrored across the same set of devices.
  *     If you have some variable that you want to colocate all the non-slot variables with, you can use
  *     `colocateVariablesWith()` to get the remaining non-slot variables on the same device. Otherwise, you can use
  *     `nonSlotDevices()` to pick a consistent set of devices to pass to both `colocateVariablesWith()` and
  *     `updateNonSlot()`.
  *
  * When using a `DistributionStrategy`, we have a new type dimension called ''locality'' that says what values are
  * compatible with which APIs:
  *
  *   - `T`: Different value for each tower (e.g., a `PerDevice`-wrapped value).
  *   - `M`: Value is "mirrored" across towers. That is, there are copies with the same value on each tower (e.g., a
  *     `Mirrored`-wrapped value).
  *   - `V(v)`: Value is "mirrored" across all the devices which have a copy of variable `v` (also a `Mirrored`-wrapped
  *     value, but over parameter devices instead of worker devices).
  *   - `N`: Value is "mirrored" across all the "non-slot" devices.
  *
  * Rules for methods with respect to locality and single-tower vs. cross-tower context:
  *
  *   - `d.scope()`: Default single-tower context -> cross-tower context for `d`.
  *   - `d.colocateVariablesWith(v)`: In tower/cross-tower context, variables will be created with locality `V(v)`. That
  *     is, if we write `d.colocateVariablesWith(v1) { val v2 = tf.variable(...) }`, then `v2` will have locality
  *     `V(v1)` (i.e., locality `V(v2)` will equal `V(v1)`).
  *   - `d.colocateVariablesWith(d.nonSlotDevices(...))`: In tower/cross-tower context, variables will be created with
  *     locality `N`.
  *   - `v = tf.variable(...)`: In tower/cross-tower context, creates a variable (which by definition will have locality
  *     `V(v)`, though will match another locality if inside a `colocateVariablesWith()` scope).
  *   - `d.distributeDataset(dataset)`: In cross-tower context, produces an iterator with locality `T`.
  *   - `d.broadcast(t)`: In cross-tower context, produces a value with locality `M`.
  *   - `d.broadcast(t, v)`: In cross-tower context, produces a value with locality `V(v)`.
  *   - `d.forEachTower(fn, ...)`: In cross-tower context, runs `fn()` in a tower context (and so may call
  *     `currentTowerContext` and use its API, including `mergeCall()` to get back to cross-tower context), once for
  *     each tower. May use values with locality `T` or `M`, and any variable.
  *   - `d.reduce(m, t)`: In cross-tower context, accepts `t` with locality `T` and produces a value with locality `M`.
  *   - `d.reduce(m, t, v)`: In cross-tower context, accepts `t` with locality `T` and produces a value with locality
  *     `V(v)`.
  *   - `d.batchReduce(m, Seq((t, v)))`: See `d.reduce()`.
  *   - `d.update(v, fn, ...)`: In cross-tower context, runs `fn()` once for each device `v` is copied to. All inputs
  *     should have locality `V(v)`, and the output will have locality `V(v)` as well.
  *   - `d.updateNonSlot(d.nonSlotDevices(), fn)`: In cross-tower context, like `d.update()` except with locality `N`.
  *   - `d.fetch(t)`: Copy `t` with any locality to the client's CPU device.
  *
  * The standard pattern for updating variables is to:
  *
  *   1. Wrap your input dataset in `d.distributeDataset()`.
  *   2. Define each tower `d.forEachTower()` up to the point of getting a list of gradient, variable pairs.
  *   3. Call `d.reduce("sum", t, v)` or `d.batchReduce()` to sum the gradients (with locality `T`) into values with
  *      locality `V(v)`.
  *   4. Call `d.update(v)` for each variable to update its value.
  *
  * Steps 3 and 4 are done automatically by the `Optimizer` class if you call its `applyGradients` method from within a
  * tower context. Otherwise, you can manually call its `distributedApply` method in a cross-tower context.
  *
  * Another thing you might want to do in the middle of your tower function is an all-reduce of some intermediate value,
  * using `d.reduce()` or `d.batchReduce()` without supplying a variable as the destination.
  *
  * Layers should expect to be called in a tower context, and can use the `currentTowerContext` function to get a
  * `TowerContext` object. The `TowerContext` object has a `mergeCall()` method for entering cross-tower context where
  * you can use `reduce()` (or `batchReduce()`) and then optionally `update()` to update state.
  *
  * You may use this API whether or not a `DistributionStrategy` is being used, since there is a default implementation
  * of `TowerContext` and `DistributionStrategy`. Or you can use the `currentTowerContext.isSingleTower` property to run
  * different code in the distributed vs. single tower cases.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class DistributionStrategy {
  //  # TODO(josh11b): Raise an exception if variable partitioning requested before
  //  #   we add support.
  //  # TODO(josh11b): Also `parameter_device_index` property?
  //  # TODO(josh11b): `map()`
  //  # TODO(josh11b): ClusterSpec/ClusterResolver
  //  # TODO(josh11b): Partitioned computations, state; sharding
  //  # TODO(josh11b): Model parallelism: "towers" with multiple devices; shuffling
  //  # TODO(josh11b): List of towers with their worker and parameter devices
  //  #   (where the parameter devices may overlap in the ps case).

  /** Finds and sets the best configuration for the provided TensorFlow session configuration. */
  def configure(sessionConfig: SessionConfig): Unit = ()

  def scope[R](block: => R): R = {
    val getter = new VariableGetter {
      def apply(
          name: String,
          dataType: DataType = FLOAT32,
          shape: Shape = null,
          initializer: Initializer = null,
          regularizer: Regularizer = null,
          trainable: Boolean = true,
          reuse: Reuse = ReuseOrCreateNew,
          collections: Set[Graph.Key[Variable]] = Set.empty,
          cachingDevice: OpSpecification => String = null,
          underlyingGetter: VariableGetter = null
      ): Variable = {
        createVariable(
          name, dataType, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice,
          underlyingGetter)
      }
    }

    // TODO: [VARIABLES/DISTRIBUTE] Disable partitioned variables when getters support them.

    implicit val context: CrossTowerContext = CrossTowerContext(this)
    val result = Variable.getter(createVariable) {
      VariableScope.updatedScope(underlyingGetter = getter) {
        block
      }
    }
    result
  }

  protected def createVariable: ColocatedVariableGetter

  /** Executes `block` within a scope where new variables will not be mirrored.
    *
    * There will still be one component variable per tower, but there is no requirement that they stay in sync. Instead,
    * when saving them or calling `fetch()`, we use the value that results when calling `reduce()` on all the towers'
    * variables. Note that tower-local implies not trainable. Instead, it is expected that each tower will directly
    * update (e.g., using `assignAdd()`) its local variable instance but only the aggregated value (accessible using
    * `fetch()`) will be exported from the model. When it is acceptable to only aggregate on export, we greatly reduce
    * communication overhead by using tower-local variables.
    *
    * Note that all component variables will be initialized to the same value, using the initialization expression from
    * the first tower. The values will match even if the initialization expression uses random numbers.
    *
    * @param  reduction Reduction method used to get the value to save when creating checkpoints.
    * @param  block     Code block to execute in this scope.
    * @return Value returned by `block`.
    */
  def towerLocalVariableScope[R](reduction: Reduction)(block: => R)(implicit context: DistributionContext): R = {
    // TODO: [DISTRIBUTE] !!! Shouldn't this require an in-tower context?
    val towerLocalVariableGetter = new ReductionVariableGetter(reduction) {
      override def apply(
          name: String,
          dataType: DataType,
          shape: Shape,
          initializer: Initializer,
          regularizer: Regularizer,
          trainable: Boolean,
          reuse: Reuse,
          collections: Set[Graph.Key[Variable]],
          cachingDevice: OpSpecification => String,
          underlyingGetter: VariableGetter
      ): Variable = {
        underlyingGetter(
          name, dataType, shape, initializer, regularizer, trainable = false, reuse, collections, cachingDevice,
          null)
      }
    }

    Variable.getter(towerLocalVariableGetter) {
      block
    }
  }

  /** Executes `block` within a scope that controls which devices variables will be created on.
    *
    * No operations should be added to the graph inside this scope; it should only be used when creating variables (some
    * implementations work by changing variable creation and others work by using a `colocateWith` scope). This may only
    * be used inside `DistributionStrategy.scope`.
    *
    * For example:
    * {{{
    *   distributionStrategy.scope {
    *     val variable1 = tf.variable(...)
    *     distributionStrategy.colocateVariablesWith(Set(variable1.op)) {
    *       // `variable2` and `variable3` will be created on the same device(s) as `variable1`.
    *       val variable2 = tf.variable(...)
    *       val variable3 = tf.variable(...)
    *     }
    *
    *     def fn(v1: Variable, v2: Variable, v3: Variable): Unit = {
    *       // Operates on `v1` from `variable1`, `v2` from `variable2`, and `v3` from `variable3`.
    *     }
    *
    *     // `fn` runs on every device `v1` is on, and `v2` and `v3` will be there too.
    *     distributionStrategy.update(variable1, fn, variable2, variable3)
    *   }
    * }}}
    *
    * @param  colocationOps Variables created in `block` will be on the same set of devices as these ops.
    * @param  block         Code block to execute in this scope.
    * @return Value returned by `block`.
    */
  def colocateVariablesWith[R](colocationOps: Set[Op])(block: => R)(implicit context: DistributionContext): R = {
    // TODO: [DISTRIBUTE] Is the argument type correct?
    val colocatedVariableGetter = new ColocatedVariableGetter(colocationOps) {
      override def apply(
          name: String,
          dataType: DataType,
          shape: Shape,
          initializer: Initializer,
          regularizer: Regularizer,
          trainable: Boolean,
          reuse: Reuse,
          collections: Set[Graph.Key[Variable]],
          cachingDevice: OpSpecification => String,
          underlyingGetter: VariableGetter
      ): Variable = {
        Op.colocateWith(colocationOps, ignoreExisting = true) {
          underlyingGetter(
            name, dataType, shape, initializer, regularizer, trainable = false, reuse, collections, cachingDevice,
            null)
        }
      }
    }

    Variable.getter(colocatedVariableGetter) {
      block
    }
  }

  //region Cross-Tower Context Methods

  // TODO: [DISTRIBUTE] `distributeDataset`.

  /** Mirrors `value` to all worker devices.
    *
    * @param  value   Value to broadcast.
    * @param  devices Destination devices.
    * @return Mirrored value.
    */
  def broadcast[O <: OutputLike](
      value: O,
      devices: Seq[DeviceSpecification] = Seq.empty
  )(implicit context: CrossTowerContext): MirroredValue[O]

  /** Runs `fn` once per tower.
    *
    * `fn` may call `tf.currentTowerContext` to access fields and methods such as `towerID` and `mergeCall()`.
    * `mergeCall()` is used to communicate between the towers and re-enter the cross-tower context. All towers pause
    * their execution having encountered a `mergeCall()` call. After that the `mergeFn`-function is executed. Its
    * results are then unwrapped and given back to each tower call. After that execution resumes until `fn` is complete
    * or another `mergeCall()` is encountered.
    *
    * For example:
    * {{{
    *   // Called once in "cross-tower" context.
    *   def mergeFn(distributionStrategy: DistributionStrategy, threePlusTowerID: Int): tf.Output = {
    *     // Sum the values across towers.
    *     tf.addN(distribution.unwrap(threePlusTowerID))
    *   }
    *
    *   // Called once per tower in `distributionStrategy`, in a "tower" context.
    *   def fn(three: Int): Output = {
    *     val towerContext = tf.currentTowerContext
    *     val v = three + towerContext.towerID
    *     // Computes the sum of the `v` values across all towers.
    *     val s = towerContext.mergeCall(mergeFn(_, v))
    *     s + v
    *   }
    *
    *   distributionStrategy.scope {
    *     // In "cross-tower" context
    *     ...
    *     val mergedResults = distributionStrategy.forEachTower(() => fn(3))
    *     // `mergedResults` has the values from every tower execution of `fn`.
    *     val resultsList = distributionStrategy.unwrap(mergedResults)
    *   }
    * }}}
    *
    * @param  fn     Function that will be run once per tower.
    * @param  values Wrapped values that will be unwrapped when invoking `fn` on each tower.
    * @return Merged return value of `fn` across all towers.
    */
  def forEachTower[T: Distributable, R](
      fn: Seq[T] => R,
      values: Seq[DistributedValue[T]]
  )(implicit context: CrossTowerContext): R

  /** Combines values across towers into one value.
    *
    * @param  reduction   Reduction method to use.
    * @param  value       Value to reduce.
    * @param  destination Optional destination on which to copy the reduced value.
    * @return Reduced value.
    */
  def reduce[D: Destination](
      reduction: Reduction,
      value: PerDeviceValue[OutputLike],
      destination: Option[D] = None
  )(implicit context: CrossTowerContext): MirroredValue[OutputLike]

  /** Combines multiple `reduce` calls into one for faster execution.
    *
    * @param  reduction             Reduction method to use.
    * @param  valueDestinationPairs Sequence of values to reduce pairs with destinations to copy the reduced values to.
    * @return Reduced values.
    */
  def batchReduce[D: Destination](
      reduction: Reduction,
      valueDestinationPairs: Seq[(PerDeviceValue[OutputLike], Option[D])]
  )(implicit context: CrossTowerContext): Seq[DistributedValue[OutputLike]] = {
    valueDestinationPairs.map(v => reduce(reduction, v._1, v._2))
  }

  /** Runs `fn` to update `variable` using inputs mirrored to the same devices.
    *
    * If `variable` is mirrored across multiple devices, then this method implements logic like:
    * {{{
    *   val results = variable.index.map {
    *     case (deviceSpec, variable) => tf.createWith(device = deviceSpec.toString) {
    *       fn(variable)
    *     }
    *   }
    *   merged(results)
    * }}}
    *
    * Otherwise this returns `fn(variable)` colocated with `variable`.
    *
    * @param  variable  Variable to update.
    * @param  fn        Update function to use.
    * @param  arguments Mirrored arguments that should be passed to `fn`.
    * @return Merged return value of `fn` across all towers.
    */
  def update[T: Distributable, R: Distributable](
      variable: MirroredVariable,
      fn: (Variable, Seq[T]) => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R]

  /** Runs `fn` on the devices specified by `colocateWith`, with the provided arguments.
    *
    * @param  colocateWith Destination on which to execute `fn`.
    * @param  fn           Function to use for the update.
    * @param  arguments    Mirrored arguments that should be passed to `fn`.
    * @return Merged return value of `fn` across all towers.
    * @throws InvalidArgumentException If the provided `colocateWith` argument is invalid (e.g., too many devices).
    */
  @throws[InvalidArgumentException]
  def updateNonSlot[D: Destination, T: Distributable, R: Distributable](
      colocateWith: D,
      fn: Seq[T] => R,
      arguments: Seq[MirroredValue[T]]
  )(implicit context: CrossTowerContext): MirroredValue[R]

  /** Returns a copy of `fn(variable.value)` on `destination`. This is useful for getting a mirrored variable value onto
    * a device. The method will attempt to avoid a copy by checking if the value is already on the destination device.
    *
    * @param  variable    Variable (which may be mirrored) to copy and fetch.
    * @param  destination Device to copy the variable value to.
    * @param  fn          Optional function to apply to the value on the source device, before copying.
    * @return Fetched value in `device`.
    * @throws InvalidArgumentException If there is an issue with the provided variable.
    */
  @throws[InvalidArgumentException]
  def fetch(
      variable: DistributedVariable,
      destination: String = "/device:CPU:0",
      fn: Output => Output = (o: Output) => o
  )(implicit context: CrossTowerContext): Output

  /** Returns the list of all per-device values contained in `value`.
    *
    * @param  value A value returned by `forEachTower()`, or a variable created in `scope`.
    * @return Sequence of values contained in `value`.
    */
  def unwrap[T: Distributable](
      value: DistributedValue[T]
  )(implicit context: CrossTowerContext): Seq[T]

  /** Acts as a shortcut for `tf.group(distributionStrategy.unwrap(value))`.
    *
    * @param  value A value returned by `forEachTower()`, or a variable created in `scope`.
    * @param  name  Name for the created op.
    * @return Grouped unwrapped `value`.
    */
  def group[T: Distributable](
      value: DistributedValue[T],
      name: String = "Group"
  )(implicit context: CrossTowerContext): Op = {
    ControlFlow.group(unwrap(value).map(implicitly[Distributable[T]].op(_)).toSet, name)
  }

  /** Returns a map from worker devices to indices.
    *
    * TODO: [DISTRIBUTE] Settle on the interface of `forEachTower()` first.
    * This map might be passed as an argument to `forEachTower()`, as in:
    * {{{
    *   distributionStrategy.scope {
    *     def fn(deviceIndex: Int): Unit = {
    *       // `fn` is being executed on device `distributionStrategy.workerDevices(deviceIndex)`.
    *     }
    *     distributionStrategy.forEachTower(fn, distributionStrategy.workerDeviceIndex)
    *   }
    * }}}
    */
  def workerDeviceIndex(implicit context: CrossTowerContext): Map[DeviceSpecification, Int]

  //endregion Cross-Tower Context Methods

  //region In-Tower Context Methods

  /** Merges arguments across towers and runs `mergeFn` in a cross-tower context.
    *
    * This allows communication and coordination when there are multiple calls to a model function triggered by a call
    * to `forEachTower(modelFn, ...)`. See `MirroredDistribution.forEachTower()` for an explanation.
    *
    * Otherwise, this is equivalent to:
    * {{{
    *   val strategy = tf.distribute.currentStrategy
    *   strategy.scope {
    *     mergeFn(strategy)
    *   }
    * }}}
    *
    * @param  mergeFn Merge function to invoke from within a cross-tower context.
    * @return Result of the `mergeFn` call, except for per-device values which are unpacked.
    */
  def mergeCall[R](mergeFn: DistributionStrategy => R)(implicit context: InTowerContext): R = {
    implicit val context: CrossTowerContext = CrossTowerContext(this)
    mergeFn(this)
  }

  //endregion In-Tower Context Methods

  /** Returns `true` if there is only a single tower, and `false`, otherwise.
    *
    * If `true`, `forEachTower(fn)` will only call `fn` once.
    * If `false`, `forEachTower(fn)` may call `fn` multiple times.
    */
  def isSingleTower: Boolean

  /** Returns number of towers, for purposes of averaging across towers. */
  def numTowers: Int

  /** Returns the devices used to run `forEachTower()` calls. */
  def workerDevices: Set[String]

  /** Returns the devices used for variable and updates placement. */
  def parameterDevices: Set[String]

  /** Returns the devices used for non-slot variables.
    *
    * Create variables on these devices in a `colocateVariablesWith(nonSlotDevices(...)):` block. Then, update them
    * using `updateNonSlot()`.
    *
    * @param  variables Variables being optimized.
    * @return Colocation ops for non-slot variables.
    */
  def nonSlotDevices(variables: Seq[Variable]): Set[DeviceSpecification]
}
