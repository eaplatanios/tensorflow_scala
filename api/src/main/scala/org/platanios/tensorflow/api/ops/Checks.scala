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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.types.{INT32, STRING}

/** Contains functions for constructing ops related to checks and assertions.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Checks {
  /** $OpDocCheckAssert
    *
    * @group CheckOps
    * @param  condition Condition to assert.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assert(condition: Output, data: Seq[Output], summarize: Int = 3, name: String = "Assert"): Op = {
    if (data.forall(d => d.dataType == STRING || d.dataType == INT32)) {
      // As a simple heuristic, we assume that STRING and INT32 tensors are on host memory to avoid the need to use
      // `cond`. If that is not case, we will pay the price copying the tensor to host memory.
      Op.Builder("Assert", name)
          .addInput(condition)
          .addInputList(data)
          .setAttribute("summarize", summarize)
          .build()
    } else {
      Op.createWithNameScope(name) {
        ControlFlow.cond(
          condition,
          () => ControlFlow.noOp(),
          () => Op.Builder("Assert", name)
              .addInput(condition)
              .addInputList(data)
              .setAttribute("summarize", summarize)
              .build(),
          name = "AssertGuard")
      }
    }
  }

  /** $OpDocCheckAssertEqual
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertEqual(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertEqual"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' == 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.equal(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertNoneEqual
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertNoneEqual(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertNoneEqual"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' != 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.notEqual(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertNear
    *
    * @group CheckOps
    * @param  x            First input tensor.
    * @param  y            Second input tensor.
    * @param  relTolerance Comparison relative tolerance value.
    * @param  absTolerance Comparison absolute tolerance value.
    * @param  message      Optional message to include in the error message, if the assertion fails.
    * @param  data         Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize    Number of tensor entries to print.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  def assertNear(
      x: Output, y: Output, relTolerance: Output = 0.00001f, absTolerance: Output = 0.00001f,
      message: Output = null, data: Seq[Output] = null, summarize: Int = 3, name: String = "AssertNear"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(
              s"'x' and 'y' are not nearly equal element-wise:",
              s" x (${x.name}) = ", x,
              s" y (${y.name}) = ", y,
              s" relative tolerance (${relTolerance.name}) = ", relTolerance,
              s" absolute tolerance (${absTolerance.name}) = ", absTolerance)
          }
        }
        if (message != null) message +: d else d
      }
      val tolerance = absTolerance + relTolerance * Math.abs(y)
      val difference = Math.abs(x - y)
      assert(Math.all(Math.less(difference, tolerance)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertLess
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertLess(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertLess"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' < 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.less(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertLessEqual
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertLessEqual(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertLessEqual"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' <= 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.lessEqual(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertGreater
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertGreater(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertGreater"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' > 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.greater(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertGreaterEqual
    *
    * @group CheckOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertGreaterEqual(
      x: Output, y: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertGreaterEqual"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'x' >= 'y' did not hold element-wise: x (${x.name}) = ", x, s" y (${y.name}) = ", y)
          }
        }
        if (message != null) message +: d else d
      }
      assert(Math.all(Math.greaterEqual(x, y)), processedData, summarize)
    }
  }

  /** $OpDocCheckAssertPositive
    *
    * @group CheckOps
    * @param  input     Input tensor to check.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertPositive(
      input: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertPositive"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'input' > 0 did not hold element-wise: input (${input.name}) = ", input)
          }
        }
        if (message != null) message +: d else d
      }
      assertLess(0, input, data = processedData, summarize = summarize)
    }
  }

  /** $OpDocCheckAssertNegative
    *
    * @group CheckOps
    * @param  input     Input tensor to check.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertNegative(
      input: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertNegative"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'input' < 0 did not hold element-wise: input (${input.name}) = ", input)
          }
        }
        if (message != null) message +: d else d
      }
      assertLess(input, 0, data = processedData, summarize = summarize)
    }
  }

  /** $OpDocCheckAssertNonPositive
    *
    * @group CheckOps
    * @param  input     Input tensor to check.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertNonPositive(
      input: Output, message: Output = null, data: Seq[Output] = null, summarize: Int = 3,
      name: String = "AssertNonPositive"): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'input' <= 0 did not hold element-wise: input (${input.name}) = ", input)
          }
        }
        if (message != null) message +: d else d
      }
      assertLessEqual(input, 0, data = processedData, summarize = summarize)
    }
  }

  /** $OpDocCheckAssertNonNegative
    *
    * @group CheckOps
    * @param  input     Input tensor to check.
    * @param  message   Optional message to include in the error message, if the assertion fails.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assertNonNegative(
      input: Output,
      message: Output = null,
      data: Seq[Output] = null,
      summarize: Int = 3,
      name: String = "AssertNonNegative"
  ): Op = {
    Op.createWithNameScope(name) {
      val processedData = {
        val d: Seq[Output] = {
          if (data != null) {
            data
          } else {
            Seq(s"Condition 'input' >= 0 did not hold element-wise: input (${input.name}) = ", input)
          }
        }
        if (message != null) message +: d else d
      }
      assertLessEqual(0, input, data = processedData, summarize = summarize)
    }
  }

  /** $OpDocCheckAssertAtMostNTrue
    *
    * @group CheckOps
    * @param  predicates Sequence containing scalar boolean tensors, representing the predicates.
    * @param  n          Maximum number of predicates allowed to be `true`.
    * @param  message    Optional message to include in the error message, if the assertion fails.
    * @param  summarize  Number of tensor entries to print.
    * @param  name       Name for the created op.
    * @return Created op.
    */
  def assertAtMostNTrue(
      predicates: Seq[Output],
      n: Int,
      message: Output = null,
      summarize: Int = 3,
      name: String = "AssertAtMostNTrue"
  ): Op = {
    Op.createWithNameScope(name) {
      val stackedPredicates = Basic.stack(predicates, name = "StackPredicates")
      val numTrue = Math.sum(Math.cast(stackedPredicates, INT32), "NumTrue")
      val condition = Math.lessEqual(numTrue, Basic.constant(n, name = "NumTrueConditionsLimit"))
      val processedData = {
        val d: Seq[Output] = {
          val predicateNames = predicates.map(p => s"'${p.name}'").mkString(", ")
          Seq(s"More than $n conditions ($predicateNames) evaluated as `true`.", stackedPredicates)
        }
        if (message != null) message +: d else d
      }
      assert(condition, processedData, summarize)
    }
  }

  /** @define OpDocCheckAssert
    *   The `assert` op asserts that the provided condition is true.
    *
    *   If `condition` evaluates to `false`, then the op prints all the op outputs in `data`. `summarize` determines how
    *   many entries of the tensors to print.
    *
    *   Note that to ensure that `assert` executes, one usually attaches it as a dependency:
    *   {{{
    *     // Ensure maximum element of x is smaller or equal to 1.
    *     val assertOp = tf.assert(tf.lessEqual(tf.max(x), 1.0), Seq(x))
    *     Op.createWith(controlDependencies = Set(assertOp)) {
    *       ... code using x ...
    *     }
    *   }}}
    *
    * @define OpDocCheckAssertEqual
    *   The `assertEqual` op asserts that the condition `x == y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertEqual(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) == y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertNoneEqual
    *   The `assertNoneEqual` op asserts that the condition `x != y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertNoneEqual(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) != y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertNear
    *   The `assertNear` op asserts that `x` and `y` are close element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertNear(x, y, relTolerance, absTolerance)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `tf.abs(x(i) - y(i)) <= absTolerance + relTolerance * tf.abs(y(i))`. If both `x` and `y` are empty, it is
    *   trivially satisfied.
    *
    * @define OpDocCheckAssertLess
    *   The `assertLess` op asserts that the condition `x < y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertLess(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) < y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertLessEqual
    *   The `assertLessEqual` op asserts that the condition `x <= y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertLessEqual(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) <= y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertGreater
    *   The `assertGreater` op asserts that the condition `x > y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertGreater(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) > y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertGreaterEqual
    *   The `assertGreaterEqual` op asserts that the condition `x >= y` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertGreaterEqual(x, y)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   The condition is satisfied if for every pair of (possibly broadcast) elements `x(i)`, `y(i)`, we have
    *   `x(i) >= y(i)`. If both `x` and `y` are empty, it is trivially satisfied.
    *
    * @define OpDocCheckAssertPositive
    *   The `assertPositive` op asserts that the condition `input > 0` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertPositive(x)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   If `input` is an empty tensor, the condition is trivially satisfied.
    *
    * @define OpDocCheckAssertNegative
    *   The `assertNegative` op asserts that the condition `input < 0` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertNegative(x)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   If `input` is an empty tensor, the condition is trivially satisfied.
    *
    * @define OpDocCheckAssertNonPositive
    *   The `assertNonPositive` op asserts that the condition `input <= 0` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertNonPositive(x)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   If `input` is an empty tensor, the condition is trivially satisfied.
    *
    * @define OpDocCheckAssertNonNegative
    *   The `assertNonNegative` op asserts that the condition `input >= 0` holds element-wise.
    *
    *   Example usage:
    *   {{{
    *     val output = tf.createWith(controlDependencies = Set(tf.assertNonNegative(x)) {
    *       x.sum()
    *     }
    *   }}}
    *
    *   If `input` is an empty tensor, the condition is trivially satisfied.
    *
    * @define OpDocCheckAssertAtMostNTrue
    *   The `assertAtMostNTrue` op asserts that at most `n` of the provided predicates can evaluate to `true` at the
    *   same time.
    */
  private[ops] trait Documentation
}

private[api] object Checks extends Checks
