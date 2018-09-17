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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.using

import org.scalatest.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class TextSuite extends JUnitSuite {
  @Test def testRegexReplaceRemovePrefix(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(Tensor("a:foo", "a:bar", "a:foo", "b:baz", "b:qux", "ca:b"))
      val output = Text.regexReplace(input, "^(a:|b:)", "", replaceGlobal = false)
      val session = Session()
      val result = session.run(fetches = output)
      assert(result.entriesIterator.toSeq.map(_.asInstanceOf[String]) == Seq("foo", "bar", "foo", "baz", "qux", "ca:b"))
    }
  }

  @Test def testRegexReplace(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(Tensor("aba\naba", "abcdabcde"))
      val output = Text.regexReplace(input, "a.*a", "(\\0)")
      val session = Session()
      val result = session.run(fetches = output)
      assert(result.entriesIterator.toSeq.map(_.asInstanceOf[String]) == Seq("(aba)\n(aba)", "(abcda)bcde"))
    }
  }

  @Test def testRegexReplaceEmptyMatch(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(Tensor("abc", "1"))
      val output = Text.regexReplace(input, "", "x")
      val session = Session()
      val result = session.run(fetches = output)
      assert(result.entriesIterator.toSeq.map(_.asInstanceOf[String]) == Seq("xaxbxcx", "x1x"))
    }
  }

  @Test def testRegexReplaceInvalidPattern(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(Tensor("abc", "1"))
      val output = Text.regexReplace(input, "A[", "x")
      val session = Session()
      assert(intercept[InvalidArgumentException](session.run(fetches = output)).getMessage ===
          "Invalid pattern: A[, error: missing ]: [\n\t " +
              "[[{{node RegexReplace}} = RegexReplace[replace_global=true, " +
              "_device=\"/job:localhost/replica:0/task:0/device:CPU:0\"](Constant, Constant_1, Constant_2)]]")
    }
  }

  @Test def testRegexReplaceGlobal(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val input = Basic.constant(Tensor("ababababab", "abcabcabc", ""))
      val output = Text.regexReplace(input, "ab", "abc", replaceGlobal = true)
      val session = Session()
      val result = session.run(fetches = output)
      assert(result.entriesIterator.toSeq.map(_.asInstanceOf[String]) == Seq("abcabcabcabcabc", "abccabccabcc", ""))
    }
  }
}
