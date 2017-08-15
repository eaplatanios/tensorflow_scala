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

package org.platanios.tensorflow

/**
  * @author Emmanouil Antonios Platanios
  */
package object api
    extends core.API
        with ops.API
        with tensors.API {
  private[api] val defaultGraph: core.Graph = core.Graph()

  //region Utilities

  trait Closeable {
    def close(): Unit
  }

  def using[T <: Closeable, R](resource: T)(block: T => R): R = {
    try {
      block(resource)
    } finally {
      if (resource != null)
        resource.close()
    }
  }

  private[api] val Disposer = utilities.Disposer

  type ProtoSerializable = utilities.Proto.Serializable

  //endregion Utilities

  //region Public Scoped API

  private[api] trait API
      extends core.ScopedAPI
          with ops.ScopedAPI
          with types.ScopedAPI {
    object learn extends api.learn.API
  }

  object tf extends API

  //endregion Public Scoped API
}
