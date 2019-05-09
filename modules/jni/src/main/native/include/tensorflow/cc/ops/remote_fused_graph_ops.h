// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_
#define TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup remote_fused_graph_ops Remote Fused Graph Ops
/// @{

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The outputs tensor.
class RemoteFusedGraphExecute {
 public:
  RemoteFusedGraphExecute(const ::tensorflow::Scope& scope,
                        ::tensorflow::InputList inputs, const DataTypeSlice&
                        Toutputs, StringPiece
                        serialized_remote_fused_graph_execute_info);
  ::tensorflow::Output operator[](size_t index) const { return outputs[index]; }


  Operation operation;
  ::tensorflow::OutputList outputs;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_
