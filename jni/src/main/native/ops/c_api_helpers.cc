#include "tensorflow/c/c_eager_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/c_eager_api_internal.h"
#include "tensorflow/c/c_eager_api_runtime.h"
#ifdef TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#endif  // TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
  // This traverses the specified nodes in topological order to verify there are
  // no cycles. Starting with inputless nodes, it visits nodes whose inputs have
  // all been visited, and counts the total number of visited nodes. If there is a
  // cycle, nodes in the cycle will never be visited, and the visited count will
  // be less than the total node count.
  Status ValidateNoCycles(const Graph& g) {
    // TODO(nolivia): check this on a subset of the graph instead of all of it.
    // A node is ready when all of its inputs have been visited.
    std::vector<const Node*> ready;
    std::vector<int> pending_count(g.num_node_ids(), 0);

    for (int i = 0; i < g.num_node_ids(); ++i) {
      const Node* n = g.FindNodeId(i);
      if (n == nullptr) continue;
      pending_count[i] = n->in_edges().size();
      if (n->IsMerge()) {
        // While-loop cycles are legal cycles so we manually adjust the
        // pending_count to make sure that the loop is visited.
        for (const Edge* e : n->in_edges()) {
          if (!e->IsControlEdge() && e->src()->IsNextIteration()) {
            pending_count[i]--;
          }
        }
      }
      if (pending_count[i] == 0) {
        ready.push_back(n);
      }
    }

    int processed = 0;
    while (!ready.empty()) {
      const Node* node = ready.back();
      ready.pop_back();
      ++processed;

      for (const Edge* out : node->out_edges()) {
        const int output_id = out->dst()->id();
        pending_count[output_id]--;
        if (pending_count[output_id] == 0) {
          ready.push_back(out->dst());
        }
      }
    }

    if (processed < g.num_nodes()) {
      std::vector<string> nodes_in_cycle;
      for (int i = 0; i < pending_count.size() && nodes_in_cycle.size() < 3;
           ++i) {
        if (pending_count[i] != 0) {
          nodes_in_cycle.push_back(g.FindNodeId(i)->name());
        }
      }
      return errors::InvalidArgument(
          "Graph is invalid, contains a cycle with ", g.num_nodes() - processed,
          " nodes, including: ", str_util::Join(nodes_in_cycle, ", "));
    }
    return Status::OK();
  }

  // TODO(josh11b,mrry): Change Session to be able to use a Graph*
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  bool ExtendSessionGraphHelper(TF_Session* session, TF_Status* status) {
    if (session->graph != nullptr) {
      // Take the graph lock before the session lock to avoid deadlock. This is
      // safe since session->graph does not change.
      session->graph->mu.lock();
      mutex_lock session_lock(session->mu);
      const Graph& graph = session->graph->graph;

      status->status = session->graph->sessions[session];
      if (!status->status.ok()) {
        session->graph->mu.unlock();
        return false;
      }

      const auto num_nodes = graph.num_node_ids();
      if (session->last_num_graph_nodes < num_nodes) {
        status->status = tensorflow::ValidateNoCycles(session->graph->graph);
        if (!status->status.ok()) {
          session->graph->mu.unlock();
          return false;
        }

        GraphDef graph_def;
        *graph_def.mutable_versions() = graph.versions();
        // Fill graph_def with nodes with ids in the range
        // [session->last_num_graph_nodes, num_nodes), that is the nodes
        // added since the last TF_SessionRun() call.
        for (auto id = session->last_num_graph_nodes; id < num_nodes; ++id) {
          Node* const node = graph.FindNodeId(id);
          if (node != nullptr && node->IsOp()) {
            NodeDef* const node_def = graph_def.add_node();
            *node_def = node->def();
          }
        }
        *graph_def.mutable_library() = graph.flib_def().ToProto();
        session->graph->mu.unlock();
        status->status = session->session->Extend(graph_def);
        if (!status->status.ok()) {
          // Contract is we always delete input_values[i].
          return false;
        }
        // Note: session->session is not modified if Extend() fails, so
        // we only set last_num_graph_nodes if it succeeds.
        session->last_num_graph_nodes = num_nodes;
      } else {
        session->graph->mu.unlock();
      }
    }
    return true;
  }
} // namespace tensorflow

tensorflow::Status TFE_Executor::WaitFor(tensorflow::uint64 node_id) {
  return WaitImpl(false, node_id);
}

tensorflow::Status TFE_Executor::WaitForAllPendingNodes() {
  return WaitImpl(true, 0);
}

tensorflow::Status TFE_Executor::WaitImpl(bool wait_all,
                                          tensorflow::uint64 node_id) {
  tensorflow::condition_variable cond;
  tensorflow::mutex_lock l(node_queue_mutex_);
  // Don't wait if an error is already set.
  if (!status_.ok()) return status_;
  if (node_queue_.empty()) return tensorflow::Status::OK();
  if (wait_all) {
    node_id = node_queue_.back()->id;
  } else if (node_id < node_queue_.front()->id) {
    // Note that we are relying on the ops being dispatched sequentially from
    // the queue.
    return tensorflow::Status::OK();
  }
  node_done_notifications_.insert(std::make_pair(node_id, &cond));
  cond.wait(l);
  // Note that we could be woken up if an error occurs, even though the node has
  // not actually executed.
  return status_;
}

bool TFE_TensorHandle::IsReady() {
  if (node_id == 0) return true;
  tensorflow::mutex_lock l(ctx_mutex_);
  return ctx_ == nullptr;
}

tensorflow::Status TFE_TensorHandle::WaitReady() {
  if (node_id == 0) return tensorflow::Status::OK();
  TFE_Executor* executor = nullptr;
  {
    tensorflow::mutex_lock l(ctx_mutex_);
    if (ctx_ == nullptr) return tensorflow::Status::OK();
    executor = &ctx_->executor;
  }
  return executor->WaitFor(node_id);
}

tensorflow::Status TFE_TensorHandle::Tensor(const tensorflow::Tensor** t) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *t = &tensor_;
  return tensorflow::Status::OK();
}

tensorflow::Status TFE_TensorHandle::Device(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = device_;
  return tensorflow::Status::OK();
}

tensorflow::Status TFE_TensorHandle::OpDevice(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = op_device_;
  return tensorflow::Status::OK();
}

tensorflow::Status TFE_TensorHandle::TensorAndDevice(
    const tensorflow::Tensor** tensor, tensorflow::Device** device,
    tensorflow::Device** op_device) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *tensor = &tensor_;
  *device = device_;
  *op_device = op_device_;
  return tensorflow::Status::OK();
}

void TFE_TensorHandle::SetTensorAndDevice(const tensorflow::Tensor& tensor,
                                          tensorflow::Device* device,
                                          tensorflow::Device* op_device) {
  tensorflow::mutex_lock l(ctx_mutex_);
  DCHECK(node_id > 0 && ctx_) << "SetTensorAndDevice should be only called  "
                              << "on non-ready handles.";
  ctx_ = nullptr;
  tensor_ = tensor;
  device_ = device;
  op_device_ = op_device;
}
