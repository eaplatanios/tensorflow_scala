/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "python_api.h"

#include "c_api.h"
#include "c_api_internal.h"

namespace tensorflow {

void RecordMutation(TF_Graph* graph, const TF_Operation& op,
                    const char* mutation_type) {
  // If any session has already run this node_id, mark this session as
  // unrunnable.
  for (auto it : graph->sessions) {
    mutex_lock session_lock(it.first->mu);
    if (it.first->last_num_graph_nodes > op.node.id()) {
      it.second = strings::StrCat(
          "Operation '", op.node.DebugString(), "' was changed by ",
          mutation_type,
          " after it was run by a session. This mutation will have no effect, "
          "and will trigger an error in the future. Either don't modify "
          "nodes after running them or create a new session.");
    }
  }
}

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst, TF_Status* status) {
  mutex_lock l(graph->mu);
//  tensorflow::shape_inference::InferenceContext* ic =
//      graph->refiner.GetContext(&new_src.oper->node);
//
//  if (ic->num_outputs() <= new_src.index) {
//    status->status = tensorflow::errors::OutOfRange(
//        "Cannot update edge. Output index [", new_src.index,
//        "] is greater than the number of total outputs [", ic->num_outputs(),
//        "].");
//    return;
//  }
//  tensorflow::shape_inference::ShapeHandle shape = ic->output(new_src.index);
//
//  tensorflow::shape_inference::InferenceContext* ic_dst =
//      graph->refiner.GetContext(&dst.oper->node);
//  if (ic_dst->num_inputs() <= dst.index) {
//    status->status = tensorflow::errors::OutOfRange(
//        "Cannot update edge. Input index [", dst.index,
//        "] is greater than the number of total inputs [", ic_dst->num_inputs(),
//        "].");
//    return;
//  }
//  if (!ic_dst->MergeInput(dst.index, shape)) {
//    status->status = tensorflow::errors::InvalidArgument(
//        "Cannot update edge, incompatible shapes: ", ic_dst->DebugString(shape),
//        " and ", ic_dst->DebugString(ic_dst->input(dst.index)), ".");
//    return;
//  }
  status->status = graph->graph.UpdateEdge(&new_src.oper->node, new_src.index, &dst.oper->node, dst.index);
//  if (status->status.ok()) {
//    // This modification only updates the destination node for
//    // the purposes of running this graph in a session. Thus, we don't
//    // record the source node as being modified.
//    RecordMutation(graph, *dst.oper, "updating input tensor");
//  }
}

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input) {
  mutex_lock l(graph->mu);
  graph->graph.AddControlEdge(&input->node, &op->node);
//  RecordMutation(graph, *op, "adding control input");
}

void ClearControlInputs(TF_Graph* graph, TF_Operation* op) {
  mutex_lock l(graph->mu);
  std::vector<const Edge*> control_edges;
  for (const Edge* edge : op->node.in_edges()) {
    if (!edge->IsControlEdge()) continue;
    control_edges.push_back(edge);
  }
  for (const Edge* edge : control_edges) {
    graph->graph.RemoveControlEdge(edge);
  }
//  RecordMutation(graph, *op, "clearing control inputs");
}

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device) {
  mutex_lock l(graph->mu);
  op->node.set_requested_device(device);
//  RecordMutation(graph, *op, "setting device");
}

void SetAttribute(
  TF_Graph* graph, TF_Operation* op, const char* attr_name, TF_Buffer* attr_value_proto, TF_Status* status) {
  AttrValue attr_val;
  if (!attr_val.ParseFromArray(attr_value_proto->data,
                               attr_value_proto->length)) {
    status->status = tensorflow::errors::InvalidArgument("Invalid AttrValue proto");
    return;
  }

  mutex_lock l(graph->mu);
  op->node.AddAttr(attr_name, attr_val);
  // RecordMutation(graph, *op, "setting attribute");
}

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

    const string& mutation_warning = session->graph->sessions[session];
    if (!mutation_warning.empty()) {
      // TODO(b/74949947): turn this back into an error status
      LOG(WARNING) << mutation_warning;
      session->graph->sessions[session].clear();
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

void ExtendSession(TF_Session* session, TF_Status* status) {
  ExtendSessionGraphHelper(session, status);
  session->extend_before_run = false;
}

}  // namespace tensorflow
