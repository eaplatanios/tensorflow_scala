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
                    const char* mutation_type)
    EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  // If any session has already run this node_id, mark this session as
  // unrunnable.
  for (auto it : graph->sessions) {
    if (it.first->last_num_graph_nodes > op.node.id()) {
      it.second = tensorflow::errors::FailedPrecondition(
          "Operation '", op.node.DebugString(), "' was changed by ",
          mutation_type,
          " after it was run by a session. Nodes can be mutated "
          "only before they are executed by a session. Either don't modify "
          "nodes after running them or create a new session.");
    }
  }
}

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst, TF_Status* status) {
  mutex_lock l(graph->mu);
  status->status = graph->graph.UpdateEdge(&new_src.oper->node, new_src.index, &dst.oper->node, dst.index);
  if (status->status.ok()) {
    // This modification only updates the destination node for
    // the purposes of running this graph in a session. Thus, we don't
    // record the source node as being modified.
    RecordMutation(graph, *dst.oper, "updating input tensor");
  }
}

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input) {
  mutex_lock l(graph->mu);
  graph->graph.AddControlEdge(&input->node, &op->node);
  RecordMutation(graph, *op, "adding control input");
}

void ClearControlInputs(TF_Graph* graph, TF_Operation* op) {
  mutex_lock l(graph->mu);
  for (const auto* edge : op->node.in_edges()) {
    if (edge->IsControlEdge()) {
      graph->graph.RemoveControlEdge(edge);
    }
  }
  RecordMutation(graph, *op, "clearing control inputs");
}

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device) {
  mutex_lock l(graph->mu);
  op->node.set_requested_device(device);
  RecordMutation(graph, *op, "setting device");
}

}  // namespace tensorflow
