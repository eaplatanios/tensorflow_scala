/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPH_CONTROL_FLOW_H_
#define TENSORFLOW_GRAPH_CONTROL_FLOW_H_

#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Control flow info for a graph node.
struct ControlFlowInfo {
  const Node* frame = nullptr;         // frame of a node
  const Node* parent_frame = nullptr;  // parent frame of a node
  string frame_name;                   // frame name of a node
};

// Assign to each node the name of the frame and the level it belongs to.
// We check the well-formedness of the graph: All inputs to a node must
// come from the same frame and have the same "static" iteration level.
// `info` is cleared and populated by this function.
// NOTE(yuanbyu): For now, we require all sends/recvs have iteration level
// 0. This essentially means there can't be multiple serial Nexts in
// an iteration, which all sane front-ends should satisfy.
Status BuildControlFlowInfo(Graph* g, std::vector<ControlFlowInfo>* info);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_CONTROL_FLOW_H_
