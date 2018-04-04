//#include "tensorflow/c/c_eager_api.h"
//
//#include <algorithm>
//#include <cstddef>
//#include <memory>
//#include <string>
//#include <vector>
//
//#include "tensorflow/c/c_api.h"
//#include "tensorflow/c/c_api_internal.h"
//#include "tensorflow/c/c_eager_api_internal.h"
//#include "tensorflow/c/c_eager_api_runtime.h"
//#ifdef TENSORFLOW_EAGER_USE_XLA
//#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
//#endif  // TENSORFLOW_EAGER_USE_XLA
//#include "tensorflow/core/common_runtime/copy_tensor.h"
//#include "tensorflow/core/common_runtime/device_factory.h"
//#include "tensorflow/core/common_runtime/device_mgr.h"
//#include "tensorflow/core/common_runtime/device_set.h"
//#include "tensorflow/core/common_runtime/function.h"
//#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
//#include "tensorflow/core/common_runtime/eager/eager_executor.h"
//#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
//#include "tensorflow/core/framework/node_def_util.h"
//#include "tensorflow/core/framework/rendezvous.h"
//#include "tensorflow/core/framework/tensor_shape.pb.h"
//#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/lib/core/refcount.h"
//#include "tensorflow/core/lib/gtl/flatmap.h"
//#include "tensorflow/core/lib/gtl/map_util.h"
//#include "tensorflow/core/lib/gtl/stl_util.h"
//#include "tensorflow/core/platform/env.h"
//#include "tensorflow/core/platform/mutex.h"
//#include "tensorflow/core/platform/thread_annotations.h"
//#include "tensorflow/core/public/version.h"
//
//namespace tensorflow {
//tensorflow::Status EagerExecutor::WaitFor(tensorflow::uint64 node_id) {
//  return WaitImpl(false, node_id);
//}
//
//tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
//  return WaitImpl(true, 0);
//}
//
//tensorflow::Status EagerExecutor::WaitImpl(bool wait_all,
//                                           tensorflow::uint64 node_id) {
//  tensorflow::condition_variable cond;
//  tensorflow::mutex_lock l(node_queue_mutex_);
//  // Don't wait if an error is already set.
//  if (!status_.ok()) return status_;
//  if (node_queue_.empty()) return tensorflow::Status::OK();
//  if (wait_all) {
//    node_id = node_queue_.back()->id;
//  } else if (node_id < node_queue_.front()->id) {
//    // Note that we are relying on the ops being dispatched sequentially from
//    // the queue.
//    return tensorflow::Status::OK();
//  }
//  node_done_notifications_.insert(std::make_pair(node_id, &cond));
//  cond.wait(l);
//  // Note that we could be woken up if an error occurs, even though the node has
//  // not actually executed.
//  return status_;
//}
//
//}  // namespace tensorflow
//
//bool TensorHandle::IsReady() {
//  if (node_id == 0) return true;
//  tensorflow::mutex_lock l(ctx_mutex_);
//  return ctx_ == nullptr;
//}
//
//tensorflow::Status TensorHandle::WaitReady() {
//  if (node_id == 0) return tensorflow::Status::OK();
//  tensorflow::EagerExecutor* executor = nullptr;
//  {
//    tensorflow::mutex_lock l(ctx_mutex_);
//    if (ctx_ == nullptr) return tensorflow::Status::OK();
//    executor = ctx_->context.Executor();
//  }
//  return executor->WaitFor(node_id);
//}
//
//tensorflow::Status TensorHandle::Tensor(const tensorflow::Tensor** t) {
//  TF_RETURN_IF_ERROR(WaitReady());
//  DCHECK(IsReady());
//  *t = &tensor_;
//  return tensorflow::Status::OK();
//}
//
//tensorflow::Status TensorHandle::Device(tensorflow::Device** d) {
//  TF_RETURN_IF_ERROR(WaitReady());
//  DCHECK(IsReady());
//  *d = device_;
//  return tensorflow::Status::OK();
//}
//
//tensorflow::Status TensorHandle::OpDevice(tensorflow::Device** d) {
//  TF_RETURN_IF_ERROR(WaitReady());
//  DCHECK(IsReady());
//  *d = op_device_;
//  return tensorflow::Status::OK();
//}
//
//tensorflow::Status TensorHandle::TensorAndDevice(
//    const tensorflow::Tensor** tensor, tensorflow::Device** device,
//    tensorflow::Device** op_device) {
//  TF_RETURN_IF_ERROR(WaitReady());
//  DCHECK(IsReady());
//  *tensor = &tensor_;
//  *device = device_;
//  *op_device = op_device_;
//  return tensorflow::Status::OK();
//}
//
//void TensorHandle::SetTensorAndDevice(const tensorflow::Tensor& tensor,
//                                          tensorflow::Device* device,
//                                          tensorflow::Device* op_device) {
//  tensorflow::mutex_lock l(ctx_mutex_);
//  DCHECK(node_id > 0 && ctx_) << "SetTensorAndDevice should be only called  "
//                              << "on non-ready handles.";
//  ctx_ = nullptr;
//  tensor_ = tensor;
//  device_ = device;
//  op_device_ = op_device;
//}
