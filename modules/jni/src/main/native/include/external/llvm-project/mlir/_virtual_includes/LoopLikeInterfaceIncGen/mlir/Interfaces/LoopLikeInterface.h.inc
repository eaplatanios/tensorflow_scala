/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct LoopLikeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual bool isDefinedOutsideOfLoop(Operation *tablegen_opaque_op, Value  value) = 0;
    virtual Region & getLoopBody(Operation *tablegen_opaque_op) = 0;
    virtual LogicalResult moveOutOfLoop(Operation *tablegen_opaque_op, ArrayRef<Operation *> ops) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    bool isDefinedOutsideOfLoop(Operation *tablegen_opaque_op, Value  value) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.isDefinedOutsideOfLoop(value);
    }
    Region & getLoopBody(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getLoopBody();
    }
    LogicalResult moveOutOfLoop(Operation *tablegen_opaque_op, ArrayRef<Operation *> ops) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.moveOutOfLoop(ops);
    }
  };
};
} // end namespace detail
class LoopLikeOpInterface : public OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits> {
public:
  using OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  bool isDefinedOutsideOfLoop(Value  value);
  Region & getLoopBody();
  LogicalResult moveOutOfLoop(ArrayRef<Operation *> ops);
};
