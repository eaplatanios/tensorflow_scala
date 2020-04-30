/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct CallOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual CallInterfaceCallable getCallableForCallee(Operation *tablegen_opaque_op) = 0;
    virtual Operation::operand_range getArgOperands(Operation *tablegen_opaque_op) = 0;
    virtual Operation * resolveCallable(Operation *tablegen_opaque_op) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    CallInterfaceCallable getCallableForCallee(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableForCallee();
    }
    Operation::operand_range getArgOperands(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getArgOperands();
    }
    Operation * resolveCallable(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;

        // If the callable isn't a value, lookup the symbol reference.
        CallInterfaceCallable callable = op.getCallableForCallee();
        if (auto symbolRef = callable.dyn_cast<SymbolRefAttr>())
          return SymbolTable::lookupNearestSymbolFrom(op, symbolRef);
        return callable.get<Value>().getDefiningOp();
      
    }
  };
};
} // end namespace detail
class CallOpInterface : public OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits> {
public:
  using OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  CallInterfaceCallable getCallableForCallee();
  Operation::operand_range getArgOperands();
  Operation * resolveCallable();
};
namespace detail {
struct CallableOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Region * getCallableRegion(Operation *tablegen_opaque_op) = 0;
    virtual ArrayRef<Type> getCallableResults(Operation *tablegen_opaque_op) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    Region * getCallableRegion(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableRegion();
    }
    ArrayRef<Type> getCallableResults(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableResults();
    }
  };
};
} // end namespace detail
class CallableOpInterface : public OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits> {
public:
  using OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  Region * getCallableRegion();
  ArrayRef<Type> getCallableResults();
};
