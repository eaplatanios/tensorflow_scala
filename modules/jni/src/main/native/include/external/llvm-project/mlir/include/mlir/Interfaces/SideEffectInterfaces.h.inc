/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct MemoryEffectOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void getEffects(Operation *tablegen_opaque_op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) = 0;
    virtual void getEffectsOnValue(Operation *tablegen_opaque_op, Value value, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) = 0;
    virtual void getEffectsOnResource(Operation *tablegen_opaque_op, SideEffects::Resource * resource, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    void getEffects(Operation *tablegen_opaque_op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getEffects(effects);
    }
    void getEffectsOnValue(Operation *tablegen_opaque_op, Value value, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;

          op.getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getValue() != value;
          });
       
    }
    void getEffectsOnResource(Operation *tablegen_opaque_op, SideEffects::Resource * resource, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;

          op.getEffects(effects);
          llvm::erase_if(effects, [&](auto &it) {
            return it.getResource() != resource;
          });
       
    }
  };
};
} // end namespace detail
class MemoryEffectOpInterface : public OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits> {
public:
  using OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<MemoryEffectOpInterface, detail::MemoryEffectOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects);
  void getEffectsOnValue(Value value, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects);
  void getEffectsOnResource(SideEffects::Resource * resource, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects);

    /// Collect all of the effect instances that correspond to the given
    /// `Effect` and place them in 'effects'.
    template <typename Effect> void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<
                                              MemoryEffects::Effect>> &effects) {
      getEffects(effects);
      llvm::erase_if(effects, [&](auto &it) {
        return !llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns true if this operation exhibits the given effect.
    template <typename Effect> bool hasEffect() {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return llvm::any_of(effects, [](const auto &it) {
        return llvm::isa<Effect>(it.getEffect());
      });
    }

    /// Returns if this operation only has the given effect.
    template <typename Effect> bool onlyHasEffect() {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return !effects.empty() && llvm::all_of(effects, [](const auto &it) {
        return isa<Effect>(it.getEffect());
      });
    }

    /// Returns if this operation has no effects.
    bool hasNoEffect() {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
      getEffects(effects);
      return effects.empty();
    }

    /// Returns if the given operation has no effects for this interface.
    static bool hasNoEffect(Operation *op) {
      if (auto interface = dyn_cast<MemoryEffectOpInterface>(op))
        return interface.hasNoEffect();
      return op->hasTrait<OpTrait::HasRecursiveSideEffects>();
    }
  
};
