/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

void MemoryEffectOpInterface::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffects(getOperation(), effects);
  }
void MemoryEffectOpInterface::getEffectsOnValue(Value value, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffectsOnValue(getOperation(), value, effects);
  }
void MemoryEffectOpInterface::getEffectsOnResource(SideEffects::Resource * resource, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffectsOnResource(getOperation(), resource, effects);
  }
