/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

AffineApplyOp,
AffineForOp,
AffineIfOp,
AffineMaxOp,
AffineMinOp,
AffineParallelOp,
AffinePrefetchOp,
AffineTerminatorOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// AffineApplyOp definitions
//===----------------------------------------------------------------------===//

AffineApplyOpOperandAdaptor::AffineApplyOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineApplyOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AffineApplyOpOperandAdaptor::mapOperands() {
  return getODSOperands(0);
}

StringRef AffineApplyOp::getOperationName() {
  return "affine.apply";
}

Operation::operand_range AffineApplyOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AffineApplyOp::mapOperands() {
  return getODSOperands(0);
}

Operation::result_range AffineApplyOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

AffineMapAttr AffineApplyOp::mapAttr() {
  return this->getAttr("map").cast<AffineMapAttr>();
}

AffineMap AffineApplyOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineApplyOp::mapAttr(AffineMapAttr attr) {
  this->getOperation()->setAttr("map", attr);
}

void AffineApplyOp::build(Builder *builder, OperationState &result, AffineMap map, ValueRange mapOperands) {
      build(builder, result, builder->getIndexType(), map, mapOperands);
    
}

void AffineApplyOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMapAttr map, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("map", map);
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMapAttr map, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("map", map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMap map, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMap map, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AffineApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineApplyOp(parser, result);
}

void AffineApplyOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AffineApplyOp::verify() {
  auto tblgen_map = this->getAttr("map");
  if (!tblgen_map) return emitOpError("requires attribute 'map'");
  {
    if (!((tblgen_map.isa<AffineMapAttr>()))) return emitOpError("attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





void AffineApplyOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AffineForOp definitions
//===----------------------------------------------------------------------===//

AffineForOpOperandAdaptor::AffineForOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineForOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

StringRef AffineForOp::getOperationName() {
  return "affine.for";
}

Operation::operand_range AffineForOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::result_range AffineForOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Region &AffineForOp::region() {
  return this->getOperation()->getRegion(0);
}





ParseResult AffineForOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineForOp(parser, result);
}

void AffineForOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AffineForOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((region.getBlocks().size() == 1))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}












//===----------------------------------------------------------------------===//
// AffineIfOp definitions
//===----------------------------------------------------------------------===//

AffineIfOpOperandAdaptor::AffineIfOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineIfOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

StringRef AffineIfOp::getOperationName() {
  return "affine.if";
}

Operation::operand_range AffineIfOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::result_range AffineIfOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Region &AffineIfOp::thenRegion() {
  return this->getOperation()->getRegion(0);
}

Region &AffineIfOp::elseRegion() {
  return this->getOperation()->getRegion(1);
}



ParseResult AffineIfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineIfOp(parser, result);
}

void AffineIfOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AffineIfOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((region.getBlocks().size() == 1))) {
        return emitOpError("region #") << index << " ('thenRegion') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(1))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('elseRegion') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}






//===----------------------------------------------------------------------===//
// AffineMaxOp definitions
//===----------------------------------------------------------------------===//

AffineMaxOpOperandAdaptor::AffineMaxOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineMaxOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AffineMaxOpOperandAdaptor::operands() {
  return getODSOperands(0);
}

StringRef AffineMaxOp::getOperationName() {
  return "affine.max";
}

Operation::operand_range AffineMaxOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AffineMaxOp::operands() {
  return getODSOperands(0);
}

Operation::result_range AffineMaxOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

AffineMapAttr AffineMaxOp::mapAttr() {
  return this->getAttr("map").cast<AffineMapAttr>();
}

AffineMap AffineMaxOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineMaxOp::mapAttr(AffineMapAttr attr) {
  this->getOperation()->setAttr("map", attr);
}

void AffineMaxOp::build(Builder *builder, OperationState &result, AffineMap affineMap, ValueRange mapOperands) {
      build(builder, result, builder->getIndexType(), affineMap, mapOperands);
    
}

void AffineMaxOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMapAttr map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", map);
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMapAttr map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMap map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMap map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AffineMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineMinMaxOp<AffineMaxOp>(parser, result);
}

void AffineMaxOp::print(OpAsmPrinter &p) {
  return ::printAffineMinMaxOp(p, *this);
}

LogicalResult AffineMaxOp::verify() {
  auto tblgen_map = this->getAttr("map");
  if (!tblgen_map) return emitOpError("requires attribute 'map'");
  {
    if (!((tblgen_map.isa<AffineMapAttr>()))) return emitOpError("attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyAffineMinMaxOp(*this);
}





void AffineMaxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AffineMinOp definitions
//===----------------------------------------------------------------------===//

AffineMinOpOperandAdaptor::AffineMinOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineMinOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AffineMinOpOperandAdaptor::operands() {
  return getODSOperands(0);
}

StringRef AffineMinOp::getOperationName() {
  return "affine.min";
}

Operation::operand_range AffineMinOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AffineMinOp::operands() {
  return getODSOperands(0);
}

Operation::result_range AffineMinOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

AffineMapAttr AffineMinOp::mapAttr() {
  return this->getAttr("map").cast<AffineMapAttr>();
}

AffineMap AffineMinOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineMinOp::mapAttr(AffineMapAttr attr) {
  this->getOperation()->setAttr("map", attr);
}

void AffineMinOp::build(Builder *builder, OperationState &result, AffineMap affineMap, ValueRange mapOperands) {
      build(builder, result, builder->getIndexType(), affineMap, mapOperands);
    
}

void AffineMinOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMapAttr map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", map);
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMapAttr map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, AffineMap map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMap map, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("map", AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AffineMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineMinMaxOp<AffineMinOp>(parser, result);
}

void AffineMinOp::print(OpAsmPrinter &p) {
  return ::printAffineMinMaxOp(p, *this);
}

LogicalResult AffineMinOp::verify() {
  auto tblgen_map = this->getAttr("map");
  if (!tblgen_map) return emitOpError("requires attribute 'map'");
  {
    if (!((tblgen_map.isa<AffineMapAttr>()))) return emitOpError("attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyAffineMinMaxOp(*this);
}





void AffineMinOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AffineParallelOp definitions
//===----------------------------------------------------------------------===//

AffineParallelOpOperandAdaptor::AffineParallelOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineParallelOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AffineParallelOpOperandAdaptor::mapOperands() {
  return getODSOperands(0);
}

StringRef AffineParallelOp::getOperationName() {
  return "affine.parallel";
}

Operation::operand_range AffineParallelOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AffineParallelOp::mapOperands() {
  return getODSOperands(0);
}

Operation::result_range AffineParallelOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Region &AffineParallelOp::region() {
  return this->getOperation()->getRegion(0);
}

AffineMapAttr AffineParallelOp::lowerBoundsMapAttr() {
  return this->getAttr("lowerBoundsMap").cast<AffineMapAttr>();
}

AffineMap AffineParallelOp::lowerBoundsMap() {
  auto attr = lowerBoundsMapAttr();
  return attr.getValue();
}

AffineMapAttr AffineParallelOp::upperBoundsMapAttr() {
  return this->getAttr("upperBoundsMap").cast<AffineMapAttr>();
}

AffineMap AffineParallelOp::upperBoundsMap() {
  auto attr = upperBoundsMapAttr();
  return attr.getValue();
}

ArrayAttr AffineParallelOp::stepsAttr() {
  return this->getAttr("steps").cast<ArrayAttr>();
}

ArrayAttr AffineParallelOp::steps() {
  auto attr = stepsAttr();
  return attr;
}

void AffineParallelOp::lowerBoundsMapAttr(AffineMapAttr attr) {
  this->getOperation()->setAttr("lowerBoundsMap", attr);
}

void AffineParallelOp::upperBoundsMapAttr(AffineMapAttr attr) {
  this->getOperation()->setAttr("upperBoundsMap", attr);
}

void AffineParallelOp::stepsAttr(ArrayAttr attr) {
  this->getOperation()->setAttr("steps", attr);
}







void AffineParallelOp::build(Builder *odsBuilder, OperationState &odsState, AffineMapAttr lowerBoundsMap, AffineMapAttr upperBoundsMap, ArrayAttr steps, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("lowerBoundsMap", lowerBoundsMap);
  odsState.addAttribute("upperBoundsMap", upperBoundsMap);
  odsState.addAttribute("steps", steps);
  (void)odsState.addRegion();
}

void AffineParallelOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMapAttr lowerBoundsMap, AffineMapAttr upperBoundsMap, ArrayAttr steps, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("lowerBoundsMap", lowerBoundsMap);
  odsState.addAttribute("upperBoundsMap", upperBoundsMap);
  odsState.addAttribute("steps", steps);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineParallelOp::build(Builder *odsBuilder, OperationState &odsState, AffineMap lowerBoundsMap, AffineMap upperBoundsMap, ArrayAttr steps, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("lowerBoundsMap", AffineMapAttr::get(lowerBoundsMap));
  odsState.addAttribute("upperBoundsMap", AffineMapAttr::get(upperBoundsMap));
  odsState.addAttribute("steps", steps);
  (void)odsState.addRegion();
}

void AffineParallelOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, AffineMap lowerBoundsMap, AffineMap upperBoundsMap, ArrayAttr steps, ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute("lowerBoundsMap", AffineMapAttr::get(lowerBoundsMap));
  odsState.addAttribute("upperBoundsMap", AffineMapAttr::get(upperBoundsMap));
  odsState.addAttribute("steps", steps);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineParallelOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AffineParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffineParallelOp(parser, result);
}

void AffineParallelOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AffineParallelOp::verify() {
  auto tblgen_lowerBoundsMap = this->getAttr("lowerBoundsMap");
  if (!tblgen_lowerBoundsMap) return emitOpError("requires attribute 'lowerBoundsMap'");
  {
    if (!((tblgen_lowerBoundsMap.isa<AffineMapAttr>()))) return emitOpError("attribute 'lowerBoundsMap' failed to satisfy constraint: AffineMap attribute");
  }
  auto tblgen_upperBoundsMap = this->getAttr("upperBoundsMap");
  if (!tblgen_upperBoundsMap) return emitOpError("requires attribute 'upperBoundsMap'");
  {
    if (!((tblgen_upperBoundsMap.isa<AffineMapAttr>()))) return emitOpError("attribute 'upperBoundsMap' failed to satisfy constraint: AffineMap attribute");
  }
  auto tblgen_steps = this->getAttr("steps");
  if (!tblgen_steps) return emitOpError("requires attribute 'steps'");
  {
    if (!(((tblgen_steps.isa<ArrayAttr>())) && (llvm::all_of(tblgen_steps.cast<ArrayAttr>(), [](Attribute attr) { return ((attr.isa<IntegerAttr>())) && ((attr.cast<IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitOpError("attribute 'steps' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((region.getBlocks().size() == 1))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}


//===----------------------------------------------------------------------===//
// AffinePrefetchOp definitions
//===----------------------------------------------------------------------===//

AffinePrefetchOpOperandAdaptor::AffinePrefetchOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffinePrefetchOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value AffinePrefetchOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> AffinePrefetchOpOperandAdaptor::indices() {
  return getODSOperands(1);
}

StringRef AffinePrefetchOp::getOperationName() {
  return "affine.prefetch";
}

Operation::operand_range AffinePrefetchOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value AffinePrefetchOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::operand_range AffinePrefetchOp::indices() {
  return getODSOperands(1);
}

Operation::result_range AffinePrefetchOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

BoolAttr AffinePrefetchOp::isWriteAttr() {
  return this->getAttr("isWrite").cast<BoolAttr>();
}

bool AffinePrefetchOp::isWrite() {
  auto attr = isWriteAttr();
  return attr.getValue();
}

IntegerAttr AffinePrefetchOp::localityHintAttr() {
  return this->getAttr("localityHint").cast<IntegerAttr>();
}

APInt AffinePrefetchOp::localityHint() {
  auto attr = localityHintAttr();
  return attr.getValue();
}

BoolAttr AffinePrefetchOp::isDataCacheAttr() {
  return this->getAttr("isDataCache").cast<BoolAttr>();
}

bool AffinePrefetchOp::isDataCache() {
  auto attr = isDataCacheAttr();
  return attr.getValue();
}

void AffinePrefetchOp::isWriteAttr(BoolAttr attr) {
  this->getOperation()->setAttr("isWrite", attr);
}

void AffinePrefetchOp::localityHintAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("localityHint", attr);
}

void AffinePrefetchOp::isDataCacheAttr(BoolAttr attr) {
  this->getOperation()->setAttr("isDataCache", attr);
}

void AffinePrefetchOp::build(Builder *builder, OperationState &result, Value memref,AffineMap map, ArrayRef<Value> mapOperands, bool isWrite,unsigned localityHint, bool isDataCache) {
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = builder->getI32IntegerAttr(localityHint);
      auto isWriteAttr = builder->getBoolAttr(isWrite);
      auto isDataCacheAttr = builder->getBoolAttr(isDataCache);
      result.addOperands(memref);
      result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
      result.addOperands(mapOperands);
      result.addAttribute(getLocalityHintAttrName(), localityHintAttr);
      result.addAttribute(getIsWriteAttrName(), isWriteAttr);
      result.addAttribute(getIsDataCacheAttrName(), isDataCacheAttr);
    
}

void AffinePrefetchOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, ValueRange indices, BoolAttr isWrite, IntegerAttr localityHint, BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
}

void AffinePrefetchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, ValueRange indices, BoolAttr isWrite, IntegerAttr localityHint, BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, ValueRange indices, bool isWrite, APInt localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", (*odsBuilder).getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", (*odsBuilder).getBoolAttr(isDataCache));
}

void AffinePrefetchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, ValueRange indices, bool isWrite, APInt localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", (*odsBuilder).getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", (*odsBuilder).getBoolAttr(isDataCache));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AffinePrefetchOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAffinePrefetchOp(parser, result);
}

void AffinePrefetchOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AffinePrefetchOp::verify() {
  auto tblgen_isWrite = this->getAttr("isWrite");
  if (!tblgen_isWrite) return emitOpError("requires attribute 'isWrite'");
  {
    if (!((tblgen_isWrite.isa<BoolAttr>()))) return emitOpError("attribute 'isWrite' failed to satisfy constraint: bool attribute");
  }
  auto tblgen_localityHint = this->getAttr("localityHint");
  if (!tblgen_localityHint) return emitOpError("requires attribute 'localityHint'");
  {
    if (!((((tblgen_localityHint.isa<IntegerAttr>())) && ((tblgen_localityHint.cast<IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<IntegerAttr>().getInt() <= 3)))) return emitOpError("attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");
  }
  auto tblgen_isDataCache = this->getAttr("isDataCache");
  if (!tblgen_isDataCache) return emitOpError("requires attribute 'isDataCache'");
  {
    if (!((tblgen_isDataCache.isa<BoolAttr>()))) return emitOpError("attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}






//===----------------------------------------------------------------------===//
// AffineTerminatorOp definitions
//===----------------------------------------------------------------------===//

AffineTerminatorOpOperandAdaptor::AffineTerminatorOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AffineTerminatorOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

StringRef AffineTerminatorOp::getOperationName() {
  return "affine.terminator";
}

Operation::operand_range AffineTerminatorOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range AffineTerminatorOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AffineTerminatorOp::build(Builder *odsBuilder, OperationState &odsState) {

}

void AffineTerminatorOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineTerminatorOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult AffineTerminatorOp::verify() {
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return mlir::success();
}

void AffineTerminatorOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

