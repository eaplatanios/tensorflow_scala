/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

ForOp,
IfOp,
ParallelOp,
ReduceOp,
ReduceReturnOp,
YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// ForOp definitions
//===----------------------------------------------------------------------===//

ForOpOperandAdaptor::ForOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ForOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value ForOpOperandAdaptor::lowerBound() {
  return *getODSOperands(0).begin();
}

Value ForOpOperandAdaptor::upperBound() {
  return *getODSOperands(1).begin();
}

Value ForOpOperandAdaptor::step() {
  return *getODSOperands(2).begin();
}

ArrayRef<Value> ForOpOperandAdaptor::initArgs() {
  return getODSOperands(3);
}

StringRef ForOp::getOperationName() {
  return "loop.for";
}

Operation::operand_range ForOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value ForOp::lowerBound() {
  return *getODSOperands(0).begin();
}

Value ForOp::upperBound() {
  return *getODSOperands(1).begin();
}

Value ForOp::step() {
  return *getODSOperands(2).begin();
}

Operation::operand_range ForOp::initArgs() {
  return getODSOperands(3);
}

Operation::result_range ForOp::getODSResults(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic result corresponds to.
  // This assumes all static variadic results have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // result (variadic or not) as size 1. So here for each previous static variadic
  // result, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static result starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->result_begin(), offset), std::next(getOperation()->result_begin(), offset + size)};
}

Operation::result_range ForOp::results() {
  return getODSResults(0);
}

Region &ForOp::region() {
  return this->getOperation()->getRegion(0);
}



ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseForOp(parser, result);
}

void ForOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ForOp::verify() {
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
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (Value v : valueGroup3) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
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
// IfOp definitions
//===----------------------------------------------------------------------===//

IfOpOperandAdaptor::IfOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> IfOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value IfOpOperandAdaptor::condition() {
  return *getODSOperands(0).begin();
}

StringRef IfOp::getOperationName() {
  return "loop.if";
}

Operation::operand_range IfOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value IfOp::condition() {
  return *getODSOperands(0).begin();
}

Operation::result_range IfOp::getODSResults(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic result corresponds to.
  // This assumes all static variadic results have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // result (variadic or not) as size 1. So here for each previous static variadic
  // result, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static result starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->result_begin(), offset), std::next(getOperation()->result_begin(), offset + size)};
}

Operation::result_range IfOp::results() {
  return getODSResults(0);
}

Region &IfOp::thenRegion() {
  return this->getOperation()->getRegion(0);
}

Region &IfOp::elseRegion() {
  return this->getOperation()->getRegion(1);
}





ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseIfOp(parser, result);
}

void IfOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult IfOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isSignlessInteger(1)))) {
        return emitOpError("operand #") << index << " must be 1-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
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
// ParallelOp definitions
//===----------------------------------------------------------------------===//

StringRef ParallelOp::getOperationName() {
  return "loop.parallel";
}

Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned end = start + (*(sizeAttr.begin() + index)).getZExtValue();
  return {std::next(getOperation()->operand_begin(), start), std::next(getOperation()->operand_begin(), end)};
}

Operation::operand_range ParallelOp::lowerBound() {
  return getODSOperands(0);
}

Operation::operand_range ParallelOp::upperBound() {
  return getODSOperands(1);
}

Operation::operand_range ParallelOp::step() {
  return getODSOperands(2);
}

Operation::operand_range ParallelOp::initVals() {
  return getODSOperands(3);
}

Operation::result_range ParallelOp::getODSResults(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic result corresponds to.
  // This assumes all static variadic results have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // result (variadic or not) as size 1. So here for each previous static variadic
  // result, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static result starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->result_begin(), offset), std::next(getOperation()->result_begin(), offset + size)};
}

Operation::result_range ParallelOp::results() {
  return getODSResults(0);
}

Region &ParallelOp::region() {
  return this->getOperation()->getRegion(0);
}



ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseParallelOp(parser, result);
}

void ParallelOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ParallelOp::verify() {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  auto numElements = sizeAttr.getType().cast<ShapedType>().getNumElements();
  if (numElements != 4) {
    return emitOpError("'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements");
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
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (Value v : valueGroup3) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
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
// ReduceOp definitions
//===----------------------------------------------------------------------===//

ReduceOpOperandAdaptor::ReduceOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ReduceOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ReduceOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef ReduceOp::getOperationName() {
  return "loop.reduce";
}

Operation::operand_range ReduceOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ReduceOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range ReduceOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Region &ReduceOp::reductionOperator() {
  return this->getOperation()->getRegion(0);
}



ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseReduceOp(parser, result);
}

void ReduceOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ReduceOp::verify() {
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
        return emitOpError("region #") << index << " ('reductionOperator') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}


//===----------------------------------------------------------------------===//
// ReduceReturnOp definitions
//===----------------------------------------------------------------------===//

ReduceReturnOpOperandAdaptor::ReduceReturnOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ReduceReturnOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ReduceReturnOpOperandAdaptor::result() {
  return *getODSOperands(0).begin();
}

StringRef ReduceReturnOp::getOperationName() {
  return "loop.reduce.return";
}

Operation::operand_range ReduceReturnOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ReduceReturnOp::result() {
  return *getODSOperands(0).begin();
}

Operation::result_range ReduceReturnOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ReduceReturnOp::build(Builder *odsBuilder, OperationState &odsState, Value result) {
  odsState.addOperands(result);
}

void ReduceReturnOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value result) {
  odsState.addOperands(result);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceReturnOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ReduceReturnOp::verify() {
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
  return ::verify(*this);
}

ParseResult ReduceReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType resultRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> resultOperands(resultRawOperands);  llvm::SMLoc resultOperandsLoc = parser.getCurrentLocation();
  (void)resultOperandsLoc;
  Type resultRawTypes[1];
  ArrayRef<Type> resultTypes(resultRawTypes);

  if (parser.parseOperand(resultRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(resultRawTypes[0]))
    return failure();
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return failure();
  return success();
}

void ReduceReturnOp::print(OpAsmPrinter &p) {
  p << "loop.reduce.return";
  p << " ";
  p << result();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(result().getType());
}

void ReduceReturnOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpOperandAdaptor::YieldOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> YieldOpOperandAdaptor::getODSOperands(unsigned index) {
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

ArrayRef<Value> YieldOpOperandAdaptor::results() {
  return getODSOperands(0);
}

StringRef YieldOp::getOperationName() {
  return "loop.yield";
}

Operation::operand_range YieldOp::getODSOperands(unsigned index) {
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

Operation::operand_range YieldOp::results() {
  return getODSOperands(0);
}

Operation::result_range YieldOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void YieldOp::build(Builder *builder, OperationState &result) {
 /* nothing to do */ 
}

void YieldOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange results) {
  odsState.addOperands(results);
}

void YieldOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange results) {
  odsState.addOperands(results);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseYieldOp(parser, result);
}

void YieldOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult YieldOp::verify() {
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
  return ::verify(*this);
}

void YieldOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

