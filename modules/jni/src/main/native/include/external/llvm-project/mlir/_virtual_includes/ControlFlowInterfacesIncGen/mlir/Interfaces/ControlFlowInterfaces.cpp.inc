/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

Optional<OperandRange> BranchOpInterface::getSuccessorOperands(unsigned index) {
      return getImpl()->getSuccessorOperands(getOperation(), index);
  }
bool BranchOpInterface::canEraseSuccessorOperand() {
      return getImpl()->canEraseSuccessorOperand(getOperation());
  }
void BranchOpInterface::eraseSuccessorOperand(unsigned index, unsigned operandIndex) {
      return getImpl()->eraseSuccessorOperand(getOperation(), index, operandIndex);
  }
Optional<BlockArgument> BranchOpInterface::getSuccessorBlockArgument(unsigned operandIndex) {
      return getImpl()->getSuccessorBlockArgument(getOperation(), operandIndex);
  }
Block * BranchOpInterface::getSuccessorForOperands(ArrayRef<Attribute> operands) {
      return getImpl()->getSuccessorForOperands(getOperation(), operands);
  }
OperandRange RegionBranchOpInterface::getSuccessorEntryOperands(unsigned index) {
      return getImpl()->getSuccessorEntryOperands(getOperation(), index);
  }
void RegionBranchOpInterface::getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getOperation(), index, operands, regions);
  }
