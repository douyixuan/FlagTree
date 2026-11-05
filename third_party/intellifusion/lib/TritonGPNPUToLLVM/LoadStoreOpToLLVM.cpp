#include "AtomicPTXBuilder.h"
#include "PTXAsmFormat.h"
#include "PatternPopulation.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace Intellifusion {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static constexpr unsigned kMaxGlobalVectorBits = 128;

static unsigned getElemBitWidth(Type elemTy) {
  if (elemTy.isIntOrFloat())
    return elemTy.getIntOrFloatBitWidth();
  return 0;
}

static unsigned getIntegerAttrFromValue(Value value, StringRef name,
                                        unsigned fallback) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (!owner)
      return fallback;
    auto funcOp = dyn_cast_or_null<FunctionOpInterface>(owner->getParentOp());
    if (!funcOp)
      return fallback;
    Attribute attr = funcOp.getArgAttr(blockArg.getArgNumber(), name);
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
      return intAttr.getInt();
    if (auto denseAttr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
      if (denseAttr.isSplat() && isa<IntegerType>(denseAttr.getElementType()))
        return denseAttr.getSplatValue<APInt>().getZExtValue();
      if (isa<IntegerType>(denseAttr.getElementType()))
        return (*denseAttr.getValues<APInt>().begin()).getZExtValue();
    }
    return fallback;
  }

  if (auto opResult = dyn_cast<OpResult>(value)) {
    Attribute attr = opResult.getOwner()->getAttr(name);
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
      return intAttr.getInt();
    if (auto denseAttr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
      if (denseAttr.isSplat() && isa<IntegerType>(denseAttr.getElementType()))
        return denseAttr.getSplatValue<APInt>().getZExtValue();
      if (isa<IntegerType>(denseAttr.getElementType()))
        return (*denseAttr.getValues<APInt>().begin()).getZExtValue();
    }
  }
  return fallback;
}

static unsigned getPointerDivisibilityBytes(Value value) {
  unsigned divisibility = getIntegerAttrFromValue(value, "tt.divisibility", 0);
  if (divisibility)
    return divisibility;
  if (auto splat = value.getDefiningOp<SplatOp>())
    return getPointerDivisibilityBytes(splat.getSrc());
  if (auto addPtr = value.getDefiningOp<AddPtrOp>())
    return getPointerDivisibilityBytes(addPtr.getPtr());
  return 0;
}

// Generate a zero/default value for the given LLVM type.
static Value getDefaultValue(OpBuilder &builder, Location loc, Type elemTy) {
  if (isa<FloatType>(elemTy))
    return LLVM::ConstantOp::create(builder, loc, elemTy,
                                    builder.getFloatAttr(elemTy, 0.0));
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return LLVM::ConstantOp::create(builder, loc, elemTy,
                                    builder.getIntegerAttr(elemTy, 0));
  llvm_unreachable("unsupported element type for default value");
}

// Keep this implementation local to IFWA so shared lowering remains unchanged.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    return std::min<unsigned>(kMaxGlobalVectorBits / pointeeBitWidth,
                              contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

static Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  SmallVector<std::pair<StringAttr, Value>> dimensions = {
      {kLane, laneId}, {kWarp, warpId}, {kBlock, blockId}};
  for (auto [dimName, dimId] : dimensions) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      Value isOwner = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, isOwner);
    }
  }
  return pred;
}

static void createAtomicResultBarrier(ConversionPatternRewriter &rewriter,
                                      Location loc, int numCTAs,
                                      Operation *sourceOp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (numCTAs == 1) {
    b.barrier(gpu::AddrSpace::Local);
  } else {
    auto barrier = ttng::ClusterBarrierOp::create(rewriter, loc);
    ttng::copyClusterBarrierMbarOffset(sourceOp, barrier);
  }
}

static Value loadScalarAtomicResult(ConversionPatternRewriter &rewriter,
                                    Location loc, const TargetInfo &targetInfo,
                                    Value sharedPtr, Type valueTy,
                                    int numCTAs) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (numCTAs == 1)
    return b.load(valueTy, sharedPtr);
  return targetInfo.loadDShared(rewriter, loc, sharedPtr, b.i32_val(0), valueTy,
                                b.true_val());
}

static unsigned chooseVectorSize(unsigned candidate, unsigned numElems,
                                 Type elemTy) {
  unsigned elemBits = getElemBitWidth(elemTy);
  if (candidate <= 1 || elemBits == 0 || elemBits > kMaxGlobalVectorBits)
    return 1;
  candidate = std::min(candidate, numElems);
  while (candidate > 1 && (numElems % candidate != 0 ||
                           elemBits * candidate > kMaxGlobalVectorBits)) {
    candidate >>= 1;
  }
  return std::max(candidate, 1u);
}

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 unsigned index) {
  Type i32Ty = builder.getI32Type();
  return LLVM::ConstantOp::create(builder, loc, i32Ty,
                                  builder.getI32IntegerAttr(index));
}

static unsigned getLoadStoreAlignment(Value ptr, unsigned vectorBytes) {
  unsigned divisibility = getPointerDivisibilityBytes(ptr);
  if (divisibility >= vectorBytes)
    return divisibility;
  return vectorBytes;
}

static unsigned getGPNPUVectorSize(const LoadStoreConversionBase &base,
                                   Value ptr) {
  if (!ptr.getParentRegion())
    return 1;
  return base.getVectorSize(ptr);
}

static unsigned getGPNPUMaskAlignment(const LoadStoreConversionBase &base,
                                      Value mask) {
  if (!mask.getParentRegion())
    return 1;
  return base.getMaskAlignment(mask);
}

static bool isGlobalLLVMPtr(Value ptr) {
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  return ptrTy && ptrTy.getAddressSpace() == 1;
}

static bool supportsPtxWideGlobal(Type elemTy, unsigned vec) {
  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  if (elemBits == 32)
    return vec == 2 || vec == 4;
  return elemBits == 16 && vec == 8;
}

static bool supportsPredicatedPtxGlobal(Type elemTy, unsigned vec) {
  if (!elemTy.isIntOrFloat() || elemTy.getIntOrFloatBitWidth() < 8)
    return false;
  const size_t elemBits = elemTy.getIntOrFloatBitWidth();
  const size_t totalWidth = elemBits * vec;
  const size_t width = std::min(totalWidth, std::max<size_t>(32, elemBits));
  if (totalWidth % width != 0 ||
      (width != 8 && width != 16 && width != 32 && width != 64))
    return false;
  const size_t nWords = totalWidth / width;
  return nWords == 1 || nWords == 2 || nWords == 4;
}

// IFWA accepts predicated `ld.global` inline PTX when the address and list
// operands use the compact spelling emitted below.
static SmallVector<Value>
emitPredicatedGlobalLoad(ConversionPatternRewriter &rewriter, Location loc,
                         Type elemTy, Value ptr, ArrayRef<Value> fallbackElems,
                         Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const size_t elemBits = std::max(8u, elemTy.getIntOrFloatBitWidth());
  const size_t vec = fallbackElems.size();
  const size_t maxWordWidth = std::max<size_t>(32, elemBits);
  const size_t totalWidth = elemBits * vec;
  const size_t width = std::min(totalWidth, maxWordWidth);
  const size_t nWords = std::max<size_t>(1, totalWidth / width);
  const size_t wordElems = width / elemBits;
  const size_t movWidth = std::max<size_t>(16, width);

  GPNPUPTXBuilder ptxBuilder;
  const std::string readConstraint =
      (width == 64) ? "l" : ((width == 32) ? "r" : "c");
  const std::string writeConstraint =
      (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

  auto *dstsOpr = ptxBuilder.newListOperand();
  for (size_t word = 0; word < nWords; ++word)
    dstsOpr->listAppend(ptxBuilder.newOperand(writeConstraint, /*init=*/false));
  dstsOpr->repr = [dstsOpr, nWords](int) {
    std::string ret = "{";
    for (size_t word = 0; word < nWords; ++word) {
      if (word)
        ret += ", ";
      ret += "$" + std::to_string(dstsOpr->listGet(word)->idx);
    }
    ret += "}";
    return ret;
  };

  // A predicated-off load must return tl.load's `other` value. Initialize
  // every output word before issuing the predicated memory instruction.
  for (size_t word = 0; word < nWords; ++word) {
    auto wordTy = LLVM::getVectorType(elemTy, wordElems);
    Value packed = b.undef(wordTy);
    for (size_t elem = 0; elem < wordElems; ++elem) {
      Value index = createIndexConstant(rewriter, loc, elem);
      packed = b.insert_element(wordTy, packed,
                                fallbackElems[word * wordElems + elem], index);
    }
    packed = b.bitcast(packed, IntegerType::get(rewriter.getContext(), width));
    // For an i8 word, the `c` constraint allocates a physical 16-bit register
    // for mov.u16 while the inline-asm value remains the logical i8 payload.
    // ld.global.b8 reads or replaces only its low 8 bits.
    auto &mov = ptxBuilder.create("mov")->o("u" + std::to_string(movWidth));
    mov(dstsOpr->listGet(word), ptxBuilder.newOperand(packed, readConstraint));
  }

  auto *addrOpr = ptxBuilder.newAddrOperand(ptr, "l", 0);
  addrOpr->repr = [](int idx) { return "[$" + std::to_string(idx) + "]"; };
  auto &ld = ptxBuilder.create("ld")->global().v(nWords).b(width);
  ld(dstsOpr, addrOpr).maybePredicate(pred, "b");

  SmallVector<Type> retTys(nWords,
                           IntegerType::get(rewriter.getContext(), width));
  Type retTy =
      nWords > 1
          ? LLVM::LLVMStructType::getLiteral(rewriter.getContext(), retTys)
          : retTys.front();
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);

  SmallVector<Value> words;
  for (size_t word = 0; word < nWords; ++word) {
    Value value = nWords > 1 ? b.extract_val(retTys[word], ret, word) : ret;
    words.push_back(b.bitcast(value, LLVM::getVectorType(elemTy, wordElems)));
  }

  SmallVector<Value> loadedElems;
  for (size_t elem = 0; elem < vec; ++elem) {
    Value index = createIndexConstant(rewriter, loc, elem % wordElems);
    loadedElems.push_back(
        b.extract_element(elemTy, words[elem / wordElems], index));
  }
  return loadedElems;
}

static void emitPredicatedGlobalStore(ConversionPatternRewriter &rewriter,
                                      Location loc, Type elemTy, Value ptr,
                                      ArrayRef<Value> storeElems, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const size_t elemBits = std::max(8u, elemTy.getIntOrFloatBitWidth());
  const size_t vec = storeElems.size();
  const size_t maxWordWidth = std::max<size_t>(32, elemBits);
  const size_t totalWidth = elemBits * vec;
  const size_t width = std::min(totalWidth, maxWordWidth);
  const size_t nWords = std::max<size_t>(1, totalWidth / width);
  const size_t wordElems = width / elemBits;

  Type wordIntTy = IntegerType::get(rewriter.getContext(), width);
  auto wordVecTy = vec_ty(elemTy, wordElems);
  SmallVector<std::pair<Value, std::string>> asmArgs;
  for (size_t word = 0; word < nWords; ++word) {
    Value packed = b.undef(wordVecTy);
    for (size_t elem = 0; elem < wordElems; ++elem) {
      Value value = storeElems[word * wordElems + elem];
      if (value.getType().isInteger(1))
        value = b.sext(i8_ty, value);
      value = b.bitcast(value, elemTy);
      packed = b.insert_element(wordVecTy, packed, value, b.i32_val(elem));
    }
    packed = b.bitcast(packed, wordIntTy);
    std::string constraint = (width == 64) ? "l" : ((width == 32) ? "r" : "c");
    asmArgs.emplace_back(packed, constraint);
  }

  GPNPUPTXBuilder ptxBuilder;
  auto *valuesOpr = ptxBuilder.newListOperand(asmArgs);
  valuesOpr->repr = [valuesOpr, nWords](int) {
    std::string ret = "{";
    for (size_t word = 0; word < nWords; ++word) {
      if (word)
        ret += ", ";
      ret += "$" + std::to_string(valuesOpr->listGet(word)->idx);
    }
    ret += "}";
    return ret;
  };
  auto *addrOpr = ptxBuilder.newAddrOperand(ptr, "l", 0);
  addrOpr->repr = [](int idx) { return "[$" + std::to_string(idx) + "]"; };
  auto &st = ptxBuilder.create("st")->global().v(nWords).b(width);
  st(addrOpr, valuesOpr).maybePredicate(pred, "b");
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

// Lower tt.load to LLVM loads or global inline PTX.
// Handles three cases:
//   1. Basic load (no mask): load all elements unconditionally.
//   2. Masked load (mask, no other): predicated with zero padding.
//   3. Masked load with other: predicated with user-supplied padding.
//
// Global masked loads predicate the memory instruction.
// Non-global masked loads use conditional branching rather than select so a
// false mask cannot execute an out-of-bounds memory access.
struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   ModuleAxisInfoAnalysis &axisInfoAnalysis,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // IFWA diff: cache policy and thread redundancy are not routed through
    // TargetInfo yet; global loads use compact PTX operand text accepted by
    // IFWA.
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resultTy = op.getType();
    Type valueLLVMElemTy =
        typeConverter->convertType(getElementTypeOrSelf(resultTy));
    unsigned numElems = gpu::getTotalElemsPerThread(op.getPtr().getType());

    unsigned vec = getGPNPUVectorSize(*this, op.getPtr());
    if (op.getMask())
      vec = std::min(vec, getGPNPUMaskAlignment(*this, op.getMask()));
    vec = chooseVectorSize(vec, numElems, valueLLVMElemTy);
    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);

    SmallVector<Value> maskElems;
    if (adaptor.getMask())
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);

    SmallVector<Value> otherElems;
    if (adaptor.getOther())
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);

    SmallVector<Value> resultVals;
    for (unsigned i = 0; i < numElems; i += vec) {
      if (maskElems.empty()) {
        if (vec == 1) {
          // No mask: unconditional scalar load.
          Value loaded =
              LLVM::LoadOp::create(rewriter, loc, valueLLVMElemTy, ptrElems[i]);
          resultVals.push_back(loaded);
          continue;
        }

        if (isGlobalLLVMPtr(ptrElems[i]) &&
            supportsPtxWideGlobal(valueLLVMElemTy, vec)) {
          // IFWA accepts the PTX instruction when its operands use compact
          // text: `{$0, ...}, [$N]`.
          const size_t valueElemNBits =
              std::max(8u, valueLLVMElemTy.getIntOrFloatBitWidth());
          const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
          const size_t totalWidth = valueElemNBits * vec;
          const size_t width = std::min(totalWidth, maxWordWidth);
          const size_t nWords = std::max<size_t>(1, totalWidth / width);
          const size_t wordNElems = width / valueElemNBits;

          GPNPUPTXBuilder ptxBuilder;
          const std::string writeConstraint =
              (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

          auto *dstsOpr = ptxBuilder.newListOperand();
          for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx)
            dstsOpr->listAppend(ptxBuilder.newOperand(writeConstraint,
                                                      /*init=*/false));
          dstsOpr->repr = [dstsOpr, nWords](int idx) {
            std::string ret = "{";
            for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
              if (wordIdx)
                ret += ", ";
              ret += "$" + std::to_string(dstsOpr->listGet(wordIdx)->idx);
            }
            ret += "}";
            return ret;
          };

          auto *addrOpr = ptxBuilder.newAddrOperand(ptrElems[i], "l", 0);
          addrOpr->repr = [](int idx) {
            return "[$" + std::to_string(idx) + "]";
          };

          auto &ld = ptxBuilder.create("ld")->global().v(nWords).b(width);
          ld(dstsOpr, addrOpr);

          SmallVector<Type> retTys(
              nWords, IntegerType::get(rewriter.getContext(), width));
          Type retTy =
              LLVM::LLVMStructType::getLiteral(rewriter.getContext(), retTys);
          Value ret = ptxBuilder.launch(rewriter, loc, retTy);

          SmallVector<Value> rets;
          for (unsigned wordIdx = 0; wordIdx < nWords; ++wordIdx) {
            Value curr = b.extract_val(
                IntegerType::get(rewriter.getContext(), width), ret, wordIdx);
            curr = b.bitcast(curr,
                             LLVM::getVectorType(valueLLVMElemTy, wordNElems));
            rets.push_back(curr);
          }

          for (unsigned lane = 0; lane < vec; ++lane) {
            Value vecIdx =
                createIndexConstant(rewriter, loc, lane % wordNElems);
            resultVals.push_back(b.extract_element(
                valueLLVMElemTy, rets[lane / wordNElems], vecIdx));
          }
          continue;
        }

        auto vecTy = LLVM::getVectorType(valueLLVMElemTy, vec);
        Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, ptrElems[i]);
        for (unsigned lane = 0; lane < vec; ++lane) {
          resultVals.push_back(
              b.extract_element(valueLLVMElemTy, loaded,
                                createIndexConstant(rewriter, loc, lane)));
        }
      } else {
        SmallVector<Value> fallbackElems;
        for (unsigned lane = 0; lane < vec; ++lane) {
          fallbackElems.push_back(
              otherElems.empty()
                  ? getDefaultValue(rewriter, loc, valueLLVMElemTy)
                  : otherElems[i + lane]);
        }

        // Predicate the global memory instruction itself. This both prevents
        // OOB accesses and avoids creating a pair
        // of CFG blocks for every register element.
        if (isGlobalLLVMPtr(ptrElems[i]) &&
            supportsPredicatedPtxGlobal(valueLLVMElemTy, vec)) {
          auto loadedElems = emitPredicatedGlobalLoad(
              rewriter, loc, valueLLVMElemTy, ptrElems[i], fallbackElems,
              maskElems[i]);
          llvm::append_range(resultVals, loadedElems);
          continue;
        }

        // Non-global masked loads retain an explicit branch because an
        // unconditional load followed by select would still access OOB.

        if (vec == 1) {
          Value fallback = fallbackElems[0];

          // Split current block: everything after this point goes to
          // mergeBlock.
          Block *curBlock = rewriter.getInsertionBlock();
          Block *mergeBlock =
              curBlock->splitBlock(rewriter.getInsertionPoint());
          // mergeBlock gets a block argument as phi node for the result.
          mergeBlock->addArgument(valueLLVMElemTy, loc);

          // Create loadBlock between curBlock and mergeBlock.
          Block *loadBlock = rewriter.createBlock(
              curBlock->getParent(), std::next(Region::iterator(curBlock)));

          // curBlock: cond_br mask -> loadBlock, mergeBlock(fallback)
          rewriter.setInsertionPointToEnd(curBlock);
          LLVM::CondBrOp::create(rewriter, loc, maskElems[i], loadBlock,
                                 ValueRange{}, mergeBlock,
                                 ValueRange{fallback});

          // loadBlock: load then br -> mergeBlock(loaded)
          rewriter.setInsertionPointToEnd(loadBlock);
          Value loaded =
              LLVM::LoadOp::create(rewriter, loc, valueLLVMElemTy, ptrElems[i]);
          LLVM::BrOp::create(rewriter, loc, ValueRange{loaded}, mergeBlock);

          // mergeBlock: use block argument as phi result
          rewriter.setInsertionPointToStart(mergeBlock);
          resultVals.push_back(mergeBlock->getArgument(0));
          continue;
        }

        SmallVector<Value> loadedElems;
        auto vecTy = LLVM::getVectorType(valueLLVMElemTy, vec);

        // Split current block: everything after this point goes to mergeBlock.
        Block *curBlock = rewriter.getInsertionBlock();
        Block *mergeBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        for (unsigned lane = 0; lane < vec; ++lane)
          mergeBlock->addArgument(valueLLVMElemTy, loc);

        // Create loadBlock between curBlock and mergeBlock.
        Block *loadBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // curBlock: cond_br mask -> loadBlock, mergeBlock(fallback)
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[i], loadBlock,
                               ValueRange{}, mergeBlock,
                               ValueRange{fallbackElems});

        // loadBlock: vector load then br -> mergeBlock(loaded vector lanes)
        rewriter.setInsertionPointToEnd(loadBlock);
        Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, ptrElems[i]);
        for (unsigned lane = 0; lane < vec; ++lane) {
          loadedElems.push_back(
              b.extract_element(valueLLVMElemTy, loaded,
                                createIndexConstant(rewriter, loc, lane)));
        }
        LLVM::BrOp::create(rewriter, loc, ValueRange{loadedElems}, mergeBlock);

        // mergeBlock: use block arguments as phi results
        rewriter.setInsertionPointToStart(mergeBlock);
        for (unsigned lane = 0; lane < vec; ++lane)
          resultVals.push_back(mergeBlock->getArgument(lane));
      }
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, resultStruct);
    return success();
  }
};

// Lower tt.store to LLVM stores or global inline PTX. Global
// masked stores predicate the instruction; non-global stores branch around it.
struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter, const TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(axisInfoAnalysis), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // IFWA diff: retain LLVM stores where no predicate is needed; global
    // predicated stores use compact PTX operand text accepted by IFWA.
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    if (!isa<RankedTensorType>(ptr.getType()) &&
        (isa<RankedTensorType>(value.getType()) ||
         (mask && isa<RankedTensorType>(mask.getType())))) {
      return op.emitOpError(
          "scalar store requires scalar pointer, value and mask");
    }

    unsigned numElems = gpu::getTotalElemsPerThread(op.getPtr().getType());

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(value.getType()));
    unsigned vec = getGPNPUVectorSize(*this, op.getPtr());
    if (op.getMask())
      vec = std::min(vec, getGPNPUMaskAlignment(*this, op.getMask()));
    vec = chooseVectorSize(vec, numElems, valueElemTy);
    unsigned elemBytes = std::max(getElemBitWidth(valueElemTy) / 8, 1u);
    unsigned alignment = getLoadStoreAlignment(op.getPtr(), vec * elemBytes);
    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valueElems = unpackLLElements(loc, adaptor.getValue(), rewriter);

    SmallVector<Value> maskElems;
    if (adaptor.getMask())
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    // Shared pointers address independent storage in each CTA. Generic
    // pointers need a runtime check to distinguish shared from global.
    unsigned addressSpace =
        cast<LLVM::LLVMPointerType>(ptrElems.front().getType())
            .getAddressSpace();
    int32_t &blockMask = freeVarMasks[str_attr("block")];
    Value ctaPred;
    if (op.getIgnoreCta() ||
        addressSpace == targetInfo.getSharedAddressSpace()) {
      blockMask = 0;
    } else if (addressSpace == 0 && blockMask != 0 &&
               gpu::lookupNumCTAs(op) > 1) {
      llvm::MapVector<StringAttr, int32_t> ctaMasks;
      ctaMasks[str_attr("block")] = blockMask;
      ctaPred = ttg::emitRedundantThreadPredicate(ctaMasks, rewriter, loc,
                                                  targetInfo);
      blockMask = 0;
    }
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, targetInfo);
    // Linear layouts name the register dimension "register". Scalars have
    // only one element, so they do not need a register mask.
    uint32_t regMask = freeVarMasks.lookup(str_attr("register"));

    for (unsigned i = 0; i < numElems; i += vec) {
      if (!isCanonicalIndex(i, regMask))
        continue;
      Value pred = threadPred;
      if (ctaPred) {
        // Different vector groups can address different memory spaces.
        Value isShared = LLVM::createLLVMIntrinsicCallOp(
                             rewriter, loc, "llvm.nvvm.isspacep.shared",
                             rewriter.getI1Type(), ptrElems[i])
                             .getResult(0);
        pred = ttg::maybeAnd(rewriter, loc, pred, b.or_(isShared, ctaPred));
      }
      if (!maskElems.empty())
        pred = ttg::maybeAnd(rewriter, loc, pred, maskElems[i]);
      if (!pred) {
        if (vec == 1) {
          // No user mask or redundant threads: unconditional scalar store.
          LLVM::StoreOp::create(rewriter, loc, valueElems[i], ptrElems[i]);
          continue;
        }
        auto storeElems = ArrayRef<Value>(valueElems).slice(i, vec);
        if (isGlobalLLVMPtr(ptrElems[i]) &&
            supportsPtxWideGlobal(valueElemTy, vec)) {
          // IFWA accepts the PTX instruction when its operands use compact
          // text: `[$N], {$0, ...}`.
          const size_t valueElemNBits =
              std::max(8u, valueElemTy.getIntOrFloatBitWidth());
          const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
          const size_t totalWidth = valueElemNBits * vec;
          const size_t width = std::min(totalWidth, maxWordWidth);
          const size_t nWords = std::max<size_t>(1, totalWidth / width);
          const size_t wordNElems = width / valueElemNBits;

          Type valArgTy = IntegerType::get(rewriter.getContext(), width);
          auto wordTy = vec_ty(valueElemTy, wordNElems);

          SmallVector<std::pair<Value, std::string>> asmArgs;
          for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
            Value llWord = b.undef(wordTy);
            for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
              size_t elemOffset = wordIdx * wordNElems + elemIdx;
              assert(elemOffset < storeElems.size());
              Value elem = storeElems[elemOffset];
              if (elem.getType().isInteger(1))
                elem = b.sext(i8_ty, elem);
              elem = b.bitcast(elem, valueElemTy);
              llWord =
                  b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
            }
            llWord = b.bitcast(llWord, valArgTy);
            std::string constraint =
                (width == 64) ? "l" : ((width == 32) ? "r" : "c");
            asmArgs.emplace_back(llWord, constraint);
          }

          GPNPUPTXBuilder ptxBuilder;
          auto *asmArgList = ptxBuilder.newListOperand(asmArgs);
          asmArgList->repr = [asmArgList, nWords](int idx) {
            std::string ret = "{";
            for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
              if (wordIdx)
                ret += ", ";
              ret += "$" + std::to_string(asmArgList->listGet(wordIdx)->idx);
            }
            ret += "}";
            return ret;
          };
          auto *asmAddr = ptxBuilder.newAddrOperand(ptrElems[i], "l", 0);
          asmAddr->repr = [](int idx) {
            return "[$" + std::to_string(idx) + "]";
          };

          auto &ptxStoreInstr =
              ptxBuilder.create("st")->global().v(nWords).b(width);
          ptxStoreInstr(asmAddr, asmArgList);
          ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
          continue;
        }
        auto vecTy = LLVM::getVectorType(valueElemTy, vec);
        Value packed = b.undef(vecTy);
        for (unsigned lane = 0; lane < vec; ++lane) {
          packed = b.insert_element(vecTy, packed, storeElems[lane],
                                    createIndexConstant(rewriter, loc, lane));
        }
        LLVM::StoreOp::create(rewriter, loc, packed, ptrElems[i], alignment);
      } else {
        SmallVector<Value> storeElems(
            ArrayRef<Value>(valueElems).slice(i, vec));

        // Keep the CFG compact by predicating the global store itself.
        if (isGlobalLLVMPtr(ptrElems[i]) &&
            supportsPredicatedPtxGlobal(valueElemTy, vec)) {
          emitPredicatedGlobalStore(rewriter, loc, valueElemTy, ptrElems[i],
                                    storeElems, pred);
          continue;
        }

        if (vec > 1) {
          // Masked store: use conditional branching to skip OOB writes.
          Block *curBlock = rewriter.getInsertionBlock();
          Block *mergeBlock =
              curBlock->splitBlock(rewriter.getInsertionPoint());

          // Create storeBlock between curBlock and mergeBlock.
          Block *storeBlock = rewriter.createBlock(
              curBlock->getParent(), std::next(Region::iterator(curBlock)));

          // curBlock: cond_br mask -> storeBlock, mergeBlock
          rewriter.setInsertionPointToEnd(curBlock);
          LLVM::CondBrOp::create(rewriter, loc, pred, storeBlock, ValueRange{},
                                 mergeBlock, ValueRange{});

          // storeBlock: vector store then br -> mergeBlock
          rewriter.setInsertionPointToEnd(storeBlock);
          auto vecTy = LLVM::getVectorType(valueElemTy, vec);
          Value packed = b.undef(vecTy);
          for (unsigned lane = 0; lane < vec; ++lane) {
            packed = b.insert_element(vecTy, packed, storeElems[lane],
                                      createIndexConstant(rewriter, loc, lane));
          }
          LLVM::StoreOp::create(rewriter, loc, packed, ptrElems[i], alignment);
          LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

          // Continue at mergeBlock for next element.
          rewriter.setInsertionPointToStart(mergeBlock);
          continue;
        }
        Value storeValue = storeElems[0];

        // Masked store: use conditional branch to skip OOB writes.
        // Split current block.
        Block *curBlock = rewriter.getInsertionBlock();
        Block *mergeBlock = curBlock->splitBlock(rewriter.getInsertionPoint());

        // Create storeBlock between curBlock and mergeBlock.
        Block *storeBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // curBlock: cond_br mask → storeBlock, mergeBlock
        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, pred, storeBlock, ValueRange{},
                               mergeBlock, ValueRange{});

        // storeBlock: store then br → mergeBlock
        rewriter.setInsertionPointToEnd(storeBlock);
        LLVM::StoreOp::create(rewriter, loc, storeValue, ptrElems[i],
                              vec == 1 ? 0 : alignment);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        // Continue at mergeBlock for next element.
        rewriter.setInsertionPointToStart(mergeBlock);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfo &targetInfo;
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(axisInfoAnalysis), targetInfo(targetInfo) {}

  bool supportsVectorized(RMWOp op, Type elemTy) const {
    return targetInfo.supportVectorizedAtomics() && op == RMWOp::FADD &&
           elemTy.isF32();
  }

  bool isPromotableToLoad(triton::AtomicRMWOp op) const {
    if (op.getSem() != MemSemantic::ACQUIRE &&
        op.getSem() != MemSemantic::RELAXED)
      return false;
    if (op.getAtomicRmwOp() != RMWOp::ADD && op.getAtomicRmwOp() != RMWOp::FADD)
      return false;
    if (isa<RankedTensorType>(op.getType()))
      return false;
    auto constant = op.getVal().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return false;
    if (auto value = dyn_cast<FloatAttr>(constant.getValueAttr()))
      return value.getValue().isZero();
    if (auto value = dyn_cast<IntegerAttr>(constant.getValueAttr()))
      return value.getValue().isZero();
    return false;
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    int numCTAs = gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    if (!matchAtomicOp(op.getAtomicRmwOp()))
      return rewriter.notifyMatchFailure(op, "unsupported GPNPU atomic RMW op");

    Type resultTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(resultTy);
    Type valueElemTy = getTypeConverter()->convertType(
        tensorTy ? tensorTy.getElementType() : resultTy);
    if (op.getAtomicRmwOp() == RMWOp::FADD) {
      if (!valueElemTy.isF16() && !valueElemTy.isBF16() &&
          !valueElemTy.isF32() && !valueElemTy.isF64())
        return op.emitOpError(
            "GPNPU atomic fadd requires f16, bf16, f32, or f64");
    } else if (op.getAtomicRmwOp() == RMWOp::XCHG) {
      if (!valueElemTy.isInteger(32) && !valueElemTy.isInteger(64) &&
          !valueElemTy.isF32() && !valueElemTy.isF64())
        return op.emitOpError(
            "GPNPU atomic exchange requires i32, i64, f32, or f64");
    } else if (!valueElemTy.isInteger(32) && !valueElemTy.isInteger(64)) {
      return op.emitOpError(
          "GPNPU integer atomic RMW requires a 32-bit or 64-bit integer "
          "element type");
    }

    auto ptrElements = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valElements = unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> maskElements;
    if (adaptor.getMask())
      maskElements = unpackLLElements(loc, adaptor.getMask(), rewriter);

    unsigned elemsPerThread =
        gpu::getTotalElemsPerThread(op.getVal().getType());
    unsigned vec = 1;
    unsigned packed = 1;
    if (tensorTy) {
      vec = getGPNPUVectorSize(*this, op.getPtr());
      if (op.getMask())
        vec = std::min(vec, getGPNPUMaskAlignment(*this, op.getMask()));
      vec = chooseVectorSize(vec, elemsPerThread, valueElemTy);
      if (!supportsVectorized(op.getAtomicRmwOp(), tensorTy.getElementType())) {
        packed = std::min<unsigned>(vec, valueElemTy.isF16() ? 2 : 1);
        vec = 1;
      }
    }

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    SmallVector<Value> resultVals(elemsPerThread);
    for (unsigned i = 0; i < elemsPerThread; i += vec * packed) {
      unsigned canonicalIndex = i & ~regMask;
      if (canonicalIndex != i) {
        for (unsigned j = 0; j < vec * packed; ++j)
          resultVals[i + j] = resultVals[canonicalIndex + j];
        continue;
      }

      Value pred = threadPred;
      if (!maskElements.empty())
        pred = b.and_(pred, maskElements[i]);

      if (isPromotableToLoad(op)) {
        triton::nvgpu::MemSyncScope loadScope;
        switch (op.getScope()) {
        case MemSyncScope::CTA:
          loadScope = triton::nvgpu::MemSyncScope::CTA;
          break;
        case MemSyncScope::GPU:
          loadScope = triton::nvgpu::MemSyncScope::GPU;
          break;
        case MemSyncScope::SYSTEM:
          loadScope = triton::nvgpu::MemSyncScope::SYSTEM;
          break;
        }
        auto loadSem = op.getSem() == MemSemantic::ACQUIRE
                           ? triton::nvgpu::MemSemantic::ACQUIRE
                           : triton::nvgpu::MemSemantic::RELAXED;
        Value old = triton::nvgpu::LoadAcquireOp::create(
            rewriter, loc, valueElemTy, ptrElements[i], pred, loadSem,
            loadScope);
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value sharedPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                    op.getOperation());
        targetInfo.storeDShared(rewriter, loc, sharedPtr, Value(), old, pred);
        createAtomicResultBarrier(rewriter, loc, numCTAs, op);
        Value result = loadScalarAtomicResult(rewriter, loc, targetInfo,
                                              sharedPtr, valueElemTy, numCTAs);
        rewriter.replaceOp(op, result);
        return success();
      }

      SmallVector<Value> rmwVals;
      for (unsigned j = 0; j < (vec > 1 ? vec : packed); ++j)
        rmwVals.push_back(valElements[i + j]);
      auto old = ptx::emitPtxAtomicRMW(
          rewriter, loc, valueElemTy, ptrElements[i], rmwVals,
          op.getAtomicRmwOp(), op.getSem(), op.getScope(), pred, vec, packed);
      if (failed(old))
        return failure();

      if (tensorTy) {
        if (vec > 1) {
          for (unsigned j = 0; j < vec; ++j)
            resultVals[i + j] = b.extract_val(valueElemTy, *old, j);
        } else if (packed > 1) {
          for (unsigned j = 0; j < packed; ++j)
            resultVals[i + j] =
                b.extract_element(valueElemTy, *old, b.i32_val(j));
        } else {
          resultVals[i] = *old;
        }
      } else {
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value sharedPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                    op.getOperation());
        targetInfo.storeDShared(rewriter, loc, sharedPtr, Value(), *old, pred);
        createAtomicResultBarrier(rewriter, loc, numCTAs, op);
        Value result = loadScalarAtomicResult(rewriter, loc, targetInfo,
                                              sharedPtr, valueElemTy, numCTAs);
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    if (tensorTy) {
      finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals,
                                  valueElemTy, b, threadPred, targetInfo,
                                  getTypeConverter());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "scalar atomic was not lowered");
  }

private:
  const TargetInfo &targetInfo;
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    int numCTAs = gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Type resultTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(resultTy);
    Type valueElemTy = getTypeConverter()->convertType(
        tensorTy ? tensorTy.getElementType() : resultTy);
    if (!valueElemTy.isIntOrFloat() ||
        (valueElemTy.getIntOrFloatBitWidth() != 16 &&
         valueElemTy.getIntOrFloatBitWidth() != 32 &&
         valueElemTy.getIntOrFloatBitWidth() != 64))
      return rewriter.notifyMatchFailure(
          op, "GPNPU atomic CAS requires a 16-, 32-, or 64-bit value");

    auto ptrElements = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto cmpElements = unpackLLElements(loc, adaptor.getCmp(), rewriter);
    auto valElements = unpackLLElements(loc, adaptor.getVal(), rewriter);

    unsigned elemsPerThread =
        gpu::getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    SmallVector<Value> resultVals(elemsPerThread);
    Value scalarAtomicPred;
    for (unsigned i = 0; i < elemsPerThread; ++i) {
      unsigned canonicalIndex = i & ~regMask;
      if (canonicalIndex != i) {
        resultVals[i] = resultVals[canonicalIndex];
        continue;
      }

      if (!tensorTy)
        scalarAtomicPred = threadPred;
      resultVals[i] = ptx::emitPtxAtomicCAS(
          rewriter, loc, valueElemTy, ptrElements[i], cmpElements[i],
          valElements[i], op.getSem(), op.getScope(), threadPred);
    }

    if (tensorTy) {
      finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals,
                                  valueElemTy, b, threadPred, targetInfo,
                                  getTypeConverter());
      return success();
    }

    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (!op->hasAttr("allocation.offset"))
      return rewriter.notifyMatchFailure(
          op, "scalar atomic result requires shared-memory allocation");

    Value sharedPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    sharedPtr = b.bitcast(sharedPtr, ptr_ty(ctx, 3));
    targetInfo.storeDShared(rewriter, loc, sharedPtr, Value(),
                            resultVals.front(), scalarAtomicPred);
    createAtomicResultBarrier(rewriter, loc, numCTAs, op);
    Value old = loadScalarAtomicResult(rewriter, loc, targetInfo, sharedPtr,
                                       valueElemTy, numCTAs);
    rewriter.replaceOp(op, old);
    return success();
  }

private:
  const TargetInfo &targetInfo;
};

struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<triton::gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (op.hasGlobalRead() || op.hasGlobalWrite()) {
      NVVM::MembarOp::create(rewriter, loc, NVVM::MemScopeKind::GPU);
    }
    NVVM::BarrierOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }
};

// IFWA diff: only device-side descriptor load is registered here; normal
// tt.load/tt.store continue to use IFWA's branch-based memory lowering above.
static Value createGPNPUElectPredicate(Location loc, OpBuilder &rewriter) {
  return NVVM::ElectSyncOp::create(rewriter, loc, i1_ty,
                                   /*membermask=*/Value());
}

static Value createGPNPUTMAMulticastMask(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         uint16_t broadcastBits,
                                         const TargetInfo &targetInfo) {
  int numCTAs = ttg::lookupNumCTAs(rewriter);
  int blockBits = llvm::Log2_32(numCTAs);
  uint32_t fixedBits = (~broadcastBits) & (numCTAs - 1);
  uint32_t pattern = 1;
  for (int i = 0; i < blockBits; ++i) {
    if ((fixedBits & (1u << i)) == 0)
      pattern |= (pattern << (1u << i));
  }
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value ctaId = targetInfo.getClusterCTAId(rewriter, loc);
  Value base = b.and_(ctaId, b.i32_val(fixedBits));
  return b.shl(b.i32_val(pattern), base);
}

static LinearLayout getMsgToPackedOffsetLayout(ttg::MemDescType ty,
                                               ttg::TMAMode mode) {
  auto ctx = ty.getContext();
  auto kMsg = str_attr("msg");
  auto kBlock = str_attr("block");
  auto shapePerCTA = ttg::getShapePerCTA(ty);
  int rank = shapePerCTA.size();
  auto blockShape = ttng::getTMABlockShape(ty, /*packedSize=*/true, mode);
  auto outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout msgToOffset;
  for (int dim = 0; dim < rank; ++dim) {
    msgToOffset *=
        LinearLayout::strided1D(shapePerCTA[dim] / blockShape[dim],
                                blockShape[dim], kMsg, outDimNames[dim]);
  }
  msgToOffset *= ttg::getCGALayout(ty.getEncoding()).getLinearLayout();
  return msgToOffset;
}

static LinearLayout
getMsgToUnpackedOffsetLayout(const LinearLayout &packedLayout,
                             ttg::MemDescType ty) {
  auto sharedEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(ty.getEncoding());
  if (!sharedEnc || !sharedEnc.getFp4Padded())
    return packedLayout;

  auto ctx = ty.getContext();
  auto rank = ty.getRank();
  auto kMsg = str_attr("msg");
  auto kLastDim = str_attr("dim" + Twine(rank - 1));
  return LinearLayout::zeros1D(1, kMsg, kLastDim, 2) * packedLayout;
}

// For asynchronous TMA gather and scatter, route cluster CTA id through
// Intellifusion TargetInfo and use the local elect predicate helper. Preserve
// the instruction scopes, predicates, coordinates, and completion mechanisms
// used by the shared operation model.
static LinearLayout getUnswizzledLayout(ttg::MemDescType type) {
  auto mmaEncoding = dyn_cast<ttg::NVMMASharedEncodingAttr>(type.getEncoding());
  if (!mmaEncoding) {
    assert(isa<ttg::SwizzledSharedEncodingAttr>(type.getEncoding()));
    return ttg::toLinearLayout(type);
  }
  assert(type.getShape() == type.getAllocShape().take_back(type.getRank()));
  return ttg::nvmmaSharedToLinearLayout(type.getShape(), mmaEncoding,
                                        ttg::TMAMode::Tiled,
                                        /*disableSwizzle=*/true);
}

static LogicalResult iterateGatherScatterIndices(
    Operation *op, ConversionPatternRewriter &rewriter,
    const TypeConverter &typeConverter, TypedValue<RankedTensorType> xCoords,
    TypedValue<ttg::MemDescType> smem, Value smemObjValue, Value xOffsetsValue,
    Value yOffsetValue, Value pred, const TargetInfo &targetInfo,
    llvm::function_ref<void(Value, Value, Value, ArrayRef<Value>)> callback) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  StringAttr kDim1 = str_attr("dim1");
  StringAttr kMsg = str_attr("msg");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  LinearLayout xCoordsLayout = ttg::toLinearLayout(xCoords.getType());
  if (xCoordsLayout.getInDimSize(kRegister) < 4)
    return op->emitError("must have at least 4 x offsets per warp");
  for (unsigned i : {0, 1}) {
    if (xCoordsLayout.getBasis(kRegister, i).front() != (1 << i))
      return op->emitError(
          "x offsets are not grouped by 4 contiguous elements");
  }

  ttg::MemDescType smemType = smem.getType();
  ArrayRef<int64_t> allocShape = smemType.getAllocShape();
  if (allocShape.size() < 2 || smemType.getShape() != allocShape.take_back(2))
    return op->emitError("memdesc shape must match alloc shape");
  auto enc = dyn_cast<ttg::NVMMASharedEncodingAttr>(smemType.getEncoding());
  if (!enc)
    return op->emitError("requires dst encoding NVMMASharedEncodingAttr");
  if (enc.getFp4Padded())
    yOffsetValue = b.mul(yOffsetValue, b.i32_val(2));
  Type llvmElemTy = typeConverter.convertType(smemType.getElementType());
  Type elemPtrTy = ptr_ty(ctx, /*addrspace=*/3);
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, smemObjValue,
                                                       llvmElemTy, rewriter);
  Value smemBase = smemObj.getShmemAffineBase(loc, rewriter, smemType);

  unsigned threadsPerWarp = xCoordsLayout.getInDimSize(kLane);
  auto shapePerCTA = ttg::getShapePerCTA(smemType);
  auto tmaBlockShape = ttng::getTMABlockShape(smemType, /*packedSize=*/true,
                                              ttg::TMAMode::Tiled);
  unsigned innerBlockSize = shapePerCTA.back();
  unsigned contigDimSize = tmaBlockShape.back();
  unsigned numMessagesPerRow = ceil<unsigned>(innerBlockSize, contigDimSize);

  assert(innerBlockSize % numMessagesPerRow == 0);
  assert(llvm::isPowerOf2_32(numMessagesPerRow));
  unsigned msgSize = innerBlockSize / numMessagesPerRow;
  LinearLayout msgToCol =
      LinearLayout::strided1D(numMessagesPerRow, msgSize, kMsg, kDim1);
  LinearLayout msgLayout = xCoordsLayout * msgToCol;
  LinearLayout sharedLayout = getUnswizzledLayout(smemType);
  LinearLayout msgToShared = msgLayout.invertAndCompose(sharedLayout);

  auto freeVars = xCoordsLayout.getFreeVariableMasks();
  unsigned regMask = freeVars[kRegister];
  unsigned warpMask = freeVars[kWarp];
  if (freeVars[kLane] != (threadsPerWarp - 1))
    return op->emitError("x offsets must be broadcasted across each warp");

  Value warpId = ttg::WarpIdOp::create(rewriter, loc);
  Value blockId = targetInfo.getClusterCTAId(rewriter, loc);
  pred = b.and_(pred,
                b.icmp_eq(b.i32_val(0), b.and_(warpId, b.i32_val(warpMask))));
  pred = b.and_(pred, createGPNPUElectPredicate(loc, rewriter));

  SmallVector<Value> xOffsets = unpackLLElements(loc, xOffsetsValue, rewriter);
  Value laneId = b.i32_val(0);
  for (auto regId : seq<unsigned>(0, xOffsets.size(), 4)) {
    if ((regMask & regId) != 0)
      continue;
    Value regIdVal = b.i32_val(regId);

    for (auto msgId : llvm::seq(numMessagesPerRow)) {
      Value msgIdVal = b.i32_val(msgId);
      auto result = applyLinearLayout(loc, rewriter, msgToShared,
                                      {{kRegister, regIdVal},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, blockId},
                                       {kMsg, msgIdVal}});
      assert(result.size() == 2 && result.front().first == "offset" &&
             result.back().first == "block");
      Value shMemOffset = result.front().second;
      Value shMemPtr = b.gep(elemPtrTy, llvmElemTy, smemBase, shMemOffset);
      Value yOffset = b.add(yOffsetValue, b.i32_val(msgId * msgSize));

      callback(pred, shMemPtr, yOffset, ArrayRef(xOffsets).slice(regId, 4));
    }
  }

  return success();
}

struct AsyncTMAGatherOpConversion
    : public ConvertOpToLLVMPattern<ttng::AsyncTMAGatherOp> {
  const TargetInfo &targetInfo;

  AsyncTMAGatherOpConversion(LLVMTypeConverter &converter,
                             const TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::AsyncTMAGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());

    auto barrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        getTypeConverter()->convertType(
            op.getBarrier().getType().getElementType()),
        rewriter);

    auto callback = [&](Value pred, Value shMemPtr, Value yOffset,
                        ArrayRef<Value> xOffsets) {
      std::string tmaInst = "@$0 cp.async.bulk.tensor.2d.tile::gather4.shared"
                            "::cta.global.mbarrier::complete_tx::bytes "
                            "[$1], [$2, {$3, $4, $5, $6, $7}], [$8];";

      GPNPUPTXBuilder ptxBuilder;
      SmallVector<GPNPUPTXBuilder::Operand *, 9> operands{
          ptxBuilder.newOperand(pred, "b"),
          ptxBuilder.newOperand(shMemPtr, "r"),
          ptxBuilder.newOperand(adaptor.getDesc(), "l"),
          ptxBuilder.newOperand(yOffset, "r")};
      for (Value xOffset : xOffsets)
        operands.push_back(ptxBuilder.newOperand(xOffset, "r"));
      operands.push_back(ptxBuilder.newOperand(barrierMemObj.getBase(), "r"));

      auto &tma = *ptxBuilder.create(tmaInst);
      tma(operands, /*attachOnlyMLIRArgs=*/true);
      ptxBuilder.launch(rewriter, loc, voidTy);
    };

    if (failed(iterateGatherScatterIndices(
            op, rewriter, *getTypeConverter(), op.getXOffsets(), op.getResult(),
            adaptor.getResult(), adaptor.getXOffsets(), adaptor.getYOffset(),
            adaptor.getPred(), targetInfo, callback)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTMAScatterOpConversion
    : public ConvertOpToLLVMPattern<ttng::AsyncTMAScatterOp> {
  const TargetInfo &targetInfo;

  AsyncTMAScatterOpConversion(LLVMTypeConverter &converter,
                              const TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::AsyncTMAScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());

    auto callback = [&](Value pred, Value shMemPtr, Value yOffset,
                        ArrayRef<Value> xOffsets) {
      std::string tmaInst = "@$0 cp.async.bulk.tensor.2d.tile::scatter4.global"
                            ".shared::cta.bulk_group "
                            "[$1, {$2, $3, $4, $5, $6}], [$7];";

      GPNPUPTXBuilder ptxBuilder;
      SmallVector<GPNPUPTXBuilder::Operand *, 8> operands{
          ptxBuilder.newOperand(pred, "b"),
          ptxBuilder.newOperand(adaptor.getDesc(), "l"),
          ptxBuilder.newOperand(yOffset, "r")};
      for (Value xOffset : xOffsets)
        operands.push_back(ptxBuilder.newOperand(xOffset, "r"));
      operands.push_back(ptxBuilder.newOperand(shMemPtr, "r"));

      auto &tma = *ptxBuilder.create(tmaInst);
      tma(operands, /*attachOnlyMLIRArgs=*/true);
      ptxBuilder.launch(rewriter, loc, voidTy);
    };

    if (failed(iterateGatherScatterIndices(
            op, rewriter, *getTypeConverter(), op.getXOffsets(), op.getSrc(),
            adaptor.getSrc(), adaptor.getXOffsets(), adaptor.getYOffset(),
            /*pred=*/b.true_val(), targetInfo, callback)))
      return failure();

    NVVM::CpAsyncBulkCommitGroupOp::create(rewriter, loc);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTMACopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<ttng::AsyncTMACopyGlobalToLocalOp> {
  const TargetInfo &targetInfo;

  AsyncTMACopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                        const TargetInfo &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::AsyncTMACopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCache() != triton::CacheModifier::NONE)
      return op.emitError("cache modifiers not supported yet");
    if (op.getEvict() != triton::EvictionPolicy::NORMAL)
      return op.emitError("eviction policy not supported yet");
    if (op.getIsVolatile())
      return op.emitError("volatile not supported yet");

    auto descType = op.getDesc().getType();
    bool isIm2Col = isa<ttng::TensorDescIm2ColType>(descType);
    ttg::TMAMode tmaMode =
        isIm2Col ? ttg::TMAMode::Im2Col : ttg::TMAMode::Tiled;

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    ttg::MemDescType dstTy = op.getResult().getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto barrierTy = op.getBarrier().getType();
    auto barrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        typeConverter->convertType(barrierTy.getElementType()), rewriter);
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    Value dstBase = dstMemObj.getShmemAffineBase(loc, rewriter, dstTy);

    auto voidTy = void_ty(op->getContext());
    auto id = getThreadId(rewriter, loc);

    auto mod = op->getParentOfType<ModuleOp>();
    int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    Value pred = adaptor.getPred();
    pred = b.and_(pred, createGPNPUElectPredicate(loc, rewriter));

    auto smemTy = op.getResult().getType();
    int rank = op.getCoord().size();

    auto msgToPackedOffset = getMsgToPackedOffsetLayout(smemTy, tmaMode);
    auto smemLayout = ttg::toLinearLayout(smemTy);
    auto msgToShared = msgToPackedOffset.invertAndCompose(smemLayout);
    auto msgToOffset = getMsgToUnpackedOffsetLayout(msgToPackedOffset, smemTy);

    auto ctx = op.getContext();
    auto kMsg = str_attr("msg");
    auto kBlock = str_attr("block");
    const auto numCopies = msgToOffset.getInDimSize(kMsg);
    auto zero = b.i32_val(0);
    auto ctaId = targetInfo.getClusterCTAId(rewriter, loc);
    uint32_t maskCGABroadcast =
        smemLayout.getFreeVariableMasks().lookup(kBlock);
    bool multicast = op.getMulticast() && maskCGABroadcast != 0;
    Value multicastMask;
    Value barrierPtr = barrierMemObj.getBase();
    if (multicast) {
      multicastMask = createGPNPUTMAMulticastMask(loc, rewriter,
                                                  maskCGABroadcast, targetInfo);
      auto ctaIdInGroup = b.and_(ctaId, b.i32_val(maskCGABroadcast));
      pred = b.and_(pred, b.icmp_eq(ctaIdInGroup, b.i32_val(0)));
    }

    uint32_t barrierMask =
        ttg::toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);
    bool clusterBarrier = barrierMask & ~maskCGABroadcast;
    if (clusterBarrier) {
      uint32_t fullMask = ~(barrierMask << 24);
      Value barrierInt = b.ptrtoint(i32_ty, barrierPtr);
      barrierInt = b.and_(barrierInt, b.i32_val(fullMask));
      barrierPtr = b.inttoptr(barrierPtr.getType(), barrierInt);
    }

    std::string ctaGroup;
    if (ttng::getModuleTwoCTAs(op)) {
      // A module configured for two CTAs can still use per-CTA TMA barriers
      // and group-1 TMA copies.
      auto oneCTACGALayout = ttg::CGAEncodingAttr::get1DLayout(
          op->getContext(), ttg::lookupNumCTAs(op));
      bool oneCTABarrier =
          ttg::getCGALayout(barrierTy.getEncoding()) == oneCTACGALayout;
      ctaGroup = oneCTABarrier ? "cta_group::1." : "cta_group::2.";
    }

    // IFWA cannot reliably select the dynamic multi-warp TMA operand shape
    // produced by the newer LLVM (warpId = tid >> 5). Emit one statically
    // indexed message at a time from the elected lane of warp 0 instead. The
    // copies remain asynchronous; only message issuance is serialized.
    Value boxPred = b.and_(pred, b.icmp_ult(id, b.i32_val(warpSize)));
    for (int copyIdx = 0; copyIdx < numCopies; ++copyIdx) {
      GPNPUPTXBuilder ptxBuilderTMA;
      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = b.i32_val(copyIdx);
      Value shMemOffset =
          applyLinearLayout(loc, rewriter, msgToShared,
                            {{kMsg, copyIdxVal}, {kBlock, zero}})[0]
              .second;
      Value shMemPtr = b.gep(elemPtrTy, llvmElemTy, dstBase, shMemOffset);
      SmallVector<GPNPUPTXBuilder::Operand *> operands = {
          ptxBuilderTMA.newOperand(boxPred, "b"),
          ptxBuilderTMA.newOperand(shMemPtr, "r"),
          ptxBuilderTMA.newOperand(adaptor.getDesc(), "l")};
      std::string tmaInst =
          "@$0 cp.async.bulk.tensor." + std::to_string(rank) + "d." + ctaGroup +
          "shared::" + ((clusterBarrier || multicast) ? "cluster" : "cta") +
          ".global" + (isIm2Col ? ".im2col" : "") +
          ".mbarrier::complete_tx::bytes";
      if (multicast)
        tmaInst += ".multicast::cluster";
      tmaInst += " [$1], [$2, {";

      auto offsets = applyLinearLayout(loc, rewriter, msgToOffset,
                                       {{kMsg, copyIdxVal}, {kBlock, ctaId}});
      int operandIdx = 3;
      auto encoding = op.getDesc().getType().getSharedLayout();
      bool fp4Padded = ttng::isFp4Padded(encoding);
      for (int i = 0; i < rank; i++) {
        Value coord = adaptor.getCoord()[rank - i - 1];
        if (fp4Padded && i == 0)
          coord = b.mul(coord, b.i32_val(2));
        if (i < offsets.size())
          coord = b.add(coord, offsets[offsets.size() - i - 1].second);

        operands.push_back(ptxBuilderTMA.newOperand(coord, "r"));
        tmaInst += "$" + std::to_string(operandIdx++);
        if (i != rank - 1)
          tmaInst += ", ";
      }
      operands.push_back(ptxBuilderTMA.newOperand(barrierPtr, "r"));
      tmaInst += "}], [$" + std::to_string(operandIdx++) + "]";
      if (multicast) {
        operands.push_back(ptxBuilderTMA.newOperand(multicastMask, "h"));
        tmaInst += ", $" + std::to_string(operandIdx++);
      }
      if (isIm2Col) {
        auto im2colOffsets = adaptor.getOffsets();
        if (!im2colOffsets.empty()) {
          tmaInst += ", {";
          for (size_t i = 0; i < im2colOffsets.size(); i++) {
            operands.push_back(ptxBuilderTMA.newOperand(
                im2colOffsets[im2colOffsets.size() - 1 - i], "h"));
            tmaInst += "$" + std::to_string(operandIdx++);
            if (i != im2colOffsets.size() - 1)
              tmaInst += ", ";
          }
          tmaInst += "}";
        }
      }
      tmaInst += ";";

      auto &tma = *ptxBuilderTMA.create(tmaInst);
      tma(operands, /*onlyAttachMLIRArgs=*/true);
      ptxBuilderTMA.launch(rewriter, loc, voidTy);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// For asynchronous TMA stores, route cluster CTA id through Intellifusion
// TargetInfo and use the local elect predicate helper. Only tiled
// local-to-global stores and the corresponding wait are enabled.
LogicalResult
convertTMAStoreLikeOp(Operation *op, const TypeConverter *typeConverter,
                      ConversionPatternRewriter &rewriter, Value tmaPtr,
                      ttg::MemDescType srcTy, Value src, ValueRange coords,
                      StringRef tmaInst, const TargetInfo &targetInfo) {
  auto loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
  auto dstMemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, llvmElemTy, rewriter);
  Value dstBase = dstMemObj.getShmemAffineBase(loc, rewriter, srcTy);
  auto voidTy = void_ty(op->getContext());
  auto id = getThreadId(rewriter, loc);
  Value pred = createGPNPUElectPredicate(loc, rewriter);

  auto mod = op->getParentOfType<ModuleOp>();
  int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

  auto msgToPackedOffset =
      getMsgToPackedOffsetLayout(srcTy, ttg::TMAMode::Tiled);
  auto smemLayout = ttg::toLinearLayout(srcTy);
  auto msgToShared = msgToPackedOffset.invertAndCompose(smemLayout);
  auto msgToOffset = getMsgToUnpackedOffsetLayout(msgToPackedOffset, srcTy);

  auto ctx = op->getContext();
  auto kMsg = str_attr("msg");
  auto kBlock = str_attr("block");
  auto numCopies = msgToOffset.getInDimSize(kMsg);
  auto zero = b.i32_val(0);
  auto ctaId = targetInfo.getClusterCTAId(rewriter, loc);
  auto rank = coords.size();

  // Keep store/reduce TMA message issuance in the same IFWA-compatible static
  // form as global-to-local copies above.
  Value boxPred = b.and_(pred, b.icmp_ult(id, b.i32_val(warpSize)));
  for (int copyIdx = 0; copyIdx < numCopies; ++copyIdx) {
    GPNPUPTXBuilder ptxBuilderTMA;
    Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value copyIdxVal = b.i32_val(copyIdx);
    Value shMemOffset =
        applyLinearLayout(loc, rewriter, msgToShared,
                          {{kMsg, copyIdxVal}, {kBlock, zero}})[0]
            .second;
    Value shMemPtr = b.gep(elemPtrTy, llvmElemTy, dstBase, shMemOffset);
    SmallVector<GPNPUPTXBuilder::Operand *> operands = {
        ptxBuilderTMA.newOperand(boxPred, "b"),
        ptxBuilderTMA.newOperand(tmaPtr, "l")};

    auto offsets = applyLinearLayout(loc, rewriter, msgToOffset,
                                     {{kMsg, copyIdxVal}, {kBlock, ctaId}});
    bool fp4Padded = ttng::isFp4Padded(srcTy.getEncoding());
    for (int i = 0; i < rank; i++) {
      Value coord = coords[rank - i - 1];
      if (fp4Padded && i == 0)
        coord = b.mul(coord, b.i32_val(2));
      if (i < offsets.size())
        coord = b.add(coord, offsets[offsets.size() - i - 1].second);
      operands.push_back(ptxBuilderTMA.newOperand(coord, "r"));
    }
    operands.push_back(ptxBuilderTMA.newOperand(shMemPtr, "r"));
    auto &tma = *ptxBuilderTMA.create(tmaInst.str());
    tma(operands, /*onlyAttachMLIRArgs=*/true);
    ptxBuilderTMA.launch(rewriter, loc, voidTy);
  }

  NVVM::CpAsyncBulkCommitGroupOp::create(rewriter, loc);

  rewriter.eraseOp(op);
  return success();
}

struct AsyncTMACopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<ttng::AsyncTMACopyLocalToGlobalOp> {
  const TargetInfo &targetInfo;

  AsyncTMACopyLocalToGlobalOpConversion(LLVMTypeConverter &converter,
                                        const TargetInfo &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::AsyncTMACopyLocalToGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::ostringstream tmaInst;
    auto rank = op.getCoord().size();
    tmaInst << "@$0 cp.async.bulk.tensor." << rank;
    tmaInst << "d.global.shared::cta.bulk_group [$1, {";
    int operandIdx = 2;
    for (int i = 0; i < rank; i++) {
      tmaInst << "$" << operandIdx++;
      if (i != rank - 1)
        tmaInst << ", ";
    }
    tmaInst << "}], [$" << operandIdx++ << "];";
    return convertTMAStoreLikeOp(op, typeConverter, rewriter, adaptor.getDesc(),
                                 op.getSrc().getType(), adaptor.getSrc(),
                                 adaptor.getCoord(), tmaInst.str(), targetInfo);
  }
};

// IFWA diff: reuse the TargetInfo-aware IFWA store-like helper.
struct AsyncTMAReduceOpConversion
    : public ConvertOpToLLVMPattern<ttng::AsyncTMAReduceOp> {
  const TargetInfo &targetInfo;

  AsyncTMAReduceOpConversion(LLVMTypeConverter &converter,
                             const TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttng::AsyncTMAReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::ostringstream tmaInst;
    auto rank = op.getCoord().size();
    auto kind = stringifyDescriptorReduceKind(op.getKind());
    tmaInst << "@$0 cp.reduce.async.bulk.tensor." << rank;
    tmaInst << "d.global.shared::cta." << kind.str() << ".bulk_group [$1, {";
    int operandIdx = 2;
    for (int i = 0; i < rank; i++) {
      tmaInst << "$" << operandIdx++;
      if (i != rank - 1)
        tmaInst << ", ";
    }
    tmaInst << "}], [$" << operandIdx++ << "];";
    return convertTMAStoreLikeOp(op, typeConverter, rewriter, adaptor.getDesc(),
                                 op.getSrc().getType(), adaptor.getSrc(),
                                 adaptor.getCoord(), tmaInst.str(), targetInfo);
  }
};

struct TMAStoreWaitOpConversion
    : public ConvertOpToLLVMPattern<ttng::TMAStoreWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TMAStoreWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto isRead = UnitAttr::get(ctx);
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkWaitGroupOp>(
        op, op.getPendingsAttr(), isRead);
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass), targetInfo(targetInfo) {}

  const TargetInfo &targetInfo;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The shared pipeliner restores nonzero fallback values with a select after
    // the local load. The async copy itself zero-fills masked lanes.
    auto ctx = getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value mask = op.getMask();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    if (!resElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(op,
                                         "unsupported async-copy element type");
    unsigned elementBits = resElemTy.getIntOrFloatBitWidth();
    if (elementBits < 8 || elementBits > 128 || 128 % elementBits != 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported async-copy element width");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // zip(src, mask)
    SmallVector<Value> vals;
    auto ptrTy = srcElems[0].getType();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      auto maskElem = llMask ? maskElems[i] : b.false_val();
      packedArr = b.insert_val(packedArr, maskElem, 1);
      vals.push_back(packedArr);
    }

    // Remove broadcasted registers
    auto srcLayout = ttg::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    vals = removeBroadcastSrc.apply(vals);

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec = getContiguity(op.getSrc());
    if (mask) {
      maxVec = std::min(maxVec, getMaskAlignment(mask));
    }
    // If the op has a contiguity hint use it to increase the vector size.
    maxVec = std::max(maxVec, op.getContiguity());
    // The maximum vector size is 128 bits.
    maxVec = std::min(maxVec, 128 / elementBits);

    int vecBytes = maxVec * resElemTy.getIntOrFloatBitWidth() / 8;
    if (vecBytes < 4) {
      return emitError(loc, "cp.async does not support transfers smaller than "
                            "4 bytes; calculated this as ")
             << vecBytes << " bytes";
    }
    if (vecBytes != 16 && vecBytes != 8 && vecBytes != 4)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported async-copy vector width");

    auto freeVarMasks = getFreeVariableMasks(srcTy);
    // NOTE(@peterbell10): We load redundant data on different CTAs, so the data
    // is available in each CTAs respective shared memory. Otherwise, we would
    // need an additional broadcast step to copy the data between CTAs.
    freeVarMasks[str_attr("block")] = 0;
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, targetInfo);

    bool invalidCopy = false;
    auto emitCpAsync =
        [&b, &invalidCopy, threadPred, ptrTy, hasMask = bool(llMask)](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value ctaId) -> SmallVector<Value> {
      auto *ctx = rewriter.getContext();
      auto elemTy = vecTy.getElementType();
      auto nBytes = vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth() / 8;
      if (ctaId || (nBytes != 16 && nBytes != 8 && nBytes != 4)) {
        invalidCopy = true;
        return {};
      }
      // Tune CG and CA.
      CacheModifier srcCacheModifier =
          nBytes == 16 ? CacheModifier::CG : CacheModifier::CA;

      auto structElem = vals[startIdx];
      auto srcElem = b.extract_val(ptrTy, structElem, 0);
      auto maskElem = b.extract_val(i1_ty, structElem, 1);

      GPNPUPTXBuilder ptxBuilder;
      auto &copyAsyncOp =
          *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
      auto *dstOperand = ptxBuilder.newAddrOperand(shmemAddr, "r");
      auto *srcOperand = ptxBuilder.newAddrOperand(srcElem, "l");
      auto *copySize = ptxBuilder.newConstantOperand(nBytes);
      auto *srcSize = copySize;
      if (hasMask) {
        // We don't use predicate in this case, setting src-size to 0
        // if there's any mask. cp.async will automatically fill the
        // remaining slots with 0 if cp-size > src-size.
        auto selectOp = b.select(maskElem, b.i32_val(nBytes), b.i32_val(0));
        srcSize = ptxBuilder.newOperand(selectOp, "r");
      }
      copyAsyncOp(dstOperand, srcOperand, copySize, srcSize)
          .maybePredicate(threadPred);
      ptxBuilder.launch(rewriter, loc, void_ty(ctx));
      return {};
    };

    // %dst
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto smemLayout = ttg::toLinearLayout(dstTy);
    auto cvt = srcLayout.invertAndCompose(smemLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(loc,
                       "cp.async does not support non-trivial block dimension");
    }
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset =
        LLVM::SharedMemoryObject::getMaskSpanOffsets(dstTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(loc, ctx, cvt, vals, resElemTy, smemObj.getBase(),
              /*paddingShifts=*/{}, affineOffset, maskSpanAffineOffset, laneId,
              warpId, rewriter, targetInfo, maxVec, emitCpAsync);
    if (invalidCopy)
      return rewriter.notifyMatchFailure(op, "unsupported async-copy layout");

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto num = op->getAttrOfType<IntegerAttr>("num");
    if (!num || num.getInt() < 0 || num.getInt() > 7)
      return rewriter.notifyMatchFailure(op,
                                         "async wait group must be in [0, 7]");
    NVVM::CpAsyncWaitGroupOp::create(rewriter, loc, num);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    NVVM::CpAsyncCommitGroupOp::create(rewriter, loc);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

void populateGPNPUTMALoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns
      .add<AsyncTMACopyGlobalToLocalOpConversion,
           AsyncTMACopyLocalToGlobalOpConversion, AsyncTMAReduceOpConversion,
           AsyncTMAGatherOpConversion, AsyncTMAScatterOpConversion>(
          typeConverter, targetInfo, benefit);
  patterns.add<TMAStoreWaitOpConversion>(typeConverter, benefit);
}

void populateGPNPUMemoryToLLVMPatterns(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<AsyncCopyGlobalToLocalOpConversion>(typeConverter, targetInfo,
                                                   axisInfoAnalysis, benefit);
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<StoreOpConversion>(typeConverter, targetInfo, axisInfoAnalysis,
                                  benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, targetInfo,
                                      axisInfoAnalysis, benefit);
  patterns.add<BarrierOpConversion>(typeConverter, PatternBenefit(100));
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
