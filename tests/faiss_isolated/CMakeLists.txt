cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
include(GNUInstallDirs)
include(ExternalProject)
include(cmake/utils/utils.cmake)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(glog REQUIRED libglog)

set(FAISS_TEST_SRCS
#    ../../thirdparty/faiss/tests/test_omp_threads.cpp
#    ../../thirdparty/faiss/tests/test_heap.cpp
#    ../../thirdparty/faiss/tests/test_partitioning.cpp
#    ../../thirdparty/faiss/tests/test_simdlib.cpp
#    ../../thirdparty/faiss/tests/test_approx_topk.cpp
#    ../../thirdparty/faiss/tests/test_code_distance.cpp
#    ../../thirdparty/faiss/tests/test_cppcontrib_uintreader.cpp
#    ../../thirdparty/faiss/tests/test_distances_simd.cpp
#    ../../thirdparty/faiss/tests/test_hnsw.cpp
#    ../../thirdparty/faiss/tests/test_binary_flat.cpp
#    ../../thirdparty/faiss/tests/test_cppcontrib_sa_decode.cpp
#    ../../thirdparty/faiss/tests/test_ondisk_ivf.cpp
#    ../../thirdparty/faiss/tests/test_pq_encoding.cpp
#    ../../thirdparty/faiss/tests/test_RCQ_cropping.cpp
#    ../../thirdparty/faiss/tests/test_ivfpq_codec.cpp
#    ../../thirdparty/faiss/tests/test_ivfpq_indexing.cpp
#    ../../thirdparty/faiss/tests/test_merge.cpp
#    ../../thirdparty/faiss/tests/test_threaded_index.cpp
#    ../../thirdparty/faiss/tests/test_pairs_decoding.cpp
#    ../../thirdparty/faiss/tests/test_dealloc_invlists.cpp
#    ../../thirdparty/faiss/tests/test_params_override.cpp
    ../../thirdparty/faiss/tests/test_sliding_ivf.cpp
#    ../../thirdparty/faiss/tests/test_transfer_invlists.cpp

../../thirdparty/faiss/faiss/utils/binary_distances.cpp
../../thirdparty/faiss/faiss/utils/bit_table.cpp
../../thirdparty/faiss/faiss/utils/distances.cpp
../../thirdparty/faiss/faiss/utils/distances_simd.cpp
../../thirdparty/faiss/faiss/utils/extra_distances.cpp
../../thirdparty/faiss/faiss/utils/hamming.cpp
../../thirdparty/faiss/faiss/utils/Heap.cpp
../../thirdparty/faiss/faiss/utils/partitioning.cpp
../../thirdparty/faiss/faiss/utils/quantize_lut.cpp
../../thirdparty/faiss/faiss/utils/random.cpp
../../thirdparty/faiss/faiss/utils/sorting.cpp
../../thirdparty/faiss/faiss/utils/utils.cpp
../../thirdparty/faiss/faiss/utils/WorkerThread.cpp
../../thirdparty/faiss/faiss/impl/AdditiveQuantizer.cpp
../../thirdparty/faiss/faiss/impl/AuxIndexStructures.cpp
../../thirdparty/faiss/faiss/impl/CodePacker.cpp
../../thirdparty/faiss/faiss/impl/FaissException.cpp
../../thirdparty/faiss/faiss/impl/HNSW.cpp
../../thirdparty/faiss/faiss/impl/IDSelector.cpp
../../thirdparty/faiss/faiss/impl/io.cpp
../../thirdparty/faiss/faiss/impl/index_read.cpp
../../thirdparty/faiss/faiss/impl/index_write.cpp
../../thirdparty/faiss/faiss/impl/kmeans1d.cpp
../../thirdparty/faiss/faiss/impl/lattice_Zn.cpp
../../thirdparty/faiss/faiss/impl/LocalSearchQuantizer.cpp
../../thirdparty/faiss/faiss/impl/NNDescent.cpp
../../thirdparty/faiss/faiss/impl/NSG.cpp
../../thirdparty/faiss/faiss/impl/PolysemousTraining.cpp
../../thirdparty/faiss/faiss/impl/pq4_fast_scan.cpp
../../thirdparty/faiss/faiss/impl/pq4_fast_scan_search_1.cpp
../../thirdparty/faiss/faiss/impl/pq4_fast_scan_search_qbs.cpp
../../thirdparty/faiss/faiss/impl/ProductAdditiveQuantizer.cpp
../../thirdparty/faiss/faiss/impl/ProductQuantizer.cpp
../../thirdparty/faiss/faiss/impl/ResidualQuantizer.cpp
../../thirdparty/faiss/faiss/impl/residual_quantizer_encode_steps.cpp
../../thirdparty/faiss/faiss/impl/ScalarQuantizer.cpp
../../thirdparty/faiss/faiss/impl/ScalarQuantizerDC.cpp
../../thirdparty/faiss/faiss/impl/ScalarQuantizerDC_avx.cpp
###../../thirdparty/faiss/faiss/impl/ScalarQuantizerDC_avx512.cpp
../../thirdparty/faiss/faiss/impl/ScalarQuantizerOp.cpp
../../thirdparty/faiss/faiss/invlists/BlockInvertedLists.cpp
../../thirdparty/faiss/faiss/invlists/DirectMap.cpp
../../thirdparty/faiss/faiss/invlists/InvertedLists.cpp
../../thirdparty/faiss/faiss/invlists/InvertedListsIOHook.cpp
../../thirdparty/faiss/faiss/invlists/OnDiskInvertedLists.cpp


../../thirdparty/faiss/faiss/Clustering.cpp
../../thirdparty/faiss/faiss/FaissHook.cpp
../../thirdparty/faiss/faiss/Index.cpp
../../thirdparty/faiss/faiss/Index2Layer.cpp
../../thirdparty/faiss/faiss/IndexFastScan.cpp
../../thirdparty/faiss/faiss/IndexFlatCodes.cpp
../../thirdparty/faiss/faiss/IndexFlat.cpp
../../thirdparty/faiss/faiss/IndexHNSW.cpp
../../thirdparty/faiss/faiss/IndexIDMap.cpp
../../thirdparty/faiss/faiss/IndexIVF.cpp
../../thirdparty/faiss/faiss/IndexIVFSpectralHash.cpp
../../thirdparty/faiss/faiss/IndexIVFFastScan.cpp
../../thirdparty/faiss/faiss/IndexIVFIndependentQuantizer.cpp
../../thirdparty/faiss/faiss/IndexIVFFlat.cpp
../../thirdparty/faiss/faiss/IndexIVFPQ.cpp
../../thirdparty/faiss/faiss/IndexIVFPQR.cpp
../../thirdparty/faiss/faiss/IndexIVFPQFastScan.cpp
../../thirdparty/faiss/faiss/IndexLattice.cpp
../../thirdparty/faiss/faiss/IndexLSH.cpp
../../thirdparty/faiss/faiss/IndexNNDescent.cpp
../../thirdparty/faiss/faiss/IndexNSG.cpp
../../thirdparty/faiss/faiss/IndexPQ.cpp
../../thirdparty/faiss/faiss/IndexPQFastScan.cpp
../../thirdparty/faiss/faiss/IndexPreTransform.cpp
../../thirdparty/faiss/faiss/IndexRowwiseMinMax.cpp
../../thirdparty/faiss/faiss/IndexAdditiveQuantizer.cpp
../../thirdparty/faiss/faiss/IndexAdditiveQuantizerFastScan.cpp
../../thirdparty/faiss/faiss/IndexIVFAdditiveQuantizer.cpp
../../thirdparty/faiss/faiss/IndexIVFAdditiveQuantizerFastScan.cpp
../../thirdparty/faiss/faiss/IndexScalarQuantizer.cpp
../../thirdparty/faiss/faiss/VectorTransform.cpp

../../thirdparty/faiss/faiss/IndexBinaryFlat.cpp
../../thirdparty/faiss/faiss/IndexBinary.cpp
../../thirdparty/faiss/faiss/IndexBinaryHash.cpp
../../thirdparty/faiss/faiss/IndexBinaryHNSW.cpp
../../thirdparty/faiss/faiss/IndexBinaryIVF.cpp
../../thirdparty/faiss/faiss/IndexBinaryFromFloat.cpp
../../thirdparty/faiss/faiss/IndexRefine.cpp
../../thirdparty/faiss/faiss/IndexShards.cpp
../../thirdparty/faiss/faiss/IndexShardsIVF.cpp

../../thirdparty/faiss/faiss/IndexReplicas.cpp
../../thirdparty/faiss/faiss/clone_index.cpp
../../thirdparty/faiss/faiss/IVFlib.cpp
../../thirdparty/faiss/faiss/index_factory.cpp

../../thirdparty/faiss/faiss/MetaIndexes.cpp
../../thirdparty/faiss/faiss/AutoTune.cpp

../../src/common/utils.cc
../../src/io/memory_io.cc
../../src/simd/hook.cc
../../src/simd/distances_avx.cc
###../../src/simd/distances_avx512.cc
../../src/simd/distances_ref.cc
../../src/simd/distances_sse.cc
)

set(FAISS_HEADER_DIRS
    ../../thirdparty/
    ../../thirdparty/faiss/
    ../../include
    ../../src
)



include(FetchContent)



FetchContent_Declare(gflags
    GIT_REPOSITORY	https://github.com/gflags/gflags.git
    GIT_TAG			master
)
#FetchContent_MakeAvailable(gflags)
FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
#    cmake_policy(SET CMP0069 NEW)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
#FetchContent_GetProperties(cmark)
#if(NOT cmark_POPULATED)
#  FetchContent_Populate(cmark)
#  add_subdirectory(${cmark_SOURCE_DIR} ${cmark_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif()



FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)
#find_package(GTest REQUIRED)


  add_executable(faiss_tests ${FAISS_TEST_SRCS})

  target_compile_options(
    faiss_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)
  target_include_directories(faiss_tests PUBLIC ${FAISS_HEADER_DIRS})

  target_link_libraries(
    faiss_tests PUBLIC GTest::gtest GTest::gtest_main)
  target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)


  find_package(BLAS REQUIRED)
  message(STATUS "Faiss BLAS libraries: ${BLAS_LIBRARIES}")
  target_link_libraries(faiss_tests PRIVATE ${BLAS_LIBRARIES})
  #target_link_libraries(faiss_avx2 PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  message(STATUS "Faiss LAPACK libraries: ${BLAS_LIBRARIES}")
  target_link_libraries(faiss_tests PRIVATE ${LAPACK_LIBRARIES})

find_package(OpenMP REQUIRED)
target_link_libraries(faiss_tests PRIVATE OpenMP::OpenMP_CXX)


set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags REQUIRED)
#find_package(gflags COMPONENTS static)
target_link_libraries(faiss_tests PUBLIC gflags::gflags)


find_package(glog REQUIRED)
target_link_libraries(faiss_tests PUBLIC glog::glog)


target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)

#target_compile_options(faiss_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx -mavx512f -mavx512dq -mavx512bw -mf16c -mpopcnt>)
target_compile_options(faiss_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx -mf16c -mpopcnt>)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_tests)
