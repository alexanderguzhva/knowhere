// /**
//  * Copyright (c) Facebook, Inc. and its affiliates.
//  *
//  * This source code is licensed under the MIT license found in the
//  * LICENSE file in the root directory of this source tree.
//  */

// // -*- c++ -*-

// #include <faiss/IndexBinaryIVF.h>

// #include <omp.h>
// #include <cinttypes>
// #include <cstdio>

// #include <algorithm>
// #include <memory>

// #include <faiss/Index.h>
// #include <faiss/IndexFlat.h>
// #include <faiss/IndexLSH.h>
// #include <faiss/impl/AuxIndexStructures.h>
// #include <faiss/impl/FaissAssert.h>
// #include <faiss/utils/Heap.h>
// #include <faiss/utils/binary_distances.h>
// #include <faiss/utils/hamming.h>
// #include <faiss/utils/jaccard-inl.h>
// #include <faiss/utils/utils.h>
// #include <cinttypes>
// namespace faiss {

// namespace {

// template <bool store_pairs>
// BinaryInvertedListScanner* select_IVFBinaryScannerL2(size_t code_size) {
// #define HC(name) return new IVFBinaryScannerL2<name>(code_size, store_pairs)
//     switch (code_size) {
//         case 4:
//             HC(HammingComputer4);
//         case 8:
//             HC(HammingComputer8);
//         case 16:
//             HC(HammingComputer16);
//         case 20:
//             HC(HammingComputer20);
//         case 32:
//             HC(HammingComputer32);
//         case 64:
//             HC(HammingComputer64);
//         default:
//             HC(HammingComputerDefault);
//     }
// #undef HC
// }

// template <bool store_pairs>
// BinaryInvertedListScanner* select_IVFBinaryScannerJaccard(size_t code_size) {
// #define HANDLE_CS(cs)                                                         \
//     case cs:                                                                  \
//         return new IVFBinaryScannerJaccard<JaccardComputer##cs, store_pairs>( \
//                 cs);
//     switch (code_size) {
//         HANDLE_CS(16)
//         HANDLE_CS(32)
//         HANDLE_CS(64)
//         HANDLE_CS(128)
//         HANDLE_CS(256)
//         HANDLE_CS(512)
//         default:
//             return new IVFBinaryScannerJaccard<
//                     JaccardComputerDefault,
//                     store_pairs>(code_size);
//     }
// #undef HANDLE_CS
// }

// template <bool store_pairs>
// void search_knn_hamming_count_1(
//         const IndexBinaryIVF& ivf,
//         size_t nx,
//         const uint8_t* x,
//         const idx_t* keys,
//         int k,
//         int32_t* distances,
//         idx_t* labels,
//         const IVFSearchParameters* params,
//         const size_t nprobe,
//         const BitsetView bitset) {
//     switch (ivf.code_size) {
// #define HANDLE_CS(cs)                         \
//     case cs:                                  \
//         search_knn_hamming_count_thread_safe< \
//                 HammingComputer##cs,          \
//                 store_pairs>(                 \
//                 ivf,                          \
//                 nx,                           \
//                 x,                            \
//                 keys,                         \
//                 k,                            \
//                 distances,                    \
//                 labels,                       \
//                 params,                       \
//                 nprobe,                       \
//                 bitset);                      \
//         break;
//         HANDLE_CS(4);
//         HANDLE_CS(8);
//         HANDLE_CS(16);
//         HANDLE_CS(20);
//         HANDLE_CS(32);
//         HANDLE_CS(64);
// #undef HANDLE_CS
//         default:
//             search_knn_hamming_count_thread_safe<
//                     HammingComputerDefault,
//                     store_pairs>(
//                     ivf,
//                     nx,
//                     x,
//                     keys,
//                     k,
//                     distances,
//                     labels,
//                     params,
//                     nprobe,
//                     bitset);
//             break;
//     }
// }

// } // namespace

// void IndexBinaryIVF::search_preassigned_thread_safe(
//         idx_t n,
//         const uint8_t* x,
//         idx_t k,
//         const idx_t* idx,
//         const int32_t* coarse_dis,
//         int32_t* distances,
//         idx_t* labels,
//         bool store_pairs,
//         const IVFSearchParameters* params,
//         const size_t nprobe,
//         const BitsetView bitset) const {
//     if (metric_type == METRIC_Jaccard) {
//         if (use_heap) {
//             float* D = new float[k * n];
//             float* c_dis = new float[n * nprobe];
//             memcpy(c_dis, coarse_dis, sizeof(float) * n * nprobe);
//             search_knn_binary_dis_heap_thread_safe(
//                     *this,
//                     n,
//                     x,
//                     k,
//                     idx,
//                     c_dis,
//                     D,
//                     labels,
//                     store_pairs,
//                     params,
//                     nprobe,
//                     bitset);
//             memcpy(distances, D, sizeof(float) * n * k);
//             delete[] D;
//             delete[] c_dis;
//         } else {
//             // not implemented
//         }
//     } else if (
//             metric_type == METRIC_Substructure ||
//             metric_type == METRIC_Superstructure) {
//         // unsupported
//     } else {
//         // // aguzhva: good
//         // if (use_heap) {
//         //     search_knn_hamming_heap_thread_safe(
//         //             *this,
//         //             n,
//         //             x,
//         //             k,
//         //             idx,
//         //             coarse_dis,
//         //             distances,
//         //             labels,
//         //             store_pairs,
//         //             params,
//         //             nprobe,
//         //             bitset);
//         // } else {
//         //     if (store_pairs) {
//         //         search_knn_hamming_count_1<true>(
//         //                 *this,
//         //                 n,
//         //                 x,
//         //                 idx,
//         //                 k,
//         //                 distances,
//         //                 labels,
//         //                 params,
//         //                 nprobe,
//         //                 bitset);
//         //     } else {
//         //         search_knn_hamming_count_1<false>(
//         //                 *this,
//         //                 n,
//         //                 x,
//         //                 idx,
//         //                 k,
//         //                 distances,
//         //                 labels,
//         //                 params,
//         //                 nprobe,
//         //                 bitset);
//         //     }
//         // }
//     }
// }


// } // namespace faiss
