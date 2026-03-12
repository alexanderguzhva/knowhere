/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <memory>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/AlignedTable.h>

#include "knowhere/object.h"

namespace faiss {

struct SIMDResultHandlerToFloat;

namespace cppcontrib {
namespace knowhere {

struct SearchParametersIVF;
using IVFSearchParameters = SearchParametersIVF;

struct IndexIVF;
struct IndexIVFFastScan;
struct IndexScaNN;
struct IndexIVFPQFastScan;

/// Base workspace for IVF iterator lifecycle.
/// Holds query data, coarse quantization results, and iteration state.
struct IVFIteratorWorkspace {
    IVFIteratorWorkspace() = default;
    IVFIteratorWorkspace(
            const float* query_data,
            const size_t d,
            const IVFSearchParameters* search_params);
    virtual ~IVFIteratorWorkspace();

    /// Scan the next batch of inverted lists, populating this->dists.
    virtual void next_batch(size_t current_backup_count) = 0;

    std::vector<float> query_data; // a copy of a single query
    const IVFSearchParameters* search_params = nullptr;
    size_t nprobe = 0;
    size_t backup_count_threshold = 0;   // count * nprobe / nlist
    std::vector<::knowhere::DistId> dists; // should be cleared after each use
    size_t next_visit_coarse_list_idx = 0;
    std::unique_ptr<float[]> coarse_dis =
            nullptr; // backup coarse centroids distances (heap)
    std::unique_ptr<idx_t[]> coarse_idx =
            nullptr; // backup coarse centroids ids (heap)
    std::unique_ptr<size_t[]> coarse_list_sizes =
            nullptr; // snapshot of the list_size
    std::unique_ptr<DistanceComputer> dis_refine;
};

/// Iterator workspace for FastScan index types.
/// Adds SIMD look-up tables, normalizers, and the index pointer for scanning.
struct IVFFastScanIteratorWorkspace : IVFIteratorWorkspace {
    const IndexIVFFastScan* index;

    IVFFastScanIteratorWorkspace(
            const IndexIVFFastScan* index,
            const float* query_data,
            const IVFSearchParameters* params);

    void next_batch(size_t current_backup_count) override;

    // one query call, no qbs
    void get_interator_next_batch_implem_10(
            faiss::SIMDResultHandlerToFloat& handler,
            size_t current_backup_count);

    size_t dim12;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    float normalizers[2];
};

/// Standalone iterator workspace for IndexScaNN.
/// Wraps an IVFFastScanIteratorWorkspace with optional refinement.
struct ScaNNIteratorWorkspace : IVFIteratorWorkspace {
    /// The inner FastScan workspace.
    std::unique_ptr<IVFIteratorWorkspace> inner;

    ScaNNIteratorWorkspace(const IndexScaNN* scann_index,
                           const float* query_data,
                           const IVFSearchParameters* params);

    void next_batch(size_t current_backup_count) override;
};

/// Standalone iterator workspace for base IVF index types
/// (IndexIVFFlat, IndexIVFFlatCC, IndexIVFScalarQuantizerCC, IndexIVFScalarQuantizer).
/// Contains the full init and scanning logic from IndexIVF.
struct IVFBaseIteratorWorkspace : IVFIteratorWorkspace {
    const IndexIVF* ivf_index;

    IVFBaseIteratorWorkspace(const IndexIVF* ivf_index,
                             const float* query_data,
                             const IVFSearchParameters* params);

    void next_batch(size_t current_backup_count) override;
};

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss
