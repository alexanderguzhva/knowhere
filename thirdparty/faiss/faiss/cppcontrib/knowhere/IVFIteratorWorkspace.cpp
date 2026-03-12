/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IVFIteratorWorkspace.h>

#include <cinttypes>

#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexScaNN.h>
#include <faiss/cppcontrib/knowhere/impl/pq4_fast_scan.h>
#include <faiss/cppcontrib/knowhere/impl/simd_result_handlers.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss::cppcontrib::knowhere {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

// ---- IVFIteratorWorkspace ----

IVFIteratorWorkspace::IVFIteratorWorkspace(
        const float* query_data_in,
        const size_t d,
        const IVFSearchParameters* search_params)
        : query_data(query_data_in, query_data_in + d),
          search_params(search_params),
          dis_refine(nullptr) {}

IVFIteratorWorkspace::~IVFIteratorWorkspace() {}

// ---- ScaNNIteratorWorkspace ----

ScaNNIteratorWorkspace::ScaNNIteratorWorkspace(
        const IndexScaNN* scann_index,
        const float* query_data,
        const IVFSearchParameters* params)
        : inner(std::make_unique<IVFFastScanIteratorWorkspace>(
                  dynamic_cast<const IndexIVFPQFastScan*>(scann_index->base_index),
                  query_data,
                  params)) {
    auto* fast_scan_base =
            dynamic_cast<const IndexIVFPQFastScan*>(scann_index->base_index);
    // Set up dis_refine (logic from IndexScaNN::getIteratorWorkspace)
    if (scann_index->refine_index) {
        auto refine = dynamic_cast<const IndexFlat*>(scann_index->refine_index);
        if (auto base_cosine =
                    dynamic_cast<const IndexIVFPQFastScanCosine*>(fast_scan_base)) {
            this->dis_refine = std::unique_ptr<faiss::DistanceComputer>(
                    new faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer(
                            base_cosine->inverse_norms_storage.inverse_l2_norms.data(),
                            base_cosine->d,
                            std::unique_ptr<faiss::DistanceComputer>(
                                    refine->get_distance_computer())));
        } else {
            this->dis_refine = std::unique_ptr<faiss::DistanceComputer>(
                    refine->get_FlatCodesDistanceComputer());
        }
        this->dis_refine->set_query(query_data);
    } else {
        this->dis_refine = nullptr;
    }
    this->search_params = inner->search_params;
}

void ScaNNIteratorWorkspace::next_batch(size_t current_backup_count) {
    inner->next_batch(current_backup_count);
    this->dists = std::move(inner->dists);
}

// ---- IVFBaseIteratorWorkspace ----

IVFBaseIteratorWorkspace::IVFBaseIteratorWorkspace(
        const IndexIVF* ivf_index_in,
        const float* query_data_in,
        const IVFSearchParameters* params)
        : IVFIteratorWorkspace(query_data_in, ivf_index_in->d, params),
          ivf_index(ivf_index_in) {
    // Snapshot list sizes
    auto coarse_list_sizes_buf = std::make_unique<size_t[]>(ivf_index->nlist);
    size_t count = 0;
    auto max_coarse_list_size = 0;
    for (size_t list_no = 0; list_no < ivf_index->nlist; ++list_no) {
        auto list_size = ivf_index->invlists->list_size(list_no);
        coarse_list_sizes_buf[list_no] = list_size;
        count += list_size;
        if (list_size > max_coarse_list_size) {
            max_coarse_list_size = list_size;
        }
    }

    // Compute nprobe and backup_count_threshold
    size_t np = this->search_params->nprobe
            ? this->search_params->nprobe
            : ivf_index->nprobe;
    np = std::min(ivf_index->nlist, np);
    this->backup_count_threshold = count * np / ivf_index->nlist;
    auto max_backup_count =
            max_coarse_list_size + this->backup_count_threshold;

    // Coarse quantization
    auto coarse_idx_buf = std::make_unique<idx_t[]>(ivf_index->nlist);
    auto coarse_dis_buf = std::make_unique<float[]>(ivf_index->nlist);
    ivf_index->quantizer->search(
            1,
            this->query_data.data(),
            ivf_index->nlist,
            coarse_dis_buf.get(),
            coarse_idx_buf.get(),
            this->search_params
                    ? this->search_params->quantizer_params
                    : nullptr);

    this->coarse_idx = std::move(coarse_idx_buf);
    this->coarse_dis = std::move(coarse_dis_buf);
    this->coarse_list_sizes = std::move(coarse_list_sizes_buf);
    this->nprobe = np;
    this->dists.reserve(max_backup_count);
}

void IVFBaseIteratorWorkspace::next_batch(size_t current_backup_count) {
    this->dists.clear();

    while (current_backup_count + this->dists.size() <
                   this->backup_count_threshold &&
           this->next_visit_coarse_list_idx < ivf_index->nlist) {
        auto next_list_idx = this->next_visit_coarse_list_idx;
        this->next_visit_coarse_list_idx++;

        ivf_index->invlists->prefetch_lists(
                this->coarse_idx.get() + next_list_idx, 1);
        const auto list_no = this->coarse_idx[next_list_idx];
        const auto coarse_list_centroid_dist =
                this->coarse_dis[next_list_idx];

        // max_codes is the size of the list when we started the
        // iteration so that we won't search vectors added during the
        // iteration (for IVFCC).
        const auto max_codes = this->coarse_list_sizes
                                       [this->coarse_idx[next_list_idx]];
        if (list_no < 0) {
            // not enough centroids for multiprobe
            continue;
        }
        FAISS_THROW_IF_NOT_FMT(
                list_no < (idx_t)ivf_index->nlist,
                "Invalid list_no=%" PRId64 " nlist=%zd\n",
                list_no,
                ivf_index->nlist);

        // don't waste time on empty lists
        void* inverted_list_context = this->search_params
                ? this->search_params->inverted_list_context
                : nullptr;

        if (ivf_index->invlists->is_empty(list_no, inverted_list_context)) {
            continue;
        }

        // get scanner
        IDSelector* sel = this->search_params
                ? this->search_params->sel
                : nullptr;
        std::unique_ptr<InvertedListScanner> scanner(
                ivf_index->get_InvertedListScanner(false, sel, this->search_params));
        scanner->set_query(this->query_data.data());
        scanner->set_list(list_no, coarse_list_centroid_dist);

        size_t segment_num = ivf_index->invlists->get_segment_num(list_no);
        size_t scan_cnt = 0;
        for (size_t segment_idx = 0; segment_idx < segment_num; segment_idx++) {
            size_t segment_size =
                    ivf_index->invlists->get_segment_size(list_no, segment_idx);
            size_t should_scan_size =
                    std::min(segment_size, max_codes - scan_cnt);
            scan_cnt += should_scan_size;
            if (should_scan_size <= 0) {
                break;
            }
            size_t segment_offset =
                    ivf_index->invlists->get_segment_offset(list_no, segment_idx);
            InvertedLists::ScopedCodes scodes(
                    ivf_index->invlists, list_no, segment_offset);
            InvertedLists::ScopedCodeNorms scode_norms(
                    ivf_index->invlists, list_no, segment_offset);
            InvertedLists::ScopedIds sids(
                    ivf_index->invlists, list_no, segment_offset);

            scanner->scan_codes_and_return(
                    should_scan_size,
                    scodes.get(),
                    scode_norms.get(),
                    sids.get(),
                    this->dists);
        }
    }
}

// ---- IVFFastScanIteratorWorkspace ----

IVFFastScanIteratorWorkspace::IVFFastScanIteratorWorkspace(
        const IndexIVFFastScan* index_in,
        const float* query_data,
        const IVFSearchParameters* params)
        : IVFIteratorWorkspace(query_data, index_in->d, params),
          index(index_in) {
    auto coarse_list_sizes_buf = std::make_unique<size_t[]>(index->nlist);
    size_t count = 0;
    auto max_coarse_list_size = 0;
    for (size_t list_no = 0; list_no < index->nlist; ++list_no) {
        auto list_size = index->invlists->list_size(list_no);
        coarse_list_sizes_buf[list_no] = list_size;
        count += list_size;
        if (list_size > max_coarse_list_size) {
            max_coarse_list_size = list_size;
        }
    }

    size_t np = this->search_params->nprobe
            ? this->search_params->nprobe
            : index->nprobe;
    np = std::min(index->nlist, np);
    this->backup_count_threshold = count * np / index->nlist;
    auto max_backup_count =
            max_coarse_list_size + this->backup_count_threshold;

    auto coarse_idx_buf = std::make_unique<idx_t[]>(index->nlist);
    auto coarse_dis_buf = std::make_unique<float[]>(index->nlist);
    index->quantizer->search(
            1,
            this->query_data.data(),
            index->nlist,
            coarse_dis_buf.get(),
            coarse_idx_buf.get(),
            this->search_params
                    ? this->search_params->quantizer_params
                    : nullptr);

    this->coarse_idx = std::move(coarse_idx_buf);
    this->coarse_dis = std::move(coarse_dis_buf);
    this->coarse_list_sizes = std::move(coarse_list_sizes_buf);
    this->nprobe = np;
    this->dists.reserve(max_backup_count);

    this->dim12 = index->ksub * index->M2;
    IndexIVFFastScan::CoarseQuantized cq{
            this->nprobe,
            this->coarse_dis.get(),
            this->coarse_idx.get()};
    index->compute_LUT_uint8(
            1,
            this->query_data.data(),
            cq,
            this->dis_tables,
            this->biases,
            this->normalizers);
}

void IVFFastScanIteratorWorkspace::next_batch(size_t current_backup_count) {
    this->dists.clear();

    std::unique_ptr<SIMDResultHandlerToFloat> handler;
    bool is_max = !faiss::is_similarity_metric(index->metric_type);
    auto id_selector = this->search_params->sel
            ? this->search_params->sel
            : nullptr;

    if (is_max) {
        handler.reset(
                new SingleQueryResultCollectHandler<
                        CMax<uint16_t, int64_t>,
                        true>(this->dists, index->ntotal, id_selector));
    } else {
        handler.reset(
                new SingleQueryResultCollectHandler<
                        CMin<uint16_t, int64_t>,
                        true>(this->dists, index->ntotal, id_selector));
    }

    this->get_interator_next_batch_implem_10(
            *handler.get(), current_backup_count);
}

void IVFFastScanIteratorWorkspace::get_interator_next_batch_implem_10(
        SIMDResultHandlerToFloat& handler,
        size_t current_backup_count) {
    bool single_LUT = !index->lookup_table_is_3d();
    handler.begin(index->skip & 16 ? nullptr : this->normalizers);
    auto dim12 = this->dim12;
    const uint8_t* LUT = nullptr;

    if (single_LUT) {
        LUT = this->dis_tables.get();
    }
    while (current_backup_count + this->dists.size() <
                   this->backup_count_threshold &&
           this->next_visit_coarse_list_idx < index->nlist) {
        auto next_list_idx = this->next_visit_coarse_list_idx;
        this->next_visit_coarse_list_idx++;
        if (!single_LUT) {
            LUT = this->dis_tables.get() + next_list_idx * dim12;
        }
        index->invlists->prefetch_lists(
                this->coarse_idx.get() + next_list_idx, 1);
        if (this->biases.get()) {
            handler.dbias = this->biases.get() + next_list_idx;
        }
        idx_t list_no = this->coarse_idx[next_list_idx];
        size_t ls = index->invlists->list_size(list_no);
        if (list_no < 0 || ls == 0)
            continue;

        InvertedLists::ScopedCodes codes(index->invlists, list_no);
        InvertedLists::ScopedIds ids(index->invlists, list_no);
        handler.ntotal = ls;
        handler.id_map = ids.get();
        pq4_accumulate_loop(
                1,
                roundup(ls, index->bbs),
                index->bbs,
                index->M2,
                codes.get(),
                LUT,
                handler,
                nullptr);
    }
    handler.end();
}

}  // namespace faiss::cppcontrib::knowhere
