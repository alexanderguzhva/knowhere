#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexScaNN.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/***************************************************
 * IndexScaNN
 ***************************************************/

IndexScaNN::IndexScaNN(Index* base_index) : IndexRefineFlat(base_index) {}

IndexScaNN::IndexScaNN(Index* base_index, const float* xb)
        : IndexRefineFlat(base_index, xb) {}

IndexScaNN::IndexScaNN() : IndexRefineFlat() {}

namespace {

template <class C>
static void reorder_2_heaps(
        idx_t n,
        idx_t k,
        idx_t* labels,
        float* distances,
        idx_t k_base,
        const idx_t* base_labels,
        const float* base_distances) {
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t* idxo = labels + i * k;
        float* diso = distances + i * k;
        const idx_t* idxi = base_labels + i * k_base;
        const float* disi = base_distances + i * k_base;

        heap_heapify<C>(k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C>(k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C>(k, diso, idxo);
    }
}

} // anonymous namespace

int64_t IndexScaNN::size() {
    auto index_ = dynamic_cast<const IndexIVFPQFastScan*>(base_index);
    FAISS_THROW_IF_NOT(index_);

    auto nb = index_->invlists->compute_ntotal();
    auto code_size = index_->code_size;
    auto pq = index_->pq;
    auto nlist = index_->nlist;
    auto d = index_->d;

    auto capacity =
            nb * code_size + nb * sizeof(int64_t) + nlist * d * sizeof(float);
    auto centroid_table = pq.M * pq.ksub * pq.dsub * sizeof(float);
    auto precomputed_table = nlist * pq.M * pq.ksub * sizeof(float);

    auto raw_data = (refine_index ? index_->ntotal * d * sizeof(float) : 0);
    return (capacity + centroid_table + precomputed_table + raw_data);
}

void IndexScaNN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    const IndexScaNNSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IndexScaNNSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexScaNN params have incorrect type");
    }

    idx_t k_base = (params != nullptr) ? params->reorder_k : idx_t(k * k_factor);
    SearchParameters* base_index_params = 
        (params != nullptr) ? params->base_index_params : nullptr;

    FAISS_THROW_IF_NOT(k_base >= k);

    auto base = dynamic_cast<const IndexIVFPQFastScan*>(base_index);
    FAISS_THROW_IF_NOT(base);

    // nothing to refine, directly return result
    if (refine_index == nullptr) {
        base->search(
            n,
            x,
            k,
            distances,
            labels,
            base_index_params);
        return;
    }

    idx_t* base_labels = labels;
    float* base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;

    if (k != k_base) {
        base_labels = new idx_t[n * k_base];
        del1.set(base_labels);
        base_distances = new float[n * k_base];
        del2.set(base_distances);
    }

    base->search(
            n,
            x,
            k_base,
            base_distances,
            base_labels,
            base_index_params);
    for (idx_t i = 0; i < n * k_base; i++)
        assert(base_labels[i] >= -1 && base_labels[i] < ntotal);

    // compute refined distances
    auto rf = dynamic_cast<const IndexFlat*>(refine_index);
    FAISS_THROW_IF_NOT(rf);

    rf->compute_distance_subset(n, x, k_base, base_distances, base_labels);

    if (base->is_cosine_) {
        for (idx_t i = 0; i < n * k_base; i++) {
            if (base_labels[i] >= 0) {
                base_distances[i] /= base->norms[base_labels[i]];
            }
        }
    }
    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin<float, idx_t> C;
        reorder_2_heaps<C>(
                n, k, labels, distances, k_base, base_labels, base_distances);
    } else {
        FAISS_THROW_MSG("Metric type not supported");
    }
}

void IndexScaNN::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(n == 1);  // currently knowhere will split nq to 1

    FAISS_THROW_IF_NOT(is_trained);

    const IndexScaNNSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IndexScaNNSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexScaNN params have incorrect type");
    }

    SearchParameters* base_index_params = 
        (params != nullptr) ? params->base_index_params : nullptr;


    auto base = dynamic_cast<const IndexIVFPQFastScan*>(base_index);
    FAISS_THROW_IF_NOT(base);

    IVFSearchParameters ivf_search_params;
    ivf_search_params.nprobe = base->nlist;
    // todo aguzhva: this is somewhat hacky
    ivf_search_params.sel = base_index_params->sel;
    base->range_search(n, x, radius, result, &ivf_search_params);

    // nothing to refine, directly return the result
    if (refine_index == nullptr) {
        return;
    }

    // compute refined distances
    auto rf = dynamic_cast<const IndexFlat*>(refine_index);
    FAISS_THROW_IF_NOT(rf);

    rf->compute_distance_subset(n, x, result->lims[1], result->distances, result->labels);

    idx_t current = 0;
    for (idx_t i = 0; i < result->lims[1]; ++i) {
        if (base->is_cosine_) {
            result->distances[i] /= base->norms[result->labels[i]];
        }
        if (metric_type == METRIC_L2) {
            if (result->distances[i] < radius) {
                result->distances[current] = result->distances[i];
                result->labels[current] = result->labels[i];
                current++;
            }

        } else if (metric_type == METRIC_INNER_PRODUCT) {
            if (result->distances[i] > radius) {
                result->distances[current] = result->distances[i];
                result->labels[current] = result->labels[i];
                current++;
            }
        } else {
            FAISS_THROW_MSG("Metric type not supported");
        }
    }
    result->lims[1] = current;
}

} // namespace faiss