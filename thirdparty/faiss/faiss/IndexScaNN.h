#pragma once

#include <faiss/Index.h>
#include <faiss/IndexRefine.h>

namespace faiss {

struct IndexScaNNSearchParameters : SearchParameters {
    size_t reorder_k = 1;
    SearchParameters* base_index_params = nullptr;  // non-owning

    virtual ~IndexScaNNSearchParameters() = default;
};

// todo aguzhva: deprecate this class and update IndexRefine properly
struct IndexScaNN : IndexRefineFlat {
    explicit IndexScaNN(Index* base_index);
    IndexScaNN(Index* base_index, const float* xb);

    IndexScaNN();

    int64_t size();

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss