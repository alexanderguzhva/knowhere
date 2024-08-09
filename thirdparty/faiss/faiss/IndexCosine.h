// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// knowhere-specific indices

#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>


namespace faiss {

// a distance computer wrapper that normalizes the distance over a query
struct WithCosineNormDistanceComputer : DistanceComputer {
    /// owned by this
    std::unique_ptr<DistanceComputer> basedis;
    // not owned by this
    const float* inverse_l2_norms = nullptr;
    // computed internally
    float inverse_query_norm = 0;
    // cached dimensionality
    int d = 0;

    // initialize in a custom way
    WithCosineNormDistanceComputer(
        const float* inverse_l2_norms_, 
        const int d_,
        std::unique_ptr<DistanceComputer>&& basedis_);

    // the query remains untouched. It is a caller's responsibility
    //   to normalize it.
    void set_query(const float* x) override;

    /// compute distance of vector i to current query
    float operator()(idx_t i) override;

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override;

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override;
};

struct HasInverseL2Norms {
    virtual ~HasInverseL2Norms() = default;

    virtual const float* get_inverse_l2_norms() const = 0;
};

// a supporting storage for L2 norms
struct L2NormsStorage {
    std::vector<float> inverse_l2_norms;

    // create from a vector of L2 norms (sqrt(sum(x^2)))
    static L2NormsStorage from_l2_norms(const std::vector<float>& l2_norms);

    // add vectors
    void add(const float* x, const idx_t n, const idx_t d);

    // add L2 norms (sqrt(sum(x^2)))
    void add_l2_norms(const float* l2_norms, const idx_t n);

    // clear the storage
    void reset();

    // produces a vector of L2 norms, effectively inverting inverse_l2_norms
    std::vector<float> as_l2_norms() const;
};

// A dedicated index used for Cosine Distance in the future.
struct IndexFlatCosine : IndexFlat, HasInverseL2Norms {
    L2NormsStorage inverse_norms_storage;

    IndexFlatCosine();
    IndexFlatCosine(idx_t d);

    void add(idx_t n, const float* x) override;
    void reset() override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    const float* get_inverse_l2_norms() const override;
};

//
struct IndexHNSWFlatCosine : IndexHNSW {
    IndexHNSWFlatCosine();
    IndexHNSWFlatCosine(int d, int M);
};



}
