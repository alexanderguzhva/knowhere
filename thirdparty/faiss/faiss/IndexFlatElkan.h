// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <faiss/IndexFlat.h>

namespace faiss {

// This is a special modification of IndexFlat that uses
//   elkan algorithm for the search. It is slower that the
//   regular IndexFlat::search() implementation, but sometimes
//   the trained index produces better recall rate.
// This index is intended to be used in Knowhere's ivf.cc file ONLY!!!
//
// Elkan algo was introduced into Knowhere in #2178, #2180 and #2258. 
struct IndexFlatElkan : IndexFlat {
    explicit IndexFlatElkan(idx_t d, MetricType metric = METRIC_L2,
                       bool is_cosine = false);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;    
};

}