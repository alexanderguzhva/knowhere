/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstddef>
#include <tuple>

#include <faiss/utils/distances.h>
#include "simd/hook.h"

namespace faiss {

/*********************************************************
 * Facilities that are used for batch distance computation
 *   for the case of a presence of a condition for the 
 *   acceptable elements.
 *********************************************************/

namespace {

constexpr size_t DEFAULT_BUFFER_SIZE = 8;

// Checks groups of BUFFER_SIZE elements and process acceptable
//   ones in groups of N. Process leftovers elements one by one. 
// This can be rewritten using std::range once an appropriate 
//   C++ standard is used.
// Concept constraints may be added once an appropriate 
//   C++ standard is used.
template<
    // A predicate for filtering elements. 
    //   bool Pred(const size_t idx);
    typename Pred, 
    // process 1 element. 
    //   void Process1(const size_t idx);
    typename Process1, 
    // process N elements. 
    //   void ProcessN(const std::array<size_t, N> ids);
    typename ProcessN,
    size_t N, 
    size_t BUFFER_SIZE>
void buffered_if(
        const size_t ny,
        Pred pred,
        Process1 process1,
        ProcessN processN) {
    static_assert((BUFFER_SIZE % N) == 0);

    // // the most generic version of the following code that is
    // //   suitable for the debugging is the following:
    //
    // for (size_t j = 0; j < ny; j++) {
    //     if (pred(j)) {
    //         process1(j);
    //     }
    // }
    //
    // return;

    // todo: maybe add a special case "ny < N" right here

    const size_t ny_buffer_size = (ny / BUFFER_SIZE) * BUFFER_SIZE;
    size_t saved_j[2 * BUFFER_SIZE + N];
    size_t counter = 0;
    size_t prev_counter = 0;
    
    for (size_t j = 0; j < ny_buffer_size; j += BUFFER_SIZE) {
        for (size_t jj = 0; jj < BUFFER_SIZE; jj++) {
            const bool is_acceptable = pred(j + jj);
            saved_j[counter] = j + jj; counter += is_acceptable ? 1 : 0;
        }

        if (counter >= N) {
            const size_t counter_n = (counter / N) * N;
            for (size_t i_counter = 0; i_counter < counter_n; i_counter += N) {
                std::array<size_t, N> tmp;
                std::copy(saved_j + i_counter, saved_j + i_counter + N, tmp.begin());
                
                processN(tmp);
            }

            // copy leftovers to the beginning of the buffer.
            // todo: use ring buffer instead, maybe?
            // for (size_t jk = counter_n; jk < counter; jk++) {
            //     saved_j[jk - counter_n] = saved_j[jk];
            // }
            for (size_t jk = counter_n; jk < counter_n + N; jk++) {
                saved_j[jk - counter_n] = saved_j[jk];
            }

            // rewind
            counter -= counter_n;
        }
    }

    for (size_t j = ny_buffer_size; j < ny; j++) {
        const bool is_acceptable = pred(j);
        saved_j[counter] = j; counter += is_acceptable ? 1 : 0;
    }

    // process leftovers
    for (size_t jj = 0; jj < counter; jj++) {
        const size_t j = saved_j[jj];
        process1(j);
    }
}

} // namespace

template<
    // A predicate for filtering elements. 
    //   bool Pred(const size_t idx);
    typename Pred, 
    // Compute distance from a query vector to 1 element.
    //   float Distance1(const size_t idx);
    typename Distance1, 
    // Compute distance from a query vector to N elements
    //   void DistanceN(
    //      const std::array<size_t, N> idx,
    //      std::array<float, N>& dis);
    typename DistanceN,
    // Apply an element.
    //   void Apply(const float dis, const size_t idx);
    typename Apply,
    size_t N,
    size_t BUFFER_SIZE>
void fvec_distance_ny_if(
        const size_t ny,
        Pred pred,
        Distance1 distance1,
        DistanceN distanceN,
        Apply apply
) {
    // process 1 element
    auto process1 = [&](const size_t idx) {
        const float distance = distance1(idx);
        apply(distance, idx);
    };

    // process N elements
    auto processN = [&](const std::array<size_t, N> indices) {
        std::array<float, N> dis;
        distanceN(indices, dis);

        #pragma unroll
        for (size_t i = 0; i < N; i++) {
            apply(dis[i], indices[i]);
        }
    };

    // process
    buffered_if<Pred, decltype(process1), decltype(processN), N, BUFFER_SIZE>(
        ny,
        pred,
        process1,
        processN
    );
}

// compute ny inner product between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   bool Pred(const size_t idx);
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const size_t idx);
    typename Apply>
void fvec_inner_products_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {    
    // compute a distance from the query to 1 element
    auto distance1 = [x, y, d](const size_t idx) { 
        return fvec_inner_product(x, y + idx * d, d); 
    };

    // compute distances from the query to 4 elements
    auto distance4 = [x, y, d](const std::array<size_t, 4> indices, std::array<float, 4>& dis) { 
        fvec_inner_product_batch_4(
            x,
            y + indices[0] * d,
            y + indices[1] * d,
            y + indices[2] * d,
            y + indices[3] * d,
            d,
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        apply
    );
}

// compute ny square L2 distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   bool Pred(const size_t idx);
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const size_t idx);
    typename Apply>
void fvec_L2sqr_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {    
    // compute a distance from the query to 1 element
    auto distance1 = [x, y, d](const size_t idx) { 
        return fvec_L2sqr(x, y + idx * d, d); 
    };

    // compute distances from the query to 4 elements
    auto distance4 = [x, y, d](const std::array<size_t, 4> indices, std::array<float, 4>& dis) { 
        fvec_L2sqr_batch_4(
            x,
            y + indices[0] * d,
            y + indices[1] * d,
            y + indices[2] * d,
            y + indices[3] * d,
            d,
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        apply
    );
}

} //namespace faiss

