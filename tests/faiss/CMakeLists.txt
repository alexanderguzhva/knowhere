set(FAISS_TEST_SRCS
    ../../thirdparty/faiss/tests/test_omp_threads.cpp)

find_package(GTest REQUIRED)

if(__X86_64)
  add_executable(faiss_tests ${FAISS_TEST_SRCS})

  add_dependencies(faiss_tests faiss)
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
  target_link_libraries(
    faiss_tests PUBLIC faiss GTest::gtest GTest::gtest_main)
  target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)
endif()

if(__AARCH64)
  add_executable(faiss_tests ${FAISS_TEST_SRCS})
  target_compile_options(
    faiss_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  add_dependencies(faiss_tests faiss)
  target_link_libraries(faiss_tests faiss GTest::gtest GTest::gtest_main)
  target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)
endif()
