cmake_minimum_required (VERSION 3.0.2)

project (HashTest)

SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable (hash_test hash_test.c str_set.c token_map.c ht32.c ht64.c ht32rh.c ht64rh.c cmetrohash64.c)
add_executable (hash_test_32 hash_test.c str_set.c token_map.c ht32.c ht64.c ht32rh.c ht64rh.c PMurHash.c)
add_executable (hash_test_rh hash_test.c str_set.c token_map.c ht32.c ht64.c ht32rh.c ht64rh.c cmetrohash64.c)

target_compile_definitions(hash_test_32 PRIVATE
    -DHT_HASH_32)
target_compile_definitions(hash_test_rh PRIVATE
    -DSTR_SET_RH -DTOKEN_MAP_RH)

add_executable (load_test load_test.c ptr_set.c)
# robin hood hash table
add_executable (load_test_rh load_test.c ptr_set.c)

target_compile_definitions(load_test PRIVATE
    -DPTR_SET_INT_HASH)
target_compile_definitions(load_test_rh PRIVATE
    -DPTR_SET_RH -DPTR_SET_INT_HASH)

# default hash function
add_executable (load_test_d load_test.c ptr_set.c cmetrohash64.c)
add_executable (load_test_d_rh load_test.c ptr_set.c cmetrohash64.c)
target_compile_definitions(load_test_rh PRIVATE
    -DPTR_SET_RH)

add_test(hash_test hash_test)
add_test(hash_test_32 hash_test_32)
add_test(hash_test_rh hash_test_rh)
add_test(load_test load_test)
add_test(load_test_rh load_test_rh)

enable_testing()
