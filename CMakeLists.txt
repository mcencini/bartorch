cmake_minimum_required(VERSION 3.18)
project(bartorch LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# What is built here
#
# One shared library, libbartorch, holding every BART translation unit from
# the submodule plus the handful of units that give BART its FFT, BLAS,
# LAPACK and in-memory I/O.  Nothing links against PyTorch or Python: the
# host reaches the library through the C ABI in csrc/include/bartorch.h.
#
# BART is written against GNU C.  The nested functions it uses become Blocks
# under clang, with the vendored runtime on platforms that lack one, which
# keeps the library loadable without an executable stack.
# ---------------------------------------------------------------------------

set(BART_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bart" CACHE PATH "BART source tree")
set(BART_SRC "${BART_ROOT}/src")

if(NOT EXISTS "${BART_SRC}/bart.c")
    message(FATAL_ERROR "BART sources not found at ${BART_SRC}; run `git submodule update --init`")
endif()

option(BARTORCH_OPENMP "Build BART with OpenMP" ON)
option(BARTORCH_CUDA "Build BART's CUDA kernels" OFF)
set(BARTORCH_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
    "Architectures to emit SASS for; PTX for the last one is added for newer cards")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# BART routes assert through its own error path; NDEBUG must not remove it.
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ---------------------------------------------------------------------------
# Compiler mode
# ---------------------------------------------------------------------------
set(BART_C_OPTIONS "")
set(BART_C_DEFS "")
set(BART_LINK_OPTIONS "")
set(BART_EXTRA_LIBS "")

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    list(APPEND BART_C_OPTIONS -fblocks)
    set(BARTORCH_NESTED "clang-blocks")
    if(NOT APPLE)
        add_library(blocksruntime STATIC
            third_party/blocksruntime/runtime.c
            third_party/blocksruntime/data.c)
        target_include_directories(blocksruntime PUBLIC third_party/blocksruntime)
        list(APPEND BART_EXTRA_LIBS blocksruntime)
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    # GCC turns BART's nested functions into trampolines.  On the stack those
    # need an executable stack, which glibc 2.41 refuses to dlopen; GCC 14
    # places them on the heap instead, so the library loads anywhere.
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 14)
        message(FATAL_ERROR
            "GCC ${CMAKE_C_COMPILER_VERSION} would need an executable stack for BART's nested "
            "functions.  Use GCC 14 or newer, which has -ftrampoline-impl=heap, or clang.")
    endif()
    list(APPEND BART_C_OPTIONS -ftrampoline-impl=heap
        -Wno-vla-parameter -Wno-nonnull -Wno-maybe-uninitialized
        -include assert.h)
    list(APPEND BART_LINK_OPTIONS -Wl,-z,noexecstack)
    set(BARTORCH_NESTED "gcc-heap-trampolines")
else()
    message(FATAL_ERROR "BART needs clang or GCC 14+; ${CMAKE_C_COMPILER_ID} cannot compile it")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# CUDA
#
# BART's kernels are compiled by nvcc and the runtime libraries are linked
# dynamically, so the wheel carries device code and nothing else: cudart,
# cuFFT, cuBLAS and cuSOLVER come from the nvidia wheels torch already
# depends on.
#
# CUDA_GET_CUDA_DEVICE_NUM makes BART ask the driver whether a pointer is on
# a device rather than consult its own allocation table, which is what lets a
# tensor the host allocated be passed straight in.
# ---------------------------------------------------------------------------
if(BARTORCH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND BART_C_DEFS USE_CUDA CUDA_GET_CUDA_DEVICE_NUM)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

    set(_arch "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _arch "${_a}-real")
    endforeach()
    list(GET BARTORCH_CUDA_ARCHITECTURES -1 _newest)
    list(APPEND _arch "${_newest}-virtual")
    set(BARTORCH_CUDA_ARCH_SPEC "${_arch}")

    message(STATUS "CUDA ${CUDAToolkit_VERSION}: architectures ${BARTORCH_CUDA_ARCH_SPEC}")
endif()

if(BARTORCH_OPENMP)
    find_package(OpenMP COMPONENTS C)
    if(OpenMP_C_FOUND)
        list(APPEND BART_EXTRA_LIBS OpenMP::OpenMP_C)
    endif()
endif()

# ---------------------------------------------------------------------------
# Files BART's Makefile generates: the version string and the tool table.
# They are written under the build tree so the submodule stays untouched.
# ---------------------------------------------------------------------------
set(BART_GEN "${CMAKE_BINARY_DIR}/bart_gen")
file(MAKE_DIRECTORY "${BART_GEN}/misc")

file(READ "${BART_ROOT}/version.txt" BART_VERSION)
string(STRIP "${BART_VERSION}" BART_VERSION)
file(WRITE "${BART_GEN}/misc/version.inc" "\"${BART_VERSION}\"\n")

file(STRINGS "${BART_ROOT}/Makefile" _tool_lines REGEX "^T(BASE|FLP|NUM|IO|RECO|CALIB|MRI|SIM|NN|MOTION)\\+=")
set(_mainlist "")
set(_all_tools "")
foreach(_cat BASE FLP NUM IO RECO CALIB MRI SIM NN MOTION)
    set(_tools "")
    foreach(_line ${_tool_lines})
        if(_line MATCHES "^T${_cat}\\+=(.*)$")
            string(REPLACE " " ";" _names "${CMAKE_MATCH_1}")
            foreach(_name ${_names})
                if(EXISTS "${BART_SRC}/${_name}.c")
                    list(APPEND _tools "${_name}")
                    list(APPEND _all_tools "${_name}")
                endif()
            endforeach()
        endif()
    endforeach()
    list(SORT _tools)
    string(REPLACE ";" ", " _joined "${_tools}")
    if(_joined)
        string(APPEND _mainlist "#define MAIN_${_cat} ${_joined}, ()\n")
    else()
        string(APPEND _mainlist "#define MAIN_${_cat} ()\n")
    endif()
endforeach()
list(REMOVE_DUPLICATES _all_tools)
list(SORT _all_tools)
string(REPLACE ";" ", " _joined "${_all_tools}")
string(PREPEND _mainlist "#define MAIN_LIST ${_joined}, ()\n")
file(WRITE "${BART_GEN}/mainlist.inc" "${_mainlist}")
list(LENGTH _all_tools _ntools)
message(STATUS "BART ${BART_VERSION}: ${_ntools} tools, nested functions via ${BARTORCH_NESTED}")

# ---------------------------------------------------------------------------
# Sources: every module directory and every tool, minus what needs a
# library this build does not provide, and minus misc/memcfl.c which
# csrc/memcfl.c replaces.
# ---------------------------------------------------------------------------
set(BART_SOURCES "")
file(GLOB _module_dirs LIST_DIRECTORIES true "${BART_SRC}/*")
foreach(_dir ${_module_dirs})
    if(IS_DIRECTORY "${_dir}")
        get_filename_component(_mod "${_dir}" NAME)
        if(NOT _mod MATCHES "^(ismrm|lapacke|win)$")
            file(GLOB _srcs "${_dir}/*.c")
            list(APPEND BART_SOURCES ${_srcs})
        endif()
    endif()
endforeach()
file(GLOB _tool_srcs "${BART_SRC}/*.c")
list(APPEND BART_SOURCES ${_tool_srcs})

if(BARTORCH_CUDA)
    foreach(_dir calib motion noncart num wavelet)
        file(GLOB _cu "${BART_SRC}/${_dir}/*.cu")
        list(APPEND BART_SOURCES ${_cu})
    endforeach()
endif()
list(REMOVE_ITEM BART_SOURCES
    "${BART_SRC}/main.c"
    "${BART_SRC}/bbox.c"
    "${BART_SRC}/ismrmrd.c"
    "${BART_SRC}/misc/memcfl.c")

set(BARTORCH_SOURCES
    csrc/api.c
    csrc/backend.c
    csrc/cblas_shim.c
    csrc/lapacke_shim.c
    csrc/cuda.c
    csrc/memcfl.c
    csrc/ops.c
    csrc/ref_blas.c
    csrc/fftw_pocketfft.cpp)

add_library(bartorch SHARED ${BART_SOURCES} ${BARTORCH_SOURCES})

target_include_directories(bartorch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/compat"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pocketfft"
    "${BART_SRC}"
    "${BART_GEN}"
    "${BART_GEN}/misc")

target_compile_definitions(bartorch PRIVATE
    USE_LOG_BACKEND
    REDEFINE_PRINTF_FOR_TRACE
    NO_PNG
    _GNU_SOURCE
    ${BART_C_DEFS}
    BARTORCH_BUILD_INFO="bart=${BART_VERSION},compiler=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION},nested=${BARTORCH_NESTED},openmp=${OpenMP_C_FOUND},cuda=${BARTORCH_CUDA}")

target_compile_options(bartorch PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${BART_C_OPTIONS}>
    # nvcc reaches the host compiler for the code around each kernel, and
    # -compress-all keeps the fatbinaries from dominating the wheel.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC;-Xfatbin=-compress-all>)

if(BARTORCH_CUDA)
    set_target_properties(bartorch PROPERTIES
        CUDA_ARCHITECTURES "${BARTORCH_CUDA_ARCH_SPEC}")
    target_link_libraries(bartorch PRIVATE
        CUDA::cudart CUDA::cufft CUDA::cublas)
    # The runtime libraries live beside torch in the nvidia wheels; torch
    # loads them before this library, and the rpath covers a bare import.
    set_property(TARGET bartorch APPEND PROPERTY INSTALL_RPATH
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cufft/lib"
        "$ORIGIN/../nvidia/cublas/lib"
        "$ORIGIN/../nvidia/cu12/lib"
        "$ORIGIN/../nvidia/cu13/lib")
endif()

target_link_options(bartorch PRIVATE ${BART_LINK_OPTIONS})

target_link_libraries(bartorch PRIVATE Threads::Threads ${BART_EXTRA_LIBS})

if(NOT WIN32)
    target_link_libraries(bartorch PRIVATE m)
endif()

set_target_properties(bartorch PROPERTIES
    OUTPUT_NAME bartorch
    PREFIX "lib")

install(TARGETS bartorch
    LIBRARY DESTINATION bartorch
    RUNTIME DESTINATION bartorch)
