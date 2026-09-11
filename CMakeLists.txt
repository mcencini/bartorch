cmake_minimum_required(VERSION 3.18)
project(bartorch LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# What is built here
#
# One shared library, libbartorch, holding every BART translation unit from
# the submodule plus the handful of units that give BART its FFT, BLAS,
# LAPACK and in-memory I/O.  Nothing links against PyTorch or Python: the
# host reaches the library through the C ABI in src/csrc/include/bartorch.h.
#
# BART is written against GNU C.  The nested functions it uses become Blocks
# under clang, with the vendored runtime on platforms that lack one, which
# keeps the library loadable without an executable stack.
# ---------------------------------------------------------------------------

set(BART_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/bart" CACHE PATH "BART source tree")
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

# BART checks its arguments with assert and src/csrc/abi/api.c puts those on its error
# path, so NDEBUG must not remove them.
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
            external/blocksruntime/runtime.c
            external/blocksruntime/data.c)
        target_include_directories(blocksruntime PUBLIC external/blocksruntime)
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

# The library exports a C ABI and exchanges no C++ objects with the process it
# is loaded into, so its GCC runtime can be its own.  Linked dynamically it
# would carry the toolchain's floor into every target system -- a GCC 14 build
# asks for GCC_14.0.0 from libgcc_s, which no released distribution has -- and
# that is a load-time failure, not a fallback.  Static leaves libc and libgomp.
if(NOT APPLE AND NOT WIN32)
    list(APPEND BART_LINK_OPTIONS -static-libgcc -static-libstdc++)
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
# src/csrc/abi/memcfl.c replaces.
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

# Three things, and a file belongs to whichever it is.  `abi/` is the boundary:
# what the host calls, and what BART's environment asks of the host in return.
# `ops/` is what the host builds and drives -- the operators and the solve.
# `substitute/` is what runs in BART's place: its transforms, and the libraries
# it would otherwise have been linked against.
set(BARTORCH_SOURCES
    src/csrc/abi/api.c
    src/csrc/abi/cuda.c
    src/csrc/abi/host_reads.c
    src/csrc/abi/memcfl.c

    src/csrc/ops/ops.c
    src/csrc/ops/sense.c
    src/csrc/ops/iter.c

    src/csrc/substitute/backend.c
    src/csrc/substitute/ref_blas.c
    src/csrc/substitute/cblas_shim.c
    src/csrc/substitute/lapacke_shim.c
    src/csrc/substitute/fft.cpp
    src/csrc/substitute/finufft.c
    src/csrc/substitute/nufft_finufft.c
    src/csrc/substitute/psf.c)

if(BARTORCH_CUDA)
    list(APPEND BARTORCH_SOURCES src/csrc/ops/kernels.cu src/csrc/ops/fft_callbacks.cu)

    # The passes around a volume's transform, which cuFFT links into the
    # transform when a plan is made (src/csrc/ops/fft_callbacks.cu).  It takes them as
    # LTO-IR at run time, so src/csrc/ops/fft_callbacks_lto.cu is compiled to that
    # alone and embedded in the library as an array.
    set(_lto_src "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ops/fft_callbacks_lto.cu")
    set(_lto_bin "${CMAKE_CURRENT_BINARY_DIR}/fft_callbacks_lto.fatbin")
    set(_lto_hdr "${CMAKE_CURRENT_BINARY_DIR}/fft_callbacks_lto.h")
    set(_lto_arch "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _lto_arch "-gencode=arch=compute_${_a},code=lto_${_a}")
    endforeach()
    set(_lto_ccbin "")
    if(CMAKE_CUDA_HOST_COMPILER)
        set(_lto_ccbin -ccbin "${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    add_custom_command(
        OUTPUT "${_lto_hdr}"
        COMMAND "${CMAKE_CUDA_COMPILER}" ${_lto_ccbin} -std=c++17 -dc -fatbin ${_lto_arch}
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc" -o "${_lto_bin}" "${_lto_src}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${_lto_bin}" "-DOUT=${_lto_hdr}"
                -DNAME=bartorch_fft_callbacks_lto
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        DEPENDS "${_lto_src}" "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/ops/coset.cuh"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
        VERBATIM)
    list(APPEND BARTORCH_SOURCES "${_lto_hdr}")
endif()

# The paired Toeplitz kernels (src/csrc/ops/paired.cu) are cuFFTDx, which is headers
# only and comes with MathDx: BARTORCH_MATHDX_DIR names its nvidia/mathdx
# directory -- `pip install nvidia-mathdx` puts one in site-packages -- and
# the kernels are compiled for the cubic grid sizes in BARTORCH_PAIRED_SIZES,
# tuned for the newest architecture listed.  The default is every size from
# 192 to 512 that cuFFTDx transforms at cuFFT's speed or near it, plus 32 for
# the tests; 340, whose factor 17 cuFFTDx takes through Bluestein's algorithm
# at 2.6 times cuFFT's time, is left to cuFFT.  Without it the library has no
# paired kernels and every size is served by the transforms it has otherwise.
set(BARTORCH_MATHDX_DIR "" CACHE PATH "MathDx's nvidia/mathdx directory, for the paired Toeplitz kernels")
set(BARTORCH_PAIRED_SIZES "32;192;200;224;240;256;288;300;320;360;384;400;448;480;512" CACHE STRING "Cubic grid sizes the paired Toeplitz kernels are compiled for")
if(BARTORCH_CUDA AND BARTORCH_MATHDX_DIR)
    if(NOT EXISTS "${BARTORCH_MATHDX_DIR}/include/cufftdx.hpp")
        message(FATAL_ERROR "BARTORCH_MATHDX_DIR=${BARTORCH_MATHDX_DIR} has no include/cufftdx.hpp")
    endif()
    set(_sizes "")
    foreach(_n ${BARTORCH_PAIRED_SIZES})
        string(APPEND _sizes " X(${_n})")
    endforeach()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h.in"
        "/* Generated from BARTORCH_PAIRED_SIZES; see CMakeLists.txt. */\n#define BARTORCH_PAIRED_SIZES(X)${_sizes}\n")
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h.in" "${CMAKE_CURRENT_BINARY_DIR}/paired_sizes.h" COPYONLY)
    list(GET BARTORCH_CUDA_ARCHITECTURES -1 _paired_arch)
    set(_paired_archs "")
    foreach(_a ${BARTORCH_CUDA_ARCHITECTURES})
        list(APPEND _paired_archs "${_a}-real")
    endforeach()
    # Compiled on their own and for the listed architectures only: PTX of a
    # template instantiated once per grid size is megabytes, and a card newer
    # than the list is served by the transforms the library has otherwise.
    add_library(bartorch_paired OBJECT src/csrc/ops/paired.cu)
    set_target_properties(bartorch_paired PROPERTIES
        CUDA_ARCHITECTURES "${_paired_archs}"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(bartorch_paired PRIVATE
        "${BARTORCH_MATHDX_DIR}/include"
        "${BARTORCH_MATHDX_DIR}/external/cutlass/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc"
        "${BART_SRC}"
        "${BART_GEN}"
        "${BART_GEN}/misc"
        "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(bartorch_paired PRIVATE ${BART_C_DEFS} _GNU_SOURCE
        BARTORCH_PAIRED_SM=${_paired_arch}0)
    target_compile_options(bartorch_paired PRIVATE -Xcompiler=-fPIC -Xfatbin=-compress-all)
    list(APPEND BART_C_DEFS BARTORCH_PAIRED)
    message(STATUS "Paired Toeplitz kernels: cuFFTDx from ${BARTORCH_MATHDX_DIR}, sizes ${BARTORCH_PAIRED_SIZES}")
endif()

add_library(bartorch SHARED ${BART_SOURCES} ${BARTORCH_SOURCES})

if(TARGET bartorch_paired)
    target_sources(bartorch PRIVATE $<TARGET_OBJECTS:bartorch_paired>)
endif()

# BART's NUFFT operator keeps its implementation under another name, so
# src/csrc/substitute/nufft_finufft.c can answer with FINUFFT's transform and still hand back
# anything it does not serve.  The rename applies to these files, not to their
# callers, so every tool that builds a NUFFT goes through the substitution and
# nufft.c's own internal uses stay with BART's operator.
set_source_files_properties("${BART_SRC}/noncart/nufft.c" PROPERTIES
    COMPILE_DEFINITIONS "nufft_create=bart_nufft_create;nufft_create2=bart_nufft_create2;nufft_get_psf_dims=bart_nufft_get_psf_dims;nufft_get_psf=bart_nufft_get_psf;nufft_get_psf2=bart_nufft_get_psf2;nufft_update_psf=bart_nufft_update_psf;nufft_update_psf2=bart_nufft_update_psf2;nufft_update_traj=bart_nufft_update_traj;compute_psf=bart_compute_psf;compute_psf2=bart_compute_psf2;compute_psf2_decomposed=bart_compute_psf2_decomposed")

# The SENSE operators keep their implementations under another name, so
# src/csrc/ops/sense.c can walk the coils a slab at a time and still hand back BART's
# own chain for an arrangement it cannot serve.
set_source_files_properties("${BART_SRC}/sense/modelnc.c" PROPERTIES
    COMPILE_DEFINITIONS "sense_nc_init=bart_sense_nc_init")

set_source_files_properties("${BART_SRC}/sense/model.c" PROPERTIES
    COMPILE_DEFINITIONS "sense_init=bart_sense_init")

set_source_files_properties("${BART_SRC}/noncart/precond.c" PROPERTIES
    COMPILE_DEFINITIONS "nufft_precond_create=bart_nufft_precond_create")

# The few entry points BART reads element by element rather than through md_
# operations; src/csrc/abi/host_reads.c answers them over a host copy when a tool is
# handed memory on a card.
set_source_files_properties("${BART_SRC}/misc/mri2.c" PROPERTIES
    COMPILE_DEFINITIONS "estimate_im_dims=bart_estimate_im_dims;estimate_fast_sq_im_dims=bart_estimate_fast_sq_im_dims")

set_source_files_properties("${BART_SRC}/sense/optcom.c" PROPERTIES
    COMPILE_DEFINITIONS "estimate_scaling_norm=bart_estimate_scaling_norm")

target_include_directories(bartorch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/compat"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/pocketfft"
    "${BART_SRC}"
    "${BART_GEN}"
    "${BART_GEN}/misc"
    "${CMAKE_CURRENT_BINARY_DIR}")

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
        CUDA::cudart CUDA::cufft CUDA::cublas ${CMAKE_DL_LIBS})
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
