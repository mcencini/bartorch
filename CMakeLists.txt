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
    # GCC compiles BART's nested functions to trampolines on the stack.  Calling
    # them without an executable stack relies on BART reading a trampoline
    # layout that GCC only emits for non-PIC executables; a shared library gets
    # the 64-bit form and the check fails at runtime.
    message(FATAL_ERROR "libbartorch must be built with clang (CC=clang CXX=clang++); GCC would need an executable stack for BART's nested functions")
else()
    message(FATAL_ERROR "BART needs clang; ${CMAKE_C_COMPILER_ID} cannot compile it")
endif()

find_package(Threads REQUIRED)

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
    BARTORCH_BUILD_INFO="bart=${BART_VERSION},compiler=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION},nested=${BARTORCH_NESTED},openmp=${OpenMP_C_FOUND}")

target_compile_options(bartorch PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${BART_C_OPTIONS}>)

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
