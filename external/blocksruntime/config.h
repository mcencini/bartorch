/* Minimal configuration for the vendored BlocksRuntime: GCC/Clang
 * __sync builtins on every non-Apple target.  Apple platforms use the
 * runtime shipped in libSystem and never compile this directory. */
#ifndef BARTORCH_BLOCKSRUNTIME_CONFIG_H
#define BARTORCH_BLOCKSRUNTIME_CONFIG_H
#define HAVE_SYNC_BOOL_COMPARE_AND_SWAP_INT 1
#define HAVE_SYNC_BOOL_COMPARE_AND_SWAP_LONG 1
#endif
