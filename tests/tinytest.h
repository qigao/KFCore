/*
 * tinytest.h - Header-only BDD/TDD testing framework for C and C++
 *
 * Version: 1.0.0
 * License: MIT
 *
 * A single-header testing library providing:
 *   - BDD syntax:  spec/describe/it/before/after/before_each/after_each
 *   - TDD syntax:  suite/section/it/check macros
 *   - Typed assertions: check_int_eq, check_str_eq, check_float_eq, etc.
 *   - C++ templates: check_equal<T>, check_not_equal<T>, check_greater<T>, check_less<T>
 *   - Exception testing: check_throws, check_throws_as, check_nothrow, etc.
 *   - Benchmarking: benchmark("title", N, scale) { code; }
 *   - Output formats: colored console, TAP, JUnit XML
 *   - Test filtering: --filter, --list, focus (fit/it_only), skip (xit)
 *
 * Usage:
 *   #include "tinytest.h"
 *
 *   spec("my module") {
 *       describe("feature") {
 *           it("should work") {
 *               check(1 + 1 == 2);
 *           }
 *       }
 *   }
 *
 * Define TINYTEST_NO_MAIN before including this header to suppress the
 * built-in main() function (useful for custom test runners).
 */

#ifndef TINYTEST_H
#define TINYTEST_H

/* C++ compatibility */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #include <stdio.h>
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <Windows.h>
  #include <direct.h>
  #include <io.h>
  #include <sys/stat.h>
  #define __BDD_IS_ATTY__() _isatty(_fileno(stdout))
#else
  #ifndef _POSIX_C_SOURCE
    /* This definition is required for `fileno` to be defined */
    #define _POSIX_C_SOURCE 200809L
  #endif
  #include <dirent.h>
  #include <stdio.h>
  #include <sys/stat.h>
  #include <unistd.h>
  /* term.h may not be available on all systems */
  #ifdef __has_include
    #if __has_include(<term.h>)
      #include <term.h>
    #endif
  #endif
  #define __BDD_IS_ATTY__() isatty(fileno(stdout))
#endif

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* C++ has bool built-in, C needs stdbool.h */
#ifndef __cplusplus
  #include <stdbool.h>
#endif

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4130)
  #pragma warning(disable : 4611) /* setjmp/longjmp in test control flow */
  #pragma warning(disable : 4996) /* _CRT_SECURE_NO_WARNINGS */
  #pragma warning(                                                                                 \
      disable : 4127) /* conditional expression is constant (check macros with constant args) */
#endif

#ifndef BDD_USE_COLOR
  #define BDD_USE_COLOR 1
#endif

#ifndef BDD_USE_TAP
  #define BDD_USE_TAP 0
#endif

#ifndef BDD_BENCH_NAME_WIDTH
  #define BDD_BENCH_NAME_WIDTH 32
#endif

#ifndef BDD_BENCH_TABLE
  #define BDD_BENCH_TABLE 1
#endif

#ifndef BDD_BENCH_COLLECT
  #define BDD_BENCH_COLLECT 1
#endif

#ifndef BDD_BENCH_MAX
  #define BDD_BENCH_MAX 64
#endif

#if defined(__clang__)
  #define __BDD_NO_SANITIZE_ADDRESS__ __attribute__((no_sanitize("address")))
#elif defined(__GNUC__)
  #define __BDD_NO_SANITIZE_ADDRESS__ __attribute__((no_sanitize_address))
#else
  #define __BDD_NO_SANITIZE_ADDRESS__
#endif

/* Cast macros to avoid -Wold-style-cast in C++ */
#ifdef __cplusplus
  #define __BDD_CAST(type, expr) static_cast<type>(expr)
  #define __BDD_REINTERPRET_CAST(type, expr) reinterpret_cast<type>(expr)
  #define __BDD_CONST_CAST(type, expr) const_cast<type>(expr)
#else
  #define __BDD_CAST(type, expr) ((type)(expr))
  #define __BDD_REINTERPRET_CAST(type, expr) ((type)(expr))
  #define __BDD_CONST_CAST(type, expr) ((type)(expr))
#endif

#define __BDD_COLOR_RESET__ "\x1B[0m"
#define __BDD_COLOR_RED__ "\x1B[31m"
#define __BDD_COLOR_GREEN__ "\x1B[32m"
#define __BDD_COLOR_YELLOW__ "\x1B[33m"
#define __BDD_COLOR_BOLD__ "\x1B[1m"
#define __BDD_COLOR_MAGENTA__ "\x1B[35m"

#ifndef __BDD_TLS
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
    #define __BDD_TLS _Thread_local
  #elif defined(_MSC_VER)
    #define __BDD_TLS __declspec(thread)
  #elif defined(__GNUC__) || defined(__clang__)
    #define __BDD_TLS __thread
  #else
    #define __BDD_TLS
  #endif
#endif

/* Cross-TU shared globals for header-only library */
#if defined(_MSC_VER)
  #define __BDD_SELECTANY __declspec(selectany)
#else
  #define __BDD_SELECTANY __attribute__((weak))
#endif

/* Cross-platform high-resolution timer */
static inline double __bdd_get_time_ms__(void) {
#ifdef _WIN32
  LARGE_INTEGER frequency, counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return __BDD_CAST(double, counter.QuadPart) * 1000.0 / __BDD_CAST(double, frequency.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return __BDD_CAST(double, ts.tv_sec) * 1000.0 + __BDD_CAST(double, ts.tv_nsec) / 1000000.0;
#endif
}

static void __bdd_indent__(FILE *fp, size_t level);

typedef struct __bdd_config_type__ __bdd_config_type__;
typedef void (*__bdd_spec_fn__)(__bdd_config_type__ *__bdd_config__);

typedef struct __bdd_spec_entry__ {
  const char *name;
  __bdd_spec_fn__ fn;
  struct __bdd_spec_entry__ *next;
} __bdd_spec_entry__;

__BDD_SELECTANY __bdd_spec_entry__ *__bdd_specs__ = NULL;
__BDD_SELECTANY size_t __bdd_spec_count__ = 0;

static void __bdd_register_spec__(const char *name, __bdd_spec_fn__ fn) {
  __bdd_spec_entry__ *e = __BDD_CAST(__bdd_spec_entry__ *, malloc(sizeof(__bdd_spec_entry__)));
  if (!e) {
    perror("malloc(spec)");
    abort();
  }
  e->name = name;
  e->fn = fn;
  e->next = __bdd_specs__;
  __bdd_specs__ = e;
  __bdd_spec_count__++;
}

static __bdd_spec_entry__ *__bdd_get_spec_entry__(size_t index) {
  __bdd_spec_entry__ *e = __bdd_specs__;
  for (size_t i = 0; i < index && e; ++i) {
    e = e->next;
  }
  return e;
}

static void __bdd_cleanup_specs__(void) {
  __bdd_spec_entry__ *e = __bdd_specs__;
  while (e) {
    __bdd_spec_entry__ *next = e->next;
    free(e);
    e = next;
  }
  __bdd_specs__ = NULL;
  __bdd_spec_count__ = 0;
}

#if defined(__cplusplus)
  #define __BDD_CONSTRUCTOR__(fn)                                                                  \
    static void fn(void);                                                                          \
    namespace {                                                                                    \
      struct __BDD_CAT2(__bdd_ctor_struct_, fn) {                                                  \
        __BDD_CAT2(__bdd_ctor_struct_, fn)() { fn(); }                                             \
      };                                                                                           \
      static __BDD_CAT2(__bdd_ctor_struct_, fn) __BDD_CAT2(__bdd_ctor_obj_, fn);                   \
    }                                                                                              \
    static void fn(void)
#elif defined(_WIN32) && defined(__clang__)
  #ifdef read
    #pragma push_macro("read")
    #undef read
    #define __BDD_POP_READ__ 1
  #endif
  #pragma section(".CRT$XCU", read)
  #ifdef __BDD_POP_READ__
    #pragma pop_macro("read")
    #undef __BDD_POP_READ__
  #endif
typedef void(__cdecl *__bdd_ctor_fn__)(void);
  #define __BDD_CONSTRUCTOR__(fn)                                                                  \
    static void __cdecl fn(void);                                                                  \
    __declspec(allocate(".CRT$XCU")) __attribute__((used)) static __bdd_ctor_fn__                  \
        __BDD_CAT2(__bdd_ctor_, fn) = fn;                                                          \
    static void __cdecl fn(void)
#elif defined(_MSC_VER)
  #ifdef read
    #pragma push_macro("read")
    #undef read
    #define __BDD_POP_READ__ 1
  #endif
  #pragma section(".CRT$XCU", read)
  #ifdef __BDD_POP_READ__
    #pragma pop_macro("read")
    #undef __BDD_POP_READ__
  #endif
typedef void(__cdecl *__bdd_ctor_fn__)(void);
  #define __BDD_CONSTRUCTOR__(fn)                                                                  \
    static void __cdecl fn(void);                                                                  \
    __declspec(allocate(".CRT$XCU")) static __bdd_ctor_fn__ __BDD_CAT2(__bdd_ctor_, fn) = fn;      \
    static void __cdecl fn(void)
#elif defined(__GNUC__) || defined(__clang__)
  #define __BDD_CONSTRUCTOR__(fn)                                                                  \
    static void fn(void) __attribute__((constructor));                                             \
    static void fn(void)
#else
  #define __BDD_CONSTRUCTOR__(fn) static void fn(void)
#endif

static inline void __bdd_bench_print_header__(__bdd_config_type__ *config, size_t level);

static inline void __bdd_bench_print__(__bdd_config_type__ *config, const char *title,
                                       size_t iters, double sum_ms,
                                       double min_ms, double max_ms, double scale,
                                       size_t level, bool use_color);

static inline void __bdd_bench_reset__(__bdd_config_type__ *config);
static inline void __bdd_bench_add__(__bdd_config_type__ *config, const char *title, size_t iters,
                                     double sum_ms, double min_ms, double max_ms, double scale);
static inline void __bdd_bench_flush__(__bdd_config_type__ *config, size_t level, bool use_color);

/* Simple file helpers (cross-platform) */
static inline char *tt_temp_dir(void) {
#ifdef _WIN32
  static char path[MAX_PATH];
  DWORD n = GetTempPathA((DWORD)sizeof(path), path);
  if (n == 0 || n >= sizeof(path)) {
    const char *env = getenv("TEMP");
    if (!env) env = getenv("TMP");
    if (!env) env = ".";
    char *res = __BDD_CAST(char *, malloc(strlen(env) + 1));
    if (res) strcpy(res, env);
    return res;
  }
  char *res = __BDD_CAST(char *, malloc(strlen(path) + 1));
  if (res) strcpy(res, path);
  return res;
#else
  const char *tmp = getenv("TMPDIR");
  if (!tmp) tmp = getenv("TMP");
  if (!tmp) tmp = getenv("TEMP");
  if (!tmp) tmp = "/tmp";
  char *res = __BDD_CAST(char *, malloc(strlen(tmp) + 1));
  if (res) strcpy(res, tmp);
  return res;
#endif
}

static inline char *tt_read_file(const char *path, size_t *out_size) {
  FILE *fp = fopen(path, "rb");
  long size;
  char *buf;
  if (!fp) return NULL;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  size = ftell(fp);
  if (size < 0) {
    fclose(fp);
    return NULL;
  }
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return NULL;
  }
  buf = __BDD_CAST(char *, malloc((size_t)size + 1));
  if (!buf) {
    fclose(fp);
    return NULL;
  }
  if (size > 0 && fread(buf, 1, (size_t)size, fp) != (size_t)size) {
    fclose(fp);
    free(buf);
    return NULL;
  }
  fclose(fp);
  buf[size] = '\0';
  if (out_size) *out_size = (size_t)size;
  return buf;
}

static inline int tt_write_file(const char *path, const void *data, size_t size) {
  FILE *fp = fopen(path, "wb");
  if (!fp) return -1;
  if (size > 0 && fwrite(data, 1, size, fp) != size) {
    fclose(fp);
    return -1;
  }
  fclose(fp);
  return 0;
}

static inline int tt_remove_file(const char *path) {
#ifdef _WIN32
  DWORD attrs = GetFileAttributesA(path);
  if (attrs != INVALID_FILE_ATTRIBUTES &&
      (attrs & FILE_ATTRIBUTE_REPARSE_POINT) &&
      (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
    return RemoveDirectoryA(path) ? 0 : -1;
  }
  return DeleteFileA(path) ? 0 : -1;
#else
  return unlink(path) == 0 ? 0 : -1;
#endif
}

static inline int tt_make_dir(const char *path) {
#ifdef _WIN32
  return _mkdir(path) == 0 ? 0 : -1;
#else
  return mkdir(path, 0777) == 0 ? 0 : -1;
#endif
}

static inline int tt_is_dir(const char *path) {
#ifdef _WIN32
  DWORD attrs = GetFileAttributesA(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) return 0;
  return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
         (attrs & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
#else
  struct stat st;
  if (lstat(path, &st) != 0) return 0;
  return S_ISDIR(st.st_mode) ? 1 : 0;
#endif
}

static inline char *tt_make_temp_file(const char *prefix, const char *suffix) {
  const char *pre = prefix ? prefix : "tt";
  const char *suf = suffix ? suffix : "";
#ifdef _WIN32
  char base[MAX_PATH];
  char file[MAX_PATH];
  if (GetTempPathA((DWORD)sizeof(base), base) == 0) return NULL;
  if (GetTempFileNameA(base, pre, 0, file) == 0) return NULL;
  if (suf[0]) {
    char renamed[MAX_PATH];
    snprintf(renamed, sizeof(renamed), "%s%s", file, suf);
    if (MoveFileExA(file, renamed, MOVEFILE_REPLACE_EXISTING)) {
      strncpy(file, renamed, sizeof(file) - 1);
      file[sizeof(file) - 1] = '\0';
    }
  }
  return _strdup(file);
#else
  char *dir = tt_temp_dir();
  if (!dir) return NULL;
  char tmpl[512];
  int fd;
  snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", dir, pre);
  fd = mkstemp(tmpl);
  if (fd < 0) {
    free(dir);
    return NULL;
  }
  close(fd);
  if (suf[0]) {
    char renamed[512];
    snprintf(renamed, sizeof(renamed), "%s%s", tmpl, suf);
    if (rename(tmpl, renamed) == 0) {
      free(dir);
      return strdup(renamed);
    }
  }
  free(dir);
  return strdup(tmpl);
#endif
}

static inline char *tt_make_temp_dir(const char *prefix) {
  const char *pre = prefix ? prefix : "tt";
#ifdef _WIN32
  char base[MAX_PATH];
  char file[MAX_PATH];
  if (GetTempPathA((DWORD)sizeof(base), base) == 0) return NULL;
  if (GetTempFileNameA(base, pre, 0, file) == 0) return NULL;
  DeleteFileA(file);
  if (CreateDirectoryA(file, NULL) == 0) return NULL;
  return _strdup(file);
#else
  char *dir = tt_temp_dir();
  if (!dir) return NULL;
  char tmpl[512];
  snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", dir, pre);
  if (!mkdtemp(tmpl)) {
    free(dir);
    return NULL;
  }
  free(dir);
  return strdup(tmpl);
#endif
}

static inline int tt_remove_tree(const char *path) {
  if (!path || !path[0]) return -1;
  if (!tt_is_dir(path)) return tt_remove_file(path);
#ifdef _WIN32
  char pattern[MAX_PATH];
  WIN32_FIND_DATAA fdata;
  HANDLE h;
  snprintf(pattern, sizeof(pattern), "%s\\*", path);
  h = FindFirstFileA(pattern, &fdata);
  if (h != INVALID_HANDLE_VALUE) {
    do {
      const char *name = fdata.cFileName;
      if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
      char child[MAX_PATH];
      snprintf(child, sizeof(child), "%s\\%s", path, name);
      if ((fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
          (fdata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0) {
        tt_remove_tree(child);
      } else {
        tt_remove_file(child);
      }
    } while (FindNextFileA(h, &fdata));
    FindClose(h);
  }
  return RemoveDirectoryA(path) ? 0 : -1;
#else
  DIR *dir = opendir(path);
  if (dir) {
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
      const char *name = ent->d_name;
      if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
      char child[1024];
      snprintf(child, sizeof(child), "%s/%s", path, name);
      if (tt_is_dir(child)) {
        tt_remove_tree(child);
      } else {
        unlink(child);
      }
    }
    closedir(dir);
  }
  return rmdir(path) == 0 ? 0 : -1;
#endif
}

typedef struct __bdd_array__ {
  void **values;
  size_t capacity;
  size_t size;
} __bdd_array__;

static inline __bdd_array__ *__bdd_array_create__(void) {
  __bdd_array__ *arr = __BDD_CAST(__bdd_array__ *, malloc(sizeof(__bdd_array__)));
  if (!arr) {
    perror("malloc(array)");
    abort();
  }
  arr->capacity = 4;
  arr->size = 0;
  arr->values = __BDD_CAST(void **, calloc(arr->capacity, sizeof(void *)));
  if (!arr->values) {
    perror("calloc(array->values)");
    free(arr);
    abort();
  }
  return arr;
}

static inline void *__bdd_array_push__(__bdd_array__ *arr, void *item) {
  if (arr->size == arr->capacity) {
    arr->capacity *= 2;
    void **v = __BDD_CAST(void **, realloc(arr->values, sizeof(void *) * arr->capacity));
    if (!v) {
      perror("realloc(array)");
      abort();
    }
    arr->values = v;
  }
  arr->values[arr->size++] = item;
  return item;
}

static inline void *__bdd_array_last__(__bdd_array__ *arr) {
  if (arr->size == 0) {
    return NULL;
  }
  return arr->values[arr->size - 1];
}

static inline void *__bdd_array_pop__(__bdd_array__ *arr) {
  if (arr->size == 0) {
    return NULL;
  }
  void *result = arr->values[arr->size - 1];
  --arr->size;
  return result;
}

static inline void __bdd_array_free__(__bdd_array__ *arr) {
  if (arr) {
    free(arr->values);
    free(arr);
  }
}

static inline __bdd_array__ *__bdd_array_get_or_create__(__bdd_array__ **arr_ptr) {
  if (!*arr_ptr) {
    *arr_ptr = __bdd_array_create__();
  }
  return *arr_ptr;
}

static inline size_t __bdd_array_size__(__bdd_array__ *arr) { return arr ? arr->size : 0; }

typedef enum __bdd_node_type__ {
  __BDD_NODE_GROUP__ = 1,
  __BDD_NODE_TEST__ = 2,
  __BDD_NODE_INTERIM__ = 3
} __bdd_node_type__;

typedef enum __bdd_test_result__ {
  __BDD_RESULT_PENDING__ = 0,
  __BDD_RESULT_PASSED__ = 1,
  __BDD_RESULT_FAILED__ = 2,
  __BDD_RESULT_SKIPPED__ = 3,
  __BDD_RESULT_EXPECTED_FAIL__ = 4,
  __BDD_RESULT_UNEXPECTED_PASS__ = 5
} __bdd_test_result__;

typedef enum __bdd_node_flags__ {
  __bdd_node_flags_none__ = 0,
  __bdd_node_flags_focus__ = 1 << 0,
  __bdd_node_flags_skip__ = 1 << 1,
  __bdd_node_flags_expected_fail__ = 1 << 2,
  __bdd_node_flags_benchmark__ = 1 << 3,
} __bdd_node_flags__;

typedef struct __bdd_test_step__ {
  size_t level;
  int id;
  char *name;
  __bdd_node_type__ type;
  __bdd_node_flags__ flags;
  bool executed;
  bool passed;
  __bdd_test_result__ result;
  const char *skip_reason;
  char *failure_message;
  char *failure_location;
  double execution_time_ms;
  char *full_path; /* Full hierarchical path like "Calculator.should add two numbers" */
} __bdd_test_step__;

typedef struct __bdd_bench_entry__ {
  const char *title;
  size_t iters;
  double avg_us;
  double min_us;
  double max_us;
  double ops_s;
} __bdd_bench_entry__;

typedef struct __bdd_node__ {
  int id;
  int next_node_id;
  char *name;
  __bdd_node_flags__ flags;
  __bdd_node_type__ type;
  __bdd_array__ *list_before;
  __bdd_array__ *list_after;
  __bdd_array__ *list_before_each;
  __bdd_array__ *list_after_each;
  __bdd_array__ *list_children;
} __bdd_node__;

enum __bdd_run_type__ { __BDD_INIT_RUN__ = 1, __BDD_TEST_RUN__ = 2 };

typedef struct __bdd_config_type__ {
  enum __bdd_run_type__ run;
  int id;
  size_t test_index;
  size_t test_tap_index;
  size_t failed_test_count;
  __bdd_test_step__ *current_test;
  __bdd_array__ *node_stack;
  __bdd_array__ *nodes;
  char *error;
  char *location;
  bool use_color;
  bool use_tap;
  bool has_focus_nodes;
  const char *junit_file;
  size_t warn_count;
  size_t assertion_count;
  size_t assertion_failed_count;
  char info_buffer[2048];
  size_t info_len;
  char location_buf[512];
  jmp_buf jump_buffer;
  bool skip_subsequent;
  bool list_only;
  const char *filter;
  __bdd_bench_entry__ *bench_entries;
  size_t bench_count;
  size_t bench_cap;
  int bench_header_printed;  /* replaces static __bdd_bench_header_printed__ */
  size_t bench_header_level; /* replaces static __bdd_bench_header_level__   */
} __bdd_config_type__;

static __BDD_NO_SANITIZE_ADDRESS__ void __bdd_longjmp_fail__(
    __bdd_config_type__ *config) {
  if (!config) {
    abort();
  }
  longjmp(config->jump_buffer, 1);
}

static inline void __bdd_bench_reset__(__bdd_config_type__ *config) {
  config->bench_count = 0;
  config->bench_header_printed = 0;
  if (!config->bench_entries) {
    config->bench_cap = BDD_BENCH_MAX;
    config->bench_entries =
        __BDD_CAST(__bdd_bench_entry__ *, calloc(config->bench_cap, sizeof(__bdd_bench_entry__)));
  }
}

static inline void __bdd_bench_add__(__bdd_config_type__ *config, const char *title, size_t iters,
                                     double sum_ms, double min_ms, double max_ms, double scale) {
  size_t logical_iters;
  double logical_scale;

  if (!config->bench_entries) {
    __bdd_bench_reset__(config);
  }
  if (config->bench_count >= config->bench_cap) {
    size_t new_cap = config->bench_cap ? config->bench_cap * 2 : BDD_BENCH_MAX;
    __bdd_bench_entry__ *n =
        __BDD_CAST(__bdd_bench_entry__ *,
                   realloc(config->bench_entries, new_cap * sizeof(__bdd_bench_entry__)));
    if (!n) return;
    config->bench_entries = n;
    config->bench_cap = new_cap;
  }
  __bdd_bench_entry__ *e = &config->bench_entries[config->bench_count++];
  logical_scale = (scale > 0.0) ? scale : 1.0;
  logical_iters = __BDD_CAST(size_t, ((double)iters) * logical_scale);
  e->title = title;
  e->iters = logical_iters;
  e->avg_us = ((sum_ms / (double)iters) * 1000.0) / logical_scale;
  e->min_us = (min_ms * 1000.0) / logical_scale;
  e->max_us = (max_ms * 1000.0) / logical_scale;
  e->ops_s = (sum_ms > 0.0) ? (((double)iters) * logical_scale) / (sum_ms / 1000.0) : 0.0;
}

static inline void __bdd_bench_flush__(__bdd_config_type__ *config, size_t level, bool use_color) {
  if (!config->bench_entries || config->bench_count == 0) return;
  __bdd_bench_print_header__(config, level);
  for (size_t i = 0; i < config->bench_count; ++i) {
    __bdd_bench_entry__ *e = &config->bench_entries[i];
    __bdd_indent__(stdout, level);
    printf("  %s%-*s%s  %8zu  %11.3f  %11.3f  %11.3f  %11.0f\n",
           use_color ? __BDD_COLOR_MAGENTA__ : "", BDD_BENCH_NAME_WIDTH, e->title,
           use_color ? __BDD_COLOR_RESET__ : "", e->iters, e->avg_us, e->min_us, e->max_us,
           e->ops_s);
  }
}

static inline void __bdd_bench_cleanup__(__bdd_config_type__ *config) {
  free(config->bench_entries);
  config->bench_entries = NULL;
  config->bench_count = 0;
  config->bench_cap = 0;
  config->bench_header_printed = 0;
  config->bench_header_level = 0;
}

static inline __bdd_test_step__ *__bdd_test_step_create__(size_t level, __bdd_node__ *node) {
  __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, malloc(sizeof(__bdd_test_step__)));
  if (!step) {
    perror("malloc(step)");
    abort();
  }
  step->id = node->id;
  step->level = level;
  step->type = node->type;
  step->name = node->name;
  step->flags = node->flags;
  step->executed = false;
  step->passed = false;
  step->result = __BDD_RESULT_PENDING__;
  step->skip_reason = NULL;
  step->failure_message = NULL;
  step->failure_location = NULL;
  step->execution_time_ms = 0.0;
  step->full_path = NULL;
  return step;
}

static inline void __bdd_test_step_free__(__bdd_test_step__ *step) {
  if (step) {
    free(step->failure_message);
    free(step->failure_location);
    free(step->full_path);
    free(step);
  }
}

static char *__bdd_build_full_path__(__bdd_test_step__ **group_stack, int stack_depth,
                                     const char *test_name) {
  size_t total_len = 0;
  char *path;
  char *dst;

  for (int i = 1; i < stack_depth; ++i) {
    const char *segment = (group_stack[i] != NULL) ? group_stack[i]->name : NULL;
    if (!segment || segment[0] == '\0') continue;
    if (total_len > 0) total_len += 1;
    total_len += strlen(segment);
  }

  if (test_name && test_name[0] != '\0') {
    if (total_len > 0) total_len += 1;
    total_len += strlen(test_name);
  }

  if (total_len == 0) return NULL;

  path = __BDD_CAST(char *, malloc(total_len + 1));
  if (!path) {
    perror("malloc(full_path)");
    abort();
  }

  dst = path;
  for (int i = 1; i < stack_depth; ++i) {
    const char *segment = (group_stack[i] != NULL) ? group_stack[i]->name : NULL;
    size_t len;
    if (!segment || segment[0] == '\0') continue;
    if (dst != path) *dst++ = '.';
    len = strlen(segment);
    memcpy(dst, segment, len);
    dst += len;
  }

  if (test_name && test_name[0] != '\0') {
    size_t len = strlen(test_name);
    if (dst != path) *dst++ = '.';
    memcpy(dst, test_name, len);
    dst += len;
  }

  *dst = '\0';
  return path;
}

static inline __bdd_node__ *__bdd_node_create__(int id, const char *name, __bdd_node_type__ type,
                                                __bdd_node_flags__ flags) {
  __bdd_node__ *n = __BDD_CAST(__bdd_node__ *, malloc(sizeof(__bdd_node__)));
  if (!n) {
    perror("malloc(node)");
    abort();
  }
  n->id = id;
  n->next_node_id = id + 1;
  n->name = __BDD_CONST_CAST(char *, name); /* node takes ownership of name */
  n->type = type;
  n->flags = flags;
  n->list_before = NULL;
  n->list_after = NULL;
  n->list_before_each = NULL;
  n->list_after_each = NULL;
  n->list_children = NULL;
  return n;
}

static inline bool __bdd_node_is_leaf__(__bdd_node__ *node) {
  return !node->list_children || node->list_children->size == 0;
}

static void __bdd_node_flatten_internal__(__bdd_config_type__ *config, size_t level,
                                          __bdd_node__ *node, __bdd_array__ *steps,
                                          __bdd_array__ *before_each_lists,
                                          __bdd_array__ *after_each_lists) {
  if (__bdd_node_is_leaf__(node)) {
    for (size_t listIndex = 0; listIndex < before_each_lists->size; ++listIndex) {
      __bdd_array__ *list = __BDD_CAST(__bdd_array__ *, before_each_lists->values[listIndex]);
      for (size_t i = 0; i < __bdd_array_size__(list); ++i) {
        __bdd_array_push__(
            steps, __bdd_test_step_create__(level, __BDD_CAST(__bdd_node__ *, list->values[i])));
      }
    }

    __bdd_array_push__(steps, __bdd_test_step_create__(level, node));

    for (size_t listIndex = 0; listIndex < after_each_lists->size; ++listIndex) {
      size_t reverseListIndex = after_each_lists->size - listIndex - 1;
      __bdd_array__ *list = __BDD_CAST(__bdd_array__ *, after_each_lists->values[reverseListIndex]);
      for (size_t i = 0; i < __bdd_array_size__(list); ++i) {
        __bdd_array_push__(
            steps, __bdd_test_step_create__(level, __BDD_CAST(__bdd_node__ *, list->values[i])));
      }
    }
    return;
  }

  __bdd_array_push__(steps, __bdd_test_step_create__(level, node));

  for (size_t i = 0; i < __bdd_array_size__(node->list_before); ++i) {
    __bdd_array_push__(
        steps, __bdd_test_step_create__(level + 1,
                                        __BDD_CAST(__bdd_node__ *, node->list_before->values[i])));
  }

  __bdd_array_push__(before_each_lists, node->list_before_each);
  __bdd_array_push__(after_each_lists, node->list_after_each);

  for (size_t i = 0; i < __bdd_array_size__(node->list_children); ++i) {
    __bdd_node_flatten_internal__(config, level + 1,
                                  __BDD_CAST(__bdd_node__ *, node->list_children->values[i]), steps,
                                  before_each_lists, after_each_lists);
  }

  __bdd_array_pop__(before_each_lists);
  __bdd_array_pop__(after_each_lists);

  for (size_t i = 0; i < __bdd_array_size__(node->list_after); ++i) {
    __bdd_array_push__(
        steps, __bdd_test_step_create__(level + 1,
                                        __BDD_CAST(__bdd_node__ *, node->list_after->values[i])));
  }
}

static __bdd_array__ *__bdd_node_flatten__(__bdd_config_type__ *config, __bdd_node__ *node,
                                           __bdd_array__ *steps) {
  if (node == NULL) {
    return steps;
  }

  __bdd_array__ *before_each_lists = __bdd_array_create__();
  __bdd_array__ *after_each_lists = __bdd_array_create__();
  __bdd_node_flatten_internal__(config, 0, node, steps, before_each_lists, after_each_lists);
  __bdd_array_free__(before_each_lists);
  __bdd_array_free__(after_each_lists);

  return steps;
}

static void __bdd_node_free__(__bdd_node__ *n) {
  free(n->name);
  __bdd_array_free__(n->list_before);
  __bdd_array_free__(n->list_after);
  __bdd_array_free__(n->list_before_each);
  __bdd_array_free__(n->list_after_each);
  __bdd_array_free__(n->list_children);
  free(n);
}

__BDD_SELECTANY __BDD_TLS __bdd_spec_fn__ __bdd_current_spec_fn__ = NULL;
__BDD_SELECTANY __BDD_TLS __bdd_config_type__ *__bdd_active_config__ = NULL;

static inline void __bdd_test_main__(__bdd_config_type__ *config) {
  __bdd_active_config__ = config;
  if (__bdd_current_spec_fn__) {
    __bdd_current_spec_fn__(config);
  }
}
static char *__bdd_vformat__(const char *format, va_list va);

static void __bdd_indent__(FILE *fp, size_t level) {
  if (!fp) return;
  for (size_t i = 0; i < level; ++i) {
    fprintf(fp, "  ");
  }
}

static inline void __bdd_bench_print_header__(__bdd_config_type__ *config, size_t level) {
  (void)config;
#if BDD_BENCH_TABLE
  if (!config) return;
  if (config->bench_header_printed && config->bench_header_level == level) return;
  config->bench_header_printed = 1;
  config->bench_header_level = level;
  __bdd_indent__(stdout, level);
  printf("  %-*s  %8s  %11s  %11s  %11s  %11s\n", BDD_BENCH_NAME_WIDTH, "benchmark", "iters",
         "avg(us)", "min(us)", "max(us)", "ops/s");
  __bdd_indent__(stdout, level);
  printf("  %-*s  %8s  %11s  %11s  %11s  %11s\n", BDD_BENCH_NAME_WIDTH, "---------", "-----",
         "-------", "-------", "-------", "-----");
#endif
}

static inline void __bdd_bench_print__(__bdd_config_type__ *config, const char *title,
                                       size_t iters, double sum_ms, double min_ms, double max_ms,
                                       double scale, size_t level, bool use_color) {
  __bdd_bench_entry__ e;

  memset(&e, 0, sizeof(e));
  e.title = title;
  scale = (scale > 0.0) ? scale : 1.0;
  e.iters = __BDD_CAST(size_t, ((double)iters) * scale);
  e.avg_us = ((sum_ms / (double)iters) * 1000.0) / scale;
  e.min_us = (min_ms * 1000.0) / scale;
  e.max_us = (max_ms * 1000.0) / scale;
  e.ops_s = (sum_ms > 0.0) ? (((double)iters) * scale) / (sum_ms / 1000.0) : 0.0;

  __bdd_bench_print_header__(config, level);
  __bdd_indent__(stdout, level);
#if BDD_BENCH_TABLE
  printf("  %s%-*s%s  %8zu  %11.3f  %11.3f  %11.3f  %11.0f\n",
         use_color ? __BDD_COLOR_MAGENTA__ : "", BDD_BENCH_NAME_WIDTH, e.title,
         use_color ? __BDD_COLOR_RESET__ : "", e.iters, e.avg_us, e.min_us, e.max_us, e.ops_s);
#else
  printf("%s%-*s%s  iters=%zu  avg=%9.3f us  min=%9.3f us  max=%9.3f us  rate/s=%9.0f\n",
         use_color ? __BDD_COLOR_MAGENTA__ : "", BDD_BENCH_NAME_WIDTH, e.title,
         use_color ? __BDD_COLOR_RESET__ : "", e.iters, e.avg_us, e.min_us, e.max_us, e.ops_s);
#endif
}

static bool __bdd_enter_node__(__bdd_node_flags__ node_flags, __bdd_config_type__ *config,
                               __bdd_node_type__ type, ptrdiff_t list_offset, const char *fmt,
                               ...) {
  if (config->run == __BDD_INIT_RUN__) {
    va_list va;
    va_start(va, fmt);
    char *name = __bdd_vformat__(fmt, va);
    va_end(va);

    __bdd_node__ *top = __BDD_CAST(__bdd_node__ *, __bdd_array_last__(config->node_stack));
    if (!top) {
      fprintf(stderr, "error: node_stack is empty\n");
      abort();
    }
    __bdd_array__ **list_ptr = __BDD_REINTERPRET_CAST(
        __bdd_array__ **, __BDD_REINTERPRET_CAST(unsigned char *, top) + list_offset);
    __bdd_array__ *list = __bdd_array_get_or_create__(list_ptr);

    int id = config->id++;
    __bdd_node__ *node = __bdd_node_create__(id, name, type, node_flags);
    if (node_flags & __bdd_node_flags_focus__) {
      top->flags = (__bdd_node_flags__)(top->flags | (node_flags & __bdd_node_flags_focus__));
      config->has_focus_nodes = true;
    }
    __bdd_array_push__(list, node);
    __bdd_array_push__(config->nodes, node);
    if (type == __BDD_NODE_GROUP__) {
      __bdd_array_push__(config->node_stack, node);
      return true;
    }
    return false;
  }

  /* TEST_RUN: no name allocation needed, use existing node */
  if (config->id >= (int)config->nodes->size) {
    fprintf(stderr, "non-deterministic spec\n");
    abort();
  }
  if (!config->nodes || !config->nodes->values) {
    fprintf(stderr, "error: config->nodes is invalid\n");
    abort();
  }
  __bdd_node__ *node = __BDD_CAST(__bdd_node__ *, config->nodes->values[config->id]);
  if (!node) {
    fprintf(stderr, "error: node at %d is NULL\n", config->id);
    abort();
  }

  __bdd_test_step__ *step = config->current_test;
  bool should_enter = step->id >= node->id && step->id < node->next_node_id;
  if (should_enter) {
    __bdd_array_push__(config->node_stack, node);
    config->id++;
  } else {
    config->id = node->next_node_id;
  }
#if defined(BDD_PRINT_TRACE)
  const char *color = config->use_color ? __BDD_COLOR_MAGENTA__ : "";
  fprintf(stderr, "%s% 3d ", color, step->id);
  __bdd_indent__(stderr, config->node_stack->size - 1 - (int)should_enter);
  const char *reset = config->use_color ? __BDD_COLOR_RESET__ : "";
  fprintf(stderr, "%s [%d, %d) %s%s\n", should_enter ? ">" : "|", node->id, node->next_node_id,
          node->name, reset);
#endif
  return should_enter;
}

static void __bdd_exit_node__(__bdd_config_type__ *config) {
  __bdd_node__ *top = __BDD_CAST(__bdd_node__ *, __bdd_array_pop__(config->node_stack));
  if (top && config->run == __BDD_INIT_RUN__) {
    top->next_node_id = config->id;
  }
}

static inline const char *__bdd_skip_message__(const __bdd_test_step__ *step) {
  return (step && step->skip_reason && step->skip_reason[0] != '\0') ? step->skip_reason
                                                                      : "Test was skipped";
}

static void __bdd_report_skip__(__bdd_config_type__ *config, __bdd_test_step__ *step) {
  if (config->run == __BDD_TEST_RUN__) {
    if (config->use_tap) {
      if (config->test_tap_index) {
        printf("ok %zu - %s # SKIP %s\n", config->test_tap_index, step->name,
               __bdd_skip_message__(step));
      }
    } else {
      __bdd_indent__(stdout, step->level);
      printf("%s[ SKIP  ]%s\n", config->use_color ? __BDD_COLOR_YELLOW__ : "",
             config->use_color ? __BDD_COLOR_RESET__ : "");
    }
  }
}

static void __bdd_report_pass__(__bdd_config_type__ *config, __bdd_test_step__ *step) {
  if (config->run == __BDD_TEST_RUN__) {
    if (config->use_tap) {
      if (config->test_tap_index) {
        printf("ok %zu - %s\n", config->test_tap_index, step->name);
      }
    } else {
      if (step->flags & __bdd_node_flags_benchmark__) {
        return;
      }
      __bdd_indent__(stdout, step->level);
      printf("%s[ OK    ]%s", config->use_color ? __BDD_COLOR_GREEN__ : "",
             config->use_color ? __BDD_COLOR_RESET__ : "");
      if (step->execution_time_ms > 0.0) {
        printf(" (%.2fms)", step->execution_time_ms);
      }
      printf("\n");
    }
  }
}

static void __bdd_report_unexpected_pass__(__bdd_config_type__ *config, __bdd_test_step__ *step) {
  ++config->failed_test_count;
  if (config->run == __BDD_TEST_RUN__) {
    if (config->use_tap) {
      if (config->test_tap_index) {
        printf("not ok %zu - %s # TODO was expected to fail but passed\n", config->test_tap_index,
               step->name);
      }
    } else {
      __bdd_indent__(stdout, step->level);
      printf("%s[ UPASS ]%s", config->use_color ? __BDD_COLOR_YELLOW__ : "",
             config->use_color ? __BDD_COLOR_RESET__ : "");
      if (step->execution_time_ms > 0.0) {
        printf(" (%.2fms)", step->execution_time_ms);
      }
      printf("\n");
      __bdd_indent__(stdout, step->level + 1);
      printf("This test was expected to fail but passed\n");
    }
  }
}

static void __bdd_report_expected_fail__(__bdd_config_type__ *config, __bdd_test_step__ *step) {
  if (config->run == __BDD_TEST_RUN__) {
    if (config->use_tap) {
      if (config->test_tap_index) {
        printf("ok %zu - %s # TODO expected failure\n", config->test_tap_index, step->name);
      }
    } else {
      __bdd_indent__(stdout, step->level);
      printf("%s[ XFAIL ]%s", config->use_color ? __BDD_COLOR_GREEN__ : "",
             config->use_color ? __BDD_COLOR_RESET__ : "");
      if (step->execution_time_ms > 0.0) {
        printf(" (%.2fms)", step->execution_time_ms);
      }
      printf("\n");
    }
  }
}

static void __bdd_report_fail__(__bdd_config_type__ *config, __bdd_test_step__ *step) {
  ++config->failed_test_count;
  if (config->use_tap) {
    if (config->test_tap_index) {
      printf("not ok %zu - %s\n", config->test_tap_index, step->name);
    }
  } else {
    __bdd_indent__(stdout, step->level);
    printf("%s[ FAIL  ]%s", config->use_color ? __BDD_COLOR_RED__ : "",
           config->use_color ? __BDD_COLOR_RESET__ : "");
    if (step->execution_time_ms > 0.0) {
      printf(" (%.2fms)", step->execution_time_ms);
    }
    printf("\n");
    if (config->info_len > 0) {
      __bdd_indent__(stdout, step->level + 1);
      printf("with info: %s\n", config->info_buffer);
    }
    __bdd_indent__(stdout, step->level + 1);
    printf("%s\n", config->error);
    __bdd_indent__(stdout, step->level + 2);
    printf("%s\n", config->location);
  }
}

static void __bdd_run__(__bdd_config_type__ *config) {
  __bdd_test_step__ *step = config->current_test;

  if (step->type == __BDD_NODE_GROUP__ && !config->use_tap) {
    if (config->has_focus_nodes && !(step->flags & __bdd_node_flags_focus__)) {
      return;
    }
    __bdd_indent__(stdout, step->level);
    printf("%s%s%s\n", config->use_color ? __BDD_COLOR_BOLD__ : "", step->name,
           config->use_color ? __BDD_COLOR_RESET__ : "");
    return;
  }

  bool skipped = false;
  const char *skip_reason = NULL;
  if (step->type == __BDD_NODE_TEST__) {
    step->result = __BDD_RESULT_PENDING__;
    step->skip_reason = NULL;
    if (step->flags & __bdd_node_flags_skip__) {
      skipped = true;
      skip_reason = "Test was skipped";
    } else if (config->has_focus_nodes && !(step->flags & __bdd_node_flags_focus__)) {
      skipped = true;
      skip_reason = "filtered by focus";
    } else if (config->filter && !strstr(step->name, config->filter) &&
               !(step->full_path && strstr(step->full_path, config->filter))) {
      skipped = true;
      skip_reason = "filtered out";
    }
    ++config->test_tap_index;

    /* Print the step name before running the test so it is visible even if the test crashes */
    if (config->run == __BDD_TEST_RUN__ && !config->use_tap) {
      __bdd_indent__(stdout, step->level);
      printf("%s\n", step->name);
      fflush(stdout);
    }

    if (!skipped) {
      if (config->error) {
        free(config->error);
        config->error = NULL;
      }
      config->location = NULL;
      config->info_buffer[0] = '\0';
      config->info_len = 0;
      config->skip_subsequent = false;

      if (step->flags & __bdd_node_flags_benchmark__) {
        __bdd_bench_reset__(config);
      }

      double start_time = __bdd_get_time_ms__();
      if (setjmp(config->jump_buffer) == 0) {
        __bdd_test_main__(config);
      }
      /* If longjmp happens, node_stack might be unbalanced. TEST_RUN resets it in main(). */
      double end_time = __bdd_get_time_ms__();
      step->execution_time_ms = end_time - start_time;
      if (step->flags & __bdd_node_flags_benchmark__) {
        __bdd_bench_flush__(config, step->level + 1, config->use_color);
      }
    }

    if (skipped) {
      step->executed = false;
      step->passed = false;
      step->result = __BDD_RESULT_SKIPPED__;
      step->skip_reason = skip_reason;
      __bdd_report_skip__(config, step);
    } else if (config->error == NULL) {
      /* Test passed */
      step->executed = true;
      bool is_expected_fail = (step->flags & __bdd_node_flags_expected_fail__);
      if (is_expected_fail) {
        step->passed = false;
        step->result = __BDD_RESULT_UNEXPECTED_PASS__;
        step->failure_message = strdup("Expected to fail but passed");
        __bdd_report_unexpected_pass__(config, step);
      } else {
        step->passed = true;
        step->result = __BDD_RESULT_PASSED__;
        __bdd_report_pass__(config, step);
      }
    } else {
      /* Test failed */
      step->executed = true;
      bool is_expected_fail = (step->flags & __bdd_node_flags_expected_fail__);
      if (is_expected_fail) {
        step->passed = true; /* Expected failure counts as pass */
        step->result = __BDD_RESULT_EXPECTED_FAIL__;
        __bdd_report_expected_fail__(config, step);
      } else {
        step->passed = false;
        step->result = __BDD_RESULT_FAILED__;
        step->failure_message = strdup(config->error);
        step->failure_location = strdup(config->location);
        __bdd_report_fail__(config, step);
      }
      free(config->error);
      config->error = NULL;
    }
  } else if (!skipped) {
    __bdd_test_main__(config);
  }
}

static char *__bdd_vformat__(const char *format, va_list va) {
  va_list va2;
  va_copy(va2, va);
  int len = vsnprintf(NULL, 0, format, va2);
  va_end(va2);

  char *result = __BDD_CAST(char *, malloc((size_t)len + 1));
  if (!result) {
    perror("malloc(result)");
    abort();
  }
  vsnprintf(result, (size_t)len + 1, format, va);
  return result;
}

static char *__bdd_format__(const char *format, ...) {
  va_list va;
  va_start(va, format);
  char *buf = __bdd_vformat__(format, va);
  va_end(va);
  return buf;
}

static bool __bdd_is_supported_term__(void) {
  bool result;
  const char *term = getenv("TERM");
  result = term && strcmp(term, "") != 0;
#ifndef _WIN32
  return result;
#else
  if (result) {
    return 1;
  }

  /* Attempt to enable virtual terminal processing on Windows.
   * See: https://msdn.microsoft.com/en-us/library/windows/desktop/mt638032(v=vs.85).aspx */
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hOut == INVALID_HANDLE_VALUE) {
    return 0;
  }

  DWORD dwMode = 0;
  if (!GetConsoleMode(hOut, &dwMode)) {
    return 0;
  }

  dwMode |= 0x4; /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */
  if (!SetConsoleMode(hOut, dwMode)) {
    return 0;
  }

  return 1;
#endif
}

static inline int __bdd_is_xml_char__(unsigned char c) {
  return c == '\t' || c == '\n' || c == '\r' || c >= 0x20;
}

static const char *__bdd_skip_ansi_escape__(const char *p) {
  if (!p || p[0] != '\x1B') return p;
  ++p;
  if (*p == '[') {
    ++p;
    while (*p && (((unsigned char)*p >= 0x30 && (unsigned char)*p <= 0x3F) ||
                  ((unsigned char)*p >= 0x20 && (unsigned char)*p <= 0x2F))) {
      ++p;
    }
    if (*p && (unsigned char)*p >= 0x40 && (unsigned char)*p <= 0x7E) {
      ++p;
    }
    return p;
  }
  if (*p == ']') {
    ++p;
    while (*p && *p != '\a') {
      if (p[0] == '\x1B' && p[1] == '\\') {
        p += 2;
        return p;
      }
      ++p;
    }
    if (*p == '\a') ++p;
    return p;
  }
  if (*p) ++p;
  return p;
}

static void __bdd_xml_escape__(FILE *f, const char *str) {
  if (!str) return;
  for (const char *p = str; *p;) {
    unsigned char c = (unsigned char)*p;
    if (c == 0x1B) {
      p = __bdd_skip_ansi_escape__(p);
      continue;
    }
    if (!__bdd_is_xml_char__(c)) {
      ++p;
      continue;
    }
    switch (*p) {
    case '&':
      fprintf(f, "&amp;");
      break;
    case '<':
      fprintf(f, "&lt;");
      break;
    case '>':
      fprintf(f, "&gt;");
      break;
    case '"':
      fprintf(f, "&quot;");
      break;
    case '\'':
      fprintf(f, "&apos;");
      break;
    default:
      fputc(*p, f);
      break;
    }
    ++p;
  }
}

static void __bdd_generate_junit__(__bdd_config_type__ *config, __bdd_array__ *steps,
                                   size_t test_count) {
  FILE *f = fopen(config->junit_file, "w");
  if (!f) {
    fprintf(stderr, "Error: Could not open JUnit output file: %s\n", config->junit_file);
    return;
  }

  /* Get current timestamp in ISO 8601 format */
  time_t now = time(NULL);
  struct tm *tm_info = gmtime(&now);
  char timestamp[32];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);

  /* Count skipped tests */
  size_t skipped_count = 0;
  for (size_t i = 0; i < steps->size; ++i) {
    __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, steps->values[i]);
    if (step->type == __BDD_NODE_TEST__ && step->result == __BDD_RESULT_SKIPPED__) {
      ++skipped_count;
    }
  }

  fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf(f,
          "<testsuites name=\"BDD Test Run\" tests=\"%zu\" failures=\"%zu\" errors=\"0\" "
          "skipped=\"%zu\" time=\"0\" timestamp=\"%s\">\n",
          test_count, config->failed_test_count, skipped_count, timestamp);

  fprintf(f,
          "  <testsuite name=\"tinytest\" tests=\"%zu\" failures=\"%zu\" errors=\"0\" "
          "skipped=\"%zu\" time=\"0\" timestamp=\"%s\">\n",
          test_count, config->failed_test_count, skipped_count, timestamp);

  for (size_t i = 0; i < steps->size; ++i) {
    __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, steps->values[i]);
    if (step->type == __BDD_NODE_TEST__) {
      fprintf(f, "    <testcase name=\"");
      __bdd_xml_escape__(f, step->name);
      fprintf(f, "\" classname=\"");
      /* Use full hierarchical path if available, otherwise use a generic name */
      if (step->full_path) {
        __bdd_xml_escape__(f, step->full_path);
      } else {
        __bdd_xml_escape__(f, "tinytest");
      }
      fprintf(f, "\" time=\"%.6f\"", step->execution_time_ms / 1000.0); /* Convert ms to seconds */

      bool is_skip = step->result == __BDD_RESULT_SKIPPED__;
      bool is_fail = step->result == __BDD_RESULT_FAILED__ ||
                     step->result == __BDD_RESULT_UNEXPECTED_PASS__;

      if (is_skip) {
        fprintf(f, ">\n");
        fprintf(f, "      <skipped message=\"");
        __bdd_xml_escape__(f, __bdd_skip_message__(step));
        fprintf(f, "\" />\n");
        fprintf(f, "    </testcase>\n");
      } else if (is_fail) {
        /* Test failed */
        fprintf(f, ">\n");
        fprintf(f, "      <failure message=\"");
        if (step->failure_message) {
          __bdd_xml_escape__(f, step->failure_message);
        } else {
          fprintf(f, "Test failed");
        }
        fprintf(f, "\" type=\"AssertionError\">");
        if (step->failure_location) {
          fprintf(f, "\n");
          __bdd_xml_escape__(f, step->failure_location);
        }
        if (step->failure_message) {
          fprintf(f, "\n");
          __bdd_xml_escape__(f, step->failure_message);
        }
        fprintf(f, "\n      </failure>\n");
        fprintf(f, "    </testcase>\n");
      } else {
        /* Test passed */
        fprintf(f, " />\n");
      }
    }
  }

  fprintf(f, "  </testsuite>\n");
  fprintf(f, "</testsuites>\n");
  fclose(f);
}

#ifdef __cplusplus
}
#endif

/* main() must not be in extern "C" block */
#ifndef TINYTEST_NO_MAIN
int main(int argc, char **argv) {
  double __bdd_start_time__ = __bdd_get_time_ms__();
  struct __bdd_config_type__ config;
  memset(&config, 0, sizeof(config));
  config.run = __BDD_INIT_RUN__;

  /* Parse command-line arguments */
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--tap") == 0) {
      config.use_tap = 1;
    } else if (strcmp(argv[i], "--color") == 0) {
      config.use_color = 1;
    } else if (strcmp(argv[i], "--no-color") == 0) {
      config.use_color = 0;
    } else if (strcmp(argv[i], "--junit") == 0) {
      if (i + 1 < argc) {
        config.junit_file = argv[++i];
      } else {
        fprintf(stderr, "Error: --junit requires a filename argument\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--list") == 0 || strcmp(argv[i], "-l") == 0) {
      config.list_only = 1;
    } else if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-f") == 0) {
      if (i + 1 < argc) {
        config.filter = argv[++i];
      } else {
        fprintf(stderr, "Error: --filter requires a pattern argument\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [options]\n", argv[0]);
      printf("Options:\n");
      printf("  --tap              Output in TAP (Test Anything Protocol) format\n");
      printf("  --color            Force colored output\n");
      printf("  --no-color         Disable colored output\n");
      printf("  --junit <file>     Generate JUnit XML report to specified file\n");
      printf("  --list, -l         List all test names without running them\n");
      printf("  --filter, -f <pat> Run only tests whose name contains <pat>\n");
      printf("  --help, -h         Show this help message\n");
      return 0;
    }
  }

  const char *tap_env = getenv("BDD_USE_TAP");
  if (!config.use_tap &&
      (BDD_USE_TAP || (tap_env && strcmp(tap_env, "") != 0 && strcmp(tap_env, "0") != 0))) {
    config.use_tap = 1;
  }

  if (!config.use_tap && !config.use_color && BDD_USE_COLOR && __BDD_IS_ATTY__() &&
      __bdd_is_supported_term__()) {
    config.use_color = 1;
  }

  if (__bdd_spec_count__ == 0) {
    fprintf(stderr, "tinytest: no specs registered\n");
    return 1;
  }

  __bdd_array__ *all_steps = __bdd_array_create__();
  __bdd_array__ *all_step_arrays = __bdd_array_create__();
  __bdd_array__ *all_roots = __bdd_array_create__();
  __bdd_array__ *all_nodes = __bdd_array_create__();
  __bdd_array__ *all_stacks = __bdd_array_create__();
  __bdd_array__ *all_specs = __bdd_array_create__();

  bool has_focus_any = false;
  size_t total_test_count = 0;

  for (size_t s = 0; s < __bdd_spec_count__; ++s) {
    __bdd_spec_entry__ *spec = __bdd_get_spec_entry__(s);
    if (!spec) continue;

    config.run = __BDD_INIT_RUN__;
    config.id = 0;
    config.has_focus_nodes = false;
    config.nodes = __bdd_array_create__();
    config.node_stack = __bdd_array_create__();

    __bdd_node__ *root =
        __bdd_node_create__(-1, spec->name, __BDD_NODE_GROUP__, __bdd_node_flags_none__);
    __bdd_array_push__(config.node_stack, root);

    __bdd_current_spec_fn__ = spec->fn;
    __bdd_test_main__(&config);

    if (config.has_focus_nodes) {
      has_focus_any = true;
    }

    __bdd_array__ *steps = __bdd_array_create__();
    __bdd_node_flatten__(&config, root, steps);

    __bdd_test_step__ *group_stack[32];
    int stack_depth = 0;

    for (size_t i = 0; i < steps->size; ++i) {
      __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, steps->values[i]);

      while (stack_depth > 0 && group_stack[stack_depth - 1]->level >= step->level) {
        stack_depth--;
      }

      if (step->type == __BDD_NODE_GROUP__) {
        if (stack_depth < 32) {
          group_stack[stack_depth++] = step;
        }
      } else if (step->type == __BDD_NODE_TEST__) {
        step->full_path = __bdd_build_full_path__(group_stack, stack_depth, step->name);
        ++total_test_count;
      }
    }

    __bdd_array_push__(all_roots, root);
    __bdd_array_push__(all_nodes, config.nodes);
    __bdd_array_push__(all_stacks, config.node_stack);
    __bdd_array_push__(all_step_arrays, steps);
    __bdd_array_push__(all_specs, spec);

    for (size_t i = 0; i < steps->size; ++i) {
      __bdd_array_push__(all_steps, steps->values[i]);
    }
  }

  if (config.list_only) {
    for (size_t i = 0; i < all_steps->size; ++i) {
      __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, all_steps->values[i]);
      if (step->type == __BDD_NODE_GROUP__) {
        __bdd_indent__(stdout, step->level);
        printf("%s\n", step->name);
      } else if (step->type == __BDD_NODE_TEST__) {
        __bdd_indent__(stdout, step->level);
        const char *tag = "";
        if (step->flags & __bdd_node_flags_skip__) tag = " [skip]";
        else if (step->flags & __bdd_node_flags_focus__) tag = " [focus]";
        else if (step->flags & __bdd_node_flags_expected_fail__) tag = " [should_fail]";
        printf("%s%s\n", step->name, tag);
      }
    }

    for (size_t i = 0; i < all_nodes->size; ++i) {
      __bdd_array__ *nodes = __BDD_CAST(__bdd_array__ *, all_nodes->values[i]);
      for (size_t j = 0; j < nodes->size; ++j) {
        __bdd_node_free__(__BDD_CAST(__bdd_node__ *, nodes->values[j]));
      }
      __bdd_array_free__(nodes);
      __bdd_array_free__(__BDD_CAST(__bdd_array__ *, all_stacks->values[i]));
      __bdd_node__ *root = __BDD_CAST(__bdd_node__ *, all_roots->values[i]);
      root->name = NULL;
      __bdd_node_free__(root);
    }
    for (size_t i = 0; i < all_steps->size; ++i) {
      __bdd_test_step_free__(__BDD_CAST(__bdd_test_step__ *, all_steps->values[i]));
    }
    for (size_t i = 0; i < all_step_arrays->size; ++i) {
      __bdd_array_free__(__BDD_CAST(__bdd_array__ *, all_step_arrays->values[i]));
    }
    __bdd_array_free__(all_roots);
    __bdd_array_free__(all_nodes);
    __bdd_array_free__(all_stacks);
    __bdd_array_free__(all_step_arrays);
    __bdd_array_free__(all_specs);
    __bdd_array_free__(all_steps);
    __bdd_bench_cleanup__(&config);
    __bdd_cleanup_specs__();
    return 0;
  }

  if (config.use_tap) {
    printf("TAP version 13\n1..%zu\n", total_test_count);
  }

  config.run = __BDD_TEST_RUN__;
  config.has_focus_nodes = has_focus_any;

  for (size_t i = 0; i < all_step_arrays->size; ++i) {
    __bdd_array__ *steps = __BDD_CAST(__bdd_array__ *, all_step_arrays->values[i]);
    __bdd_spec_entry__ *spec = __BDD_CAST(__bdd_spec_entry__ *, all_specs->values[i]);
    __bdd_current_spec_fn__ = spec ? spec->fn : NULL;
    config.node_stack = __BDD_CAST(__bdd_array__ *, all_stacks->values[i]);
    config.nodes = __BDD_CAST(__bdd_array__ *, all_nodes->values[i]);
    for (size_t j = 0; j < steps->size; ++j) {
      __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, steps->values[j]);
      config.node_stack->size = 1;
      config.id = 0;
      if (config.error) {
        free(config.error);
        config.error = NULL;
      }
      config.location = NULL;
      config.current_test = step;
      __bdd_run__(&config);
    }
  }

  double __bdd_end_time__ = __bdd_get_time_ms__();
  double __bdd_duration__ = (__bdd_end_time__ - __bdd_start_time__) / 1000.0;

  size_t passed_count = 0;
  size_t skipped_count = 0;
  size_t todo_count = 0;
  for (size_t i = 0; i < all_steps->size; ++i) {
    __bdd_test_step__ *step = __BDD_CAST(__bdd_test_step__ *, all_steps->values[i]);
    if (step->type != __BDD_NODE_TEST__) continue;
    if (step->result == __BDD_RESULT_SKIPPED__) {
      skipped_count++;
    } else if (step->result == __BDD_RESULT_EXPECTED_FAIL__) {
      todo_count++;
    } else if (step->result == __BDD_RESULT_PASSED__) {
      passed_count++;
    }
  }

  if (!config.use_tap) {
    const char *c_rst = config.use_color ? __BDD_COLOR_RESET__ : "";
    const char *c_red = config.use_color ? __BDD_COLOR_RED__ : "";
    const char *c_grn = config.use_color ? __BDD_COLOR_GREEN__ : "";
    const char *c_ylw = config.use_color ? __BDD_COLOR_YELLOW__ : "";
    const char *c_bld = config.use_color ? __BDD_COLOR_BOLD__ : "";

    printf("\n==All Tests Summary==\n");
    printf("Total tests %s[PASSED]:\t%zu%s\n", c_grn, passed_count, c_rst);
    printf("Total tests %s[FAILED]:\t%zu%s\n", c_red, config.failed_test_count, c_rst);
    if (config.warn_count > 0) {
      printf("Total tests %s[WARNED]:\t%zu%s\n", c_ylw, config.warn_count, c_rst);
    }
    printf("Total tests %s[SKIPPED]:\t%zu%s\n", c_ylw, skipped_count, c_rst);
    printf("Total tests %s[TODO]:   \t%zu%s\n", c_bld, todo_count, c_rst);
    printf("Assertions: %s%zu passed%s, %s%zu failed%s\n", c_grn,
           config.assertion_count - config.assertion_failed_count, c_rst,
           config.assertion_failed_count > 0 ? c_red : "", config.assertion_failed_count, c_rst);

    printf("%zu passed, %zu failed, %zu skipped, %zu todo, %zu assertions. Finished in %f sec.\n",
           passed_count, config.failed_test_count, skipped_count, todo_count,
           config.assertion_count, __bdd_duration__);

    if (config.failed_test_count == 0 && todo_count == 0) {
      printf("%sAll tests passed in %f sec.%s\n\n", c_grn, __bdd_duration__, c_rst);
    } else {
      printf("\n");
    }
  }

  /* Generate JUnit XML report if requested - must be done before freeing steps */
  if (config.junit_file) {
    __bdd_generate_junit__(&config, all_steps, total_test_count);
  }

  for (size_t i = 0; i < all_nodes->size; ++i) {
    __bdd_array__ *nodes = __BDD_CAST(__bdd_array__ *, all_nodes->values[i]);
    for (size_t j = 0; j < nodes->size; ++j) {
      __bdd_node_free__(__BDD_CAST(__bdd_node__ *, nodes->values[j]));
    }
    __bdd_array_free__(nodes);
    __bdd_array_free__(__BDD_CAST(__bdd_array__ *, all_stacks->values[i]));
    __bdd_node__ *root = __BDD_CAST(__bdd_node__ *, all_roots->values[i]);
    root->name = NULL;
    __bdd_node_free__(root);
  }
  for (size_t i = 0; i < all_step_arrays->size; ++i) {
    __bdd_array_free__(__BDD_CAST(__bdd_array__ *, all_step_arrays->values[i]));
  }
  __bdd_array_free__(all_roots);
  __bdd_array_free__(all_nodes);
  __bdd_array_free__(all_stacks);
  __bdd_array_free__(all_step_arrays);
  __bdd_array_free__(all_specs);
  for (size_t i = 0; i < all_steps->size; ++i) {
    __bdd_test_step_free__(__BDD_CAST(__bdd_test_step__ *, all_steps->values[i]));
  }
  __bdd_array_free__(all_steps);
  __bdd_bench_cleanup__(&config);
  __bdd_cleanup_specs__();

  return config.failed_test_count > 0 ? 1 : 0;
}
#endif /* TINYTEST_NO_MAIN */

#if defined(__GNUC__) || defined(__clang__)
  #define __BDD_UNUSED_PARAM__ __attribute__((unused))
#else
  #define __BDD_UNUSED_PARAM__
#endif

#define spec(name)                                                                                 \
  static void __BDD_CAT2(__bdd_spec_fn_, __LINE__)(__BDD_UNUSED_PARAM__ __bdd_config_type__ * __bdd_config__);          \
  __BDD_CONSTRUCTOR__(__BDD_CAT2(__bdd_spec_reg_, __LINE__)) {                                     \
    __bdd_register_spec__((name), __BDD_CAT2(__bdd_spec_fn_, __LINE__));                           \
  }                                                                                                \
  static void __BDD_CAT2(__bdd_spec_fn_, __LINE__)(__BDD_UNUSED_PARAM__ __bdd_config_type__ * __bdd_config__)

#define __BDD_CAT(a, b) a##b
#define __BDD_CAT2(a, b) __BDD_CAT(a, b)

#define __BDD_NODE__(flags, node_list, type, ...)                                                  \
  for (bool __BDD_CAT2(__bdd_has_run_, __LINE__) = 0;                                              \
       (!__BDD_CAT2(__bdd_has_run_, __LINE__) &&                                                   \
        __bdd_enter_node__(flags, __bdd_active_config__, (type),                                   \
                           offsetof(struct __bdd_node__, node_list), __VA_ARGS__));                \
       __bdd_exit_node__(__bdd_active_config__), __BDD_CAT2(__bdd_has_run_, __LINE__) = 1)

#define describe(...)                                                                              \
  __BDD_NODE__(__bdd_node_flags_none__, list_children, __BDD_NODE_GROUP__, __VA_ARGS__)
#define group(...) describe(__VA_ARGS__)
#define suite(...) spec(__VA_ARGS__)
#define it(...) __BDD_NODE__(__bdd_node_flags_none__, list_children, __BDD_NODE_TEST__, __VA_ARGS__)
#define bench(...)                                                                                 \
  __BDD_NODE__(__bdd_node_flags_benchmark__, list_children, __BDD_NODE_TEST__, __VA_ARGS__)
#define it_only(...)                                                                               \
  __BDD_NODE__(__bdd_node_flags_focus__, list_children, __BDD_NODE_TEST__, __VA_ARGS__)
#define fit(...) it_only(__VA_ARGS__)
#define it_skip(...)                                                                               \
  __BDD_NODE__(__bdd_node_flags_skip__, list_children, __BDD_NODE_TEST__, __VA_ARGS__)
#define xit(...) it_skip(__VA_ARGS__)
#define it_should_fail(...)                                                                        \
  __BDD_NODE__(__bdd_node_flags_expected_fail__, list_children, __BDD_NODE_TEST__, __VA_ARGS__)
#define before_each()                                                                              \
  __BDD_NODE__(__bdd_node_flags_none__, list_before_each, __BDD_NODE_INTERIM__, "before_each")
#define after_each()                                                                               \
  __BDD_NODE__(__bdd_node_flags_none__, list_after_each, __BDD_NODE_INTERIM__, "after_each")

#ifndef BDD_NO_CONTEXT_KEYWORD
  #define context(name) describe(name)
#endif

#define bdd_invoke(func, ...) func(__bdd_active_config__, ##__VA_ARGS__)

#define __BDD_MACRO__(M, ...) __BDD_OVERLOAD__(M, __BDD_COUNT_ARGS__(__VA_ARGS__))(__VA_ARGS__)
#define __BDD_OVERLOAD__(macro_name, suffix) __BDD_EXPAND_OVERLOAD__(macro_name, suffix)
#define __BDD_EXPAND_OVERLOAD__(macro_name, suffix) macro_name##suffix

#define __BDD_COUNT_ARGS__(...) __BDD_PATTERN_MATCH__(__VA_ARGS__, _, _, _, _, _, _, _, _, _, ONE__)
#define __BDD_PATTERN_MATCH__(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

#define __BDD_STRING_HELPER__(x) #x
#define __BDD_STRING__(x) __BDD_STRING_HELPER__(x)
#define __STRING__LINE__ __BDD_STRING__(__LINE__)

#define __BDD_FMT_COLOR__ __BDD_COLOR_RED__ "Check failed:" __BDD_COLOR_RESET__ " %s"
#define __BDD_FMT_PLAIN__ "Check failed: %s"

static inline int __bdd_eval_bool__(int v) { return v; }

/* Internal implementation that takes file and line */
#define __BDD_CHECK_IMPL__(condition, file, line, ...)                                             \
  do {                                                                                             \
    if (!__bdd_eval_bool__(!!(condition))) {                                                       \
      if (__bdd_active_config__) {                                                                 \
        ++__bdd_active_config__->assertion_count;                                                  \
        ++__bdd_active_config__->assertion_failed_count;                                           \
        if (__bdd_active_config__->run == __BDD_TEST_RUN__ && !__bdd_active_config__->error) {     \
          char *__bdd_message__ = __bdd_format__(__VA_ARGS__);                                     \
          const char *fmt =                                                                        \
              __bdd_active_config__->use_color ? __BDD_FMT_COLOR__ : __BDD_FMT_PLAIN__;            \
          snprintf(__bdd_active_config__->location_buf,                                            \
                   sizeof(__bdd_active_config__->location_buf), "at %s:%s", file, line);           \
          __bdd_active_config__->location = __bdd_active_config__->location_buf;                   \
          size_t bufflen = strlen(fmt) + strlen(__bdd_message__) + 1;                              \
          __bdd_active_config__->error = __BDD_CAST(char *, calloc(bufflen, sizeof(char)));        \
          if (__bdd_active_config__->use_color) {                                                  \
            snprintf(__bdd_active_config__->error, bufflen, __BDD_FMT_COLOR__, __bdd_message__);   \
          } else {                                                                                 \
            snprintf(__bdd_active_config__->error, bufflen, __BDD_FMT_PLAIN__, __bdd_message__);   \
          }                                                                                        \
          free(__bdd_message__);                                                                   \
          __bdd_longjmp_fail__(__bdd_active_config__);                                             \
        }                                                                                          \
      }                                                                                            \
    } else {                                                                                       \
      if (__bdd_active_config__) ++__bdd_active_config__->assertion_count;                         \
    }                                                                                              \
  } while (0)

/* Wrapper that captures __FILE__ and __LINE__ */
#define __BDD_CHECK__(condition, ...)                                                              \
  __BDD_CHECK_IMPL__(condition, __FILE__, __STRING__LINE__, __VA_ARGS__)

#define __BDD_CHECK_ONE__(condition)                                                               \
  __BDD_CHECK_IMPL__(condition, __FILE__, __STRING__LINE__, #condition)

#define check(...) __BDD_MACRO__(__BDD_CHECK_, __VA_ARGS__)

#define __BDD_INT_COMPARE__(emitter, actual, expected, op, fmt)                                    \
  do {                                                                                             \
    int __bdd_a__ = __BDD_CAST(int, (actual));                                                     \
    int __bdd_e__ = __BDD_CAST(int, (expected));                                                   \
    emitter(__bdd_a__ op __bdd_e__, fmt, __bdd_e__, __bdd_a__);                                   \
  } while (0)

#define __BDD_UINT_COMPARE__(emitter, actual, expected, op, fmt)                                   \
  do {                                                                                             \
    unsigned __bdd_a__ = __BDD_CAST(unsigned, (actual));                                           \
    unsigned __bdd_e__ = __BDD_CAST(unsigned, (expected));                                         \
    emitter(__bdd_a__ op __bdd_e__, fmt, __bdd_e__, __bdd_a__);                                   \
  } while (0)

#define __BDD_SIZE_COMPARE__(emitter, actual, expected, op, fmt)                                   \
  do {                                                                                             \
    size_t __bdd_a__ = __BDD_CAST(size_t, (actual));                                               \
    size_t __bdd_e__ = __BDD_CAST(size_t, (expected));                                             \
    emitter(__bdd_a__ op __bdd_e__, fmt, __bdd_e__, __bdd_a__);                                   \
  } while (0)

#define __BDD_LLONG_COMPARE__(emitter, actual, expected, op, fmt)                                  \
  do {                                                                                             \
    long long __bdd_a__ = __BDD_CAST(long long, (actual));                                         \
    long long __bdd_e__ = __BDD_CAST(long long, (expected));                                       \
    emitter(__bdd_a__ op __bdd_e__, fmt, __bdd_e__, __bdd_a__);                                   \
  } while (0)

#define __BDD_DOUBLE_COMPARE__(emitter, actual, expected, op, fmt)                                 \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    emitter(__bdd_a__ op __bdd_e__, fmt, __bdd_e__, __bdd_a__);                                   \
  } while (0)

/* --- Typed assertion macros --- */

/* Integer comparisons */
#define check_int_eq(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, ==, "expected %d but got %d")
#define check_int_eq_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, ==, "expected %d but got %d")

#define check_int_ne(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, !=, "expected != %d but got %d")
#define check_int_ne_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, !=, "expected != %d but got %d")

#define check_int_gt(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, >, "expected > %d but got %d")
#define check_int_gt_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, >, "expected > %d but got %d")

#define check_int_ge(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, >=, "expected >= %d but got %d")
#define check_int_ge_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, >=, "expected >= %d but got %d")

#define check_int_lt(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, <, "expected < %d but got %d")
#define check_int_lt_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, <, "expected < %d but got %d")

#define check_int_le(actual, expected)                                                             \
  __BDD_INT_COMPARE__(__BDD_CHECK__, actual, expected, <=, "expected <= %d but got %d")
#define check_int_le_warn(actual, expected)                                                        \
  __BDD_INT_COMPARE__(__BDD_WARN__, actual, expected, <=, "expected <= %d but got %d")

/* Unsigned integer comparisons */
#define check_uint_eq(actual, expected)                                                            \
  __BDD_UINT_COMPARE__(__BDD_CHECK__, actual, expected, ==, "expected %u but got %u")
#define check_uint_eq_warn(actual, expected)                                                       \
  __BDD_UINT_COMPARE__(__BDD_WARN__, actual, expected, ==, "expected %u but got %u")

#define check_uint_ne(actual, expected)                                                            \
  __BDD_UINT_COMPARE__(__BDD_CHECK__, actual, expected, !=, "expected != %u but got %u")
#define check_uint_ne_warn(actual, expected)                                                       \
  __BDD_UINT_COMPARE__(__BDD_WARN__, actual, expected, !=, "expected != %u but got %u")

/* Size_t comparisons */
#define check_size_eq(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, ==, "expected %zu but got %zu")
#define check_size_eq_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, ==, "expected %zu but got %zu")

#define check_size_ne(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, !=, "expected != %zu but got %zu")
#define check_size_ne_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, !=, "expected != %zu but got %zu")

#define check_size_gt(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, >, "expected > %zu but got %zu")
#define check_size_gt_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, >, "expected > %zu but got %zu")

#define check_size_ge(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, >=, "expected >= %zu but got %zu")
#define check_size_ge_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, >=, "expected >= %zu but got %zu")

#define check_size_lt(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, <, "expected < %zu but got %zu")
#define check_size_lt_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, <, "expected < %zu but got %zu")

#define check_size_le(actual, expected)                                                            \
  __BDD_SIZE_COMPARE__(__BDD_CHECK__, actual, expected, <=, "expected <= %zu but got %zu")
#define check_size_le_warn(actual, expected)                                                       \
  __BDD_SIZE_COMPARE__(__BDD_WARN__, actual, expected, <=, "expected <= %zu but got %zu")

/* Long / 64-bit comparisons */
#define check_long_eq(actual, expected)                                                            \
  __BDD_LLONG_COMPARE__(__BDD_CHECK__, actual, expected, ==, "expected %lld but got %lld")

/* Float/double comparisons with epsilon */
#define check_float_eq(actual, expected, epsilon)                                                  \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_eps__ = __BDD_CAST(double, (epsilon));                                            \
    __BDD_CHECK__(fabs(__bdd_a__ - __bdd_e__) <= __bdd_eps__,                                      \
                  "expected %f (+/- %f) but got %f", __bdd_e__, __bdd_eps__, __bdd_a__);          \
  } while (0)
#define check_float_eq_warn(actual, expected, epsilon)                                             \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_eps__ = __BDD_CAST(double, (epsilon));                                            \
    __BDD_WARN__(fabs(__bdd_a__ - __bdd_e__) <= __bdd_eps__,                                       \
                 "expected %f (+/- %f) but got %f", __bdd_e__, __bdd_eps__, __bdd_a__);           \
  } while (0)

#define check_float_ne(actual, expected, epsilon)                                                  \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_eps__ = __BDD_CAST(double, (epsilon));                                            \
    __BDD_CHECK__(fabs(__bdd_a__ - __bdd_e__) > __bdd_eps__,                                       \
                  "expected != %f (+/- %f) but got %f", __bdd_e__, __bdd_eps__, __bdd_a__);       \
  } while (0)
#define check_float_ne_warn(actual, expected, epsilon)                                             \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_eps__ = __BDD_CAST(double, (epsilon));                                            \
    __BDD_WARN__(fabs(__bdd_a__ - __bdd_e__) > __bdd_eps__,                                        \
                 "expected != %f (+/- %f) but got %f", __bdd_e__, __bdd_eps__, __bdd_a__);        \
  } while (0)

#define check_float_gt(actual, expected)                                                           \
  __BDD_DOUBLE_COMPARE__(__BDD_CHECK__, actual, expected, >, "expected > %f but got %f")
#define check_float_gt_warn(actual, expected)                                                      \
  __BDD_DOUBLE_COMPARE__(__BDD_WARN__, actual, expected, >, "expected > %f but got %f")

#define check_float_lt(actual, expected)                                                           \
  __BDD_DOUBLE_COMPARE__(__BDD_CHECK__, actual, expected, <, "expected < %f but got %f")
#define check_float_lt_warn(actual, expected)                                                      \
  __BDD_DOUBLE_COMPARE__(__BDD_WARN__, actual, expected, <, "expected < %f but got %f")

#define check_float_ge(actual, expected)                                                           \
  __BDD_DOUBLE_COMPARE__(__BDD_CHECK__, actual, expected, >=, "expected >= %f but got %f")
#define check_float_ge_warn(actual, expected)                                                      \
  __BDD_DOUBLE_COMPARE__(__BDD_WARN__, actual, expected, >=, "expected >= %f but got %f")

#define check_float_le(actual, expected)                                                           \
  __BDD_DOUBLE_COMPARE__(__BDD_CHECK__, actual, expected, <=, "expected <= %f but got %f")
#define check_float_le_warn(actual, expected)                                                      \
  __BDD_DOUBLE_COMPARE__(__BDD_WARN__, actual, expected, <=, "expected <= %f but got %f")

/* Relative tolerance: |actual - expected| <= rel * |expected| */
#define check_float_within_rel(actual, expected, rel)                                              \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_rel__ = __BDD_CAST(double, (rel));                                                \
    double __bdd_delta__ = fabs(__bdd_a__ - __bdd_e__);                                            \
    __BDD_CHECK__(__bdd_delta__ <= __bdd_rel__ * fabs(__bdd_e__),                                  \
                  "expected %f within %f%% of %f, delta was %f", __bdd_a__,                        \
                  __bdd_rel__ * 100.0, __bdd_e__, __bdd_delta__);                                  \
  } while (0)
#define check_float_within_rel_warn(actual, expected, rel)                                         \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    double __bdd_e__ = __BDD_CAST(double, (expected));                                             \
    double __bdd_rel__ = __BDD_CAST(double, (rel));                                                \
    double __bdd_delta__ = fabs(__bdd_a__ - __bdd_e__);                                            \
    __BDD_WARN__(__bdd_delta__ <= __bdd_rel__ * fabs(__bdd_e__),                                   \
                 "expected %f within %f%% of %f, delta was %f", __bdd_a__,                         \
                 __bdd_rel__ * 100.0, __bdd_e__, __bdd_delta__);                                   \
  } while (0)

/* Absolute tolerance — same as check_float_eq but named for clarity
 */
#define check_float_within_abs(actual, expected, margin) check_float_eq(actual, expected, margin)
#define check_float_within_abs_warn(actual, expected, margin)                                      \
  check_float_eq_warn(actual, expected, margin)

/* Double precision comparisons (aliases to float/double checks with correct casting) */
#define check_double_eq(actual, expected, epsilon) check_float_eq(actual, expected, epsilon)
#define check_double_eq_warn(actual, expected, epsilon)                                            \
  check_float_eq_warn(actual, expected, epsilon)

#define check_double_ne(actual, expected, epsilon) check_float_ne(actual, expected, epsilon)
#define check_double_ne_warn(actual, expected, epsilon)                                            \
  check_float_ne_warn(actual, expected, epsilon)

#define check_double_gt(actual, expected) check_float_gt(actual, expected)
#define check_double_gt_warn(actual, expected) check_float_gt_warn(actual, expected)

#define check_double_lt(actual, expected) check_float_lt(actual, expected)
#define check_double_lt_warn(actual, expected) check_float_lt_warn(actual, expected)

#define check_double_ge(actual, expected) check_float_ge(actual, expected)
#define check_double_ge_warn(actual, expected) check_float_ge_warn(actual, expected)

#define check_double_le(actual, expected) check_float_le(actual, expected)
#define check_double_le_warn(actual, expected) check_float_le_warn(actual, expected)

#define check_double_within_rel(actual, expected, rel) check_float_within_rel(actual, expected, rel)
#define check_double_within_rel_warn(actual, expected, rel)                                        \
  check_float_within_rel_warn(actual, expected, rel)

#define check_double_within_abs(actual, expected, margin)                                          \
  check_float_within_abs(actual, expected, margin)
#define check_double_within_abs_warn(actual, expected, margin)                                     \
  check_float_within_abs_warn(actual, expected, margin)

/* String comparisons */
static inline int __bdd_str_eq__(const char *a, const char *b) {
  return a && b && strcmp(a, b) == 0;
}
static inline int __bdd_str_ne__(const char *a, const char *b) {
  return !a || !b || strcmp(a, b) != 0;
}
static inline int __bdd_str_contains__(const char *haystack, const char *needle) {
  return haystack && needle && strstr(haystack, needle) != NULL;
}
static inline int __bdd_str_starts_with__(const char *s, const char *prefix) {
  return s && prefix && strncmp(s, prefix, strlen(prefix)) == 0;
}
static inline int __bdd_str_ends_with__(const char *s, const char *suffix) {
  size_t slen;
  size_t suflen;
  if (!s || !suffix) return 0;
  slen = strlen(s);
  suflen = strlen(suffix);
  return slen >= suflen && strcmp(s + slen - suflen, suffix) == 0;
}

static inline const char *__bdd_cstr_or_null__(const char *s) { return s ? s : "(null)"; }

#define check_str_eq(actual, expected)                                                             \
  do {                                                                                             \
    const char *__bdd_a__ = (const char *)(actual);                                                \
    const char *__bdd_e__ = (const char *)(expected);                                              \
    __BDD_CHECK__(__bdd_str_eq__(__bdd_a__, __bdd_e__), "expected \"%s\" but got \"%s\"",          \
                  __bdd_cstr_or_null__(__bdd_e__), __bdd_cstr_or_null__(__bdd_a__));               \
  } while (0)
#define check_str_eq_warn(actual, expected)                                                        \
  do {                                                                                             \
    const char *__bdd_a__ = (const char *)(actual);                                                \
    const char *__bdd_e__ = (const char *)(expected);                                              \
    __BDD_WARN__(__bdd_str_eq__(__bdd_a__, __bdd_e__), "expected \"%s\" but got \"%s\"",           \
                 __bdd_cstr_or_null__(__bdd_e__), __bdd_cstr_or_null__(__bdd_a__));                \
  } while (0)

#define check_str_ne(actual, expected)                                                             \
  do {                                                                                             \
    const char *__bdd_a__ = (const char *)(actual);                                                \
    const char *__bdd_e__ = (const char *)(expected);                                              \
    __BDD_CHECK__(__bdd_str_ne__(__bdd_a__, __bdd_e__), "expected != \"%s\" but got \"%s\"",       \
                  __bdd_cstr_or_null__(__bdd_e__), __bdd_cstr_or_null__(__bdd_a__));               \
  } while (0)
#define check_str_ne_warn(actual, expected)                                                        \
  do {                                                                                             \
    const char *__bdd_a__ = (const char *)(actual);                                                \
    const char *__bdd_e__ = (const char *)(expected);                                              \
    __BDD_WARN__(__bdd_str_ne__(__bdd_a__, __bdd_e__), "expected != \"%s\" but got \"%s\"",        \
                 __bdd_cstr_or_null__(__bdd_e__), __bdd_cstr_or_null__(__bdd_a__));                \
  } while (0)

#define check_str_contains(haystack, needle)                                                       \
  do {                                                                                             \
    const char *__bdd_h__ = (const char *)(haystack);                                              \
    const char *__bdd_n__ = (const char *)(needle);                                                \
    __BDD_CHECK__(__bdd_str_contains__(__bdd_h__, __bdd_n__), "expected \"%s\" to contain \"%s\"", \
                  __bdd_cstr_or_null__(__bdd_h__), __bdd_cstr_or_null__(__bdd_n__));               \
  } while (0)
#define check_str_contains_warn(haystack, needle)                                                  \
  do {                                                                                             \
    const char *__bdd_h__ = (const char *)(haystack);                                              \
    const char *__bdd_n__ = (const char *)(needle);                                                \
    __BDD_WARN__(__bdd_str_contains__(__bdd_h__, __bdd_n__), "expected \"%s\" to contain \"%s\"",  \
                 __bdd_cstr_or_null__(__bdd_h__), __bdd_cstr_or_null__(__bdd_n__));                \
  } while (0)

#define check_str_starts_with(str, prefix)                                                         \
  do {                                                                                             \
    const char *__bdd_s__ = (const char *)(str);                                                   \
    const char *__bdd_p__ = (const char *)(prefix);                                                \
    __BDD_CHECK__(__bdd_str_starts_with__(__bdd_s__, __bdd_p__),                                   \
                  "expected \"%s\" to start with \"%s\"", __bdd_cstr_or_null__(__bdd_s__),         \
                  __bdd_cstr_or_null__(__bdd_p__));                                                \
  } while (0)
#define check_str_starts_with_warn(str, prefix)                                                    \
  do {                                                                                             \
    const char *__bdd_s__ = (const char *)(str);                                                   \
    const char *__bdd_p__ = (const char *)(prefix);                                                \
    __BDD_WARN__(__bdd_str_starts_with__(__bdd_s__, __bdd_p__),                                    \
                 "expected \"%s\" to start with \"%s\"", __bdd_cstr_or_null__(__bdd_s__),          \
                 __bdd_cstr_or_null__(__bdd_p__));                                                 \
  } while (0)

#define check_str_ends_with(str, suffix)                                                           \
  do {                                                                                             \
    const char *__bdd_s__ = (const char *)(str);                                                   \
    const char *__bdd_x__ = (const char *)(suffix);                                                \
    __BDD_CHECK__(__bdd_str_ends_with__(__bdd_s__, __bdd_x__), "expected \"%s\" to end with \"%s\"",\
                  __bdd_cstr_or_null__(__bdd_s__), __bdd_cstr_or_null__(__bdd_x__));               \
  } while (0)
#define check_str_ends_with_warn(str, suffix)                                                      \
  do {                                                                                             \
    const char *__bdd_s__ = (const char *)(str);                                                   \
    const char *__bdd_x__ = (const char *)(suffix);                                                \
    __BDD_WARN__(__bdd_str_ends_with__(__bdd_s__, __bdd_x__),                                      \
                 "expected \"%s\" to end with \"%s\"", __bdd_cstr_or_null__(__bdd_s__),            \
                 __bdd_cstr_or_null__(__bdd_x__));                                                 \
  } while (0)

/* Memory comparisons */
#define check_mem_eq(actual, expected, len)                                                        \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    size_t __bdd_n__ = __BDD_CAST(size_t, (len));                                                  \
    __BDD_CHECK__(memcmp(__bdd_a__, __bdd_e__, __bdd_n__) == 0, "memory mismatch at %zu bytes",   \
                  __bdd_n__);                                                                       \
  } while (0)
#define check_mem_eq_warn(actual, expected, len)                                                   \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    size_t __bdd_n__ = __BDD_CAST(size_t, (len));                                                  \
    __BDD_WARN__(memcmp(__bdd_a__, __bdd_e__, __bdd_n__) == 0, "memory mismatch at %zu bytes",    \
                 __bdd_n__);                                                                        \
  } while (0)

#define check_mem_ne(actual, expected, len)                                                        \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    size_t __bdd_n__ = __BDD_CAST(size_t, (len));                                                  \
    __BDD_CHECK__(memcmp(__bdd_a__, __bdd_e__, __bdd_n__) != 0,                                    \
                  "expected memory to differ at %zu bytes", __bdd_n__);                            \
  } while (0)
#define check_mem_ne_warn(actual, expected, len)                                                   \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    size_t __bdd_n__ = __BDD_CAST(size_t, (len));                                                  \
    __BDD_WARN__(memcmp(__bdd_a__, __bdd_e__, __bdd_n__) != 0,                                     \
                 "expected memory to differ at %zu bytes", __bdd_n__);                             \
  } while (0)

/* Pointer checks */
#define check_not_null(ptr)                                                                        \
  do {                                                                                             \
    const void *__bdd_ptr__ = (const void *)(ptr);                                                 \
    __BDD_CHECK__(__bdd_ptr__ != NULL, "expected non-null but got NULL");                          \
  } while (0)
#define check_not_null_warn(ptr)                                                                   \
  do {                                                                                             \
    const void *__bdd_ptr__ = (const void *)(ptr);                                                 \
    __BDD_WARN__(__bdd_ptr__ != NULL, "expected non-null but got NULL");                           \
  } while (0)

#define check_null(ptr)                                                                            \
  do {                                                                                             \
    const void *__bdd_ptr__ = (const void *)(ptr);                                                 \
    __BDD_CHECK__(__bdd_ptr__ == NULL, "expected NULL but got %p", __bdd_ptr__);                   \
  } while (0)
#define check_null_warn(ptr)                                                                       \
  do {                                                                                             \
    const void *__bdd_ptr__ = (const void *)(ptr);                                                 \
    __BDD_WARN__(__bdd_ptr__ == NULL, "expected NULL but got %p", __bdd_ptr__);                    \
  } while (0)

/* Hex comparisons */
#define check_hex_eq(actual, expected)                                                             \
  __BDD_UINT_COMPARE__(__BDD_CHECK__, actual, expected, ==, "expected 0x%x but got 0x%x")

#define check_hex64_eq(actual, expected)                                                           \
  do {                                                                                             \
    unsigned long long __bdd_a__ = __BDD_CAST(unsigned long long, (actual));                       \
    unsigned long long __bdd_e__ = __BDD_CAST(unsigned long long, (expected));                     \
    __BDD_CHECK__(__bdd_a__ == __bdd_e__, "expected 0x%llx but got 0x%llx", __bdd_e__, __bdd_a__);\
  } while (0)

/* Boolean assertions */
#define check_true(actual) __BDD_CHECK__((actual), "expected true but got false")

#define check_false(actual) __BDD_CHECK__(!(actual), "expected false but got true")

/* Pointer address comparisons */
#define check_ptr_eq(actual, expected)                                                             \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    __BDD_CHECK__(__bdd_a__ == __bdd_e__, "expected %p but got %p", __bdd_e__, __bdd_a__);        \
  } while (0)

#define check_ptr_ne(actual, expected)                                                             \
  do {                                                                                             \
    const void *__bdd_a__ = (const void *)(actual);                                                \
    const void *__bdd_e__ = (const void *)(expected);                                              \
    __BDD_CHECK__(__bdd_a__ != __bdd_e__, "expected != %p but got %p", __bdd_e__, __bdd_a__);     \
  } while (0)

/* Range assertion: lo <= actual <= hi */
#define check_int_range(actual, lo, hi)                                                            \
  do {                                                                                             \
    int __bdd_a__ = __BDD_CAST(int, (actual));                                                     \
    int __bdd_lo__ = __BDD_CAST(int, (lo));                                                        \
    int __bdd_hi__ = __BDD_CAST(int, (hi));                                                        \
    __BDD_CHECK__(__bdd_a__ >= __bdd_lo__ && __bdd_a__ <= __bdd_hi__,                              \
                  "expected %d in range [%d, %d]", __bdd_a__, __bdd_lo__, __bdd_hi__);            \
  } while (0)

/* Float special value assertions */
#define check_float_nan(actual)                                                                    \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    __BDD_CHECK__(isnan(__bdd_a__), "expected NaN but got %f", __bdd_a__);                         \
  } while (0)
#define check_float_nan_warn(actual)                                                               \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    __BDD_WARN__(isnan(__bdd_a__), "expected NaN but got %f", __bdd_a__);                          \
  } while (0)

#define check_float_inf(actual)                                                                    \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    __BDD_CHECK__(isinf(__bdd_a__), "expected Inf but got %f", __bdd_a__);                         \
  } while (0)
#define check_float_inf_warn(actual)                                                               \
  do {                                                                                             \
    double __bdd_a__ = __BDD_CAST(double, (actual));                                               \
    __BDD_WARN__(isinf(__bdd_a__), "expected Inf but got %f", __bdd_a__);                          \
  } while (0)

#define check_double_nan(actual) check_float_nan(actual)
#define check_double_nan_warn(actual) check_float_nan_warn(actual)

#define check_double_inf(actual) check_float_inf(actual)
#define check_double_inf_warn(actual) check_float_inf_warn(actual)

/* Bitmask assertion: (val & mask) == mask */
#define check_bits(actual, mask)                                                                   \
  do {                                                                                             \
    unsigned __bdd_a__ = __BDD_CAST(unsigned, (actual));                                           \
    unsigned __bdd_m__ = __BDD_CAST(unsigned, (mask));                                             \
    unsigned __bdd_got__ = __bdd_a__ & __bdd_m__;                                                  \
    __BDD_CHECK__(__bdd_got__ == __bdd_m__, "expected bits 0x%x set in 0x%x, got 0x%x",           \
                  __bdd_m__, __bdd_a__, __bdd_got__);                                              \
  } while (0)

/* --- Array comparison helpers --- */

static inline bool __bdd_int_array_eq__(const int *actual, const int *expected, size_t n,
                                        size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    if (actual[i] != expected[i]) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

static inline bool __bdd_uint8_array_eq__(const unsigned char *actual,
                                          const unsigned char *expected, size_t n,
                                          size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    if (actual[i] != expected[i]) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

static inline bool __bdd_size_array_eq__(const size_t *actual, const size_t *expected, size_t n,
                                         size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    if (actual[i] != expected[i]) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

static inline bool __bdd_float_array_eq__(const double *actual, const double *expected, size_t n,
                                          double eps, size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    double d = actual[i] - expected[i];
    if (d > eps || d < -eps) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

static inline bool __bdd_ptr_array_eq__(const void *const *actual, const void *const *expected,
                                        size_t n, size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    if (actual[i] != expected[i]) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

static inline bool __bdd_str_array_eq__(const char *const *actual, const char *const *expected,
                                        size_t n, size_t *fail_idx) {
  for (size_t i = 0; i < n; i++) {
    if (actual[i] == NULL && expected[i] == NULL) continue;
    if (actual[i] == NULL || expected[i] == NULL || strcmp(actual[i], expected[i]) != 0) {
      *fail_idx = i;
      return false;
    }
  }
  return true;
}

/* --- Array assertion macros --- */

#define check_int_array_eq(actual, expected, n)                                                    \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_int_array_eq__((const int *)(actual), (const int *)(expected), (n), &__bdd_fi__)) { \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected %d but got %d", __bdd_fi__,              \
                    ((const int *)(expected))[__bdd_fi__], ((const int *)(actual))[__bdd_fi__]);   \
    }                                                                                              \
  } while (0)
#define check_int_array_eq_warn(actual, expected, n)                                               \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_int_array_eq__((const int *)(actual), (const int *)(expected), (n), &__bdd_fi__)) { \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected %d but got %d", __bdd_fi__,               \
                   ((const int *)(expected))[__bdd_fi__], ((const int *)(actual))[__bdd_fi__]);    \
    }                                                                                              \
  } while (0)

#define check_uint8_array_eq(actual, expected, n)                                                  \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_uint8_array_eq__((const unsigned char *)(actual),                                   \
                                (const unsigned char *)(expected), (n), &__bdd_fi__)) {            \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected 0x%02x but got 0x%02x", __bdd_fi__,      \
                    ((const unsigned char *)(expected))[__bdd_fi__],                               \
                    ((const unsigned char *)(actual))[__bdd_fi__]);                                \
    }                                                                                              \
  } while (0)
#define check_uint8_array_eq_warn(actual, expected, n)                                             \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_uint8_array_eq__((const unsigned char *)(actual),                                   \
                                (const unsigned char *)(expected), (n), &__bdd_fi__)) {            \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected 0x%02x but got 0x%02x", __bdd_fi__,       \
                   ((const unsigned char *)(expected))[__bdd_fi__],                                \
                   ((const unsigned char *)(actual))[__bdd_fi__]);                                 \
    }                                                                                              \
  } while (0)

#define check_size_array_eq(actual, expected, n)                                                   \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_size_array_eq__((const size_t *)(actual), (const size_t *)(expected), (n),          \
                               &__bdd_fi__)) {                                                     \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected %zu but got %zu", __bdd_fi__,            \
                    ((const size_t *)(expected))[__bdd_fi__],                                      \
                    ((const size_t *)(actual))[__bdd_fi__]);                                       \
    }                                                                                              \
  } while (0)
#define check_size_array_eq_warn(actual, expected, n)                                              \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_size_array_eq__((const size_t *)(actual), (const size_t *)(expected), (n),          \
                               &__bdd_fi__)) {                                                     \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected %zu but got %zu", __bdd_fi__,             \
                   ((const size_t *)(expected))[__bdd_fi__],                                       \
                   ((const size_t *)(actual))[__bdd_fi__]);                                        \
    }                                                                                              \
  } while (0)

#define check_float_array_eq(actual, expected, n, epsilon)                                         \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_float_array_eq__((const double *)(actual), (const double *)(expected), (n),         \
                                (epsilon), &__bdd_fi__)) {                                         \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected %f but got %f (+/- %f)", __bdd_fi__,     \
                    ((const double *)(expected))[__bdd_fi__],                                      \
                    ((const double *)(actual))[__bdd_fi__], __BDD_CAST(double, (epsilon)));        \
    }                                                                                              \
  } while (0)
#define check_float_array_eq_warn(actual, expected, n, epsilon)                                    \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_float_array_eq__((const double *)(actual), (const double *)(expected), (n),         \
                                (epsilon), &__bdd_fi__)) {                                         \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected %f but got %f (+/- %f)", __bdd_fi__,      \
                   ((const double *)(expected))[__bdd_fi__],                                       \
                   ((const double *)(actual))[__bdd_fi__], __BDD_CAST(double, (epsilon)));         \
    }                                                                                              \
  } while (0)

#define check_double_array_eq(actual, expected, n, epsilon)                                        \
  check_float_array_eq(actual, expected, n, epsilon)
#define check_double_array_eq_warn(actual, expected, n, epsilon)                                   \
  check_float_array_eq_warn(actual, expected, n, epsilon)

#define check_ptr_array_eq(actual, expected, n)                                                    \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_ptr_array_eq__((const void *const *)(actual), (const void *const *)(expected), (n), \
                              &__bdd_fi__)) {                                                      \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected %p but got %p", __bdd_fi__,              \
                    ((const void *const *)(expected))[__bdd_fi__],                                 \
                    ((const void *const *)(actual))[__bdd_fi__]);                                  \
    }                                                                                              \
  } while (0)
#define check_ptr_array_eq_warn(actual, expected, n)                                               \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_ptr_array_eq__((const void *const *)(actual), (const void *const *)(expected), (n), \
                              &__bdd_fi__)) {                                                      \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected %p but got %p", __bdd_fi__,               \
                   ((const void *const *)(expected))[__bdd_fi__],                                  \
                   ((const void *const *)(actual))[__bdd_fi__]);                                   \
    }                                                                                              \
  } while (0)

#define check_str_array_eq(actual, expected, n)                                                    \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_str_array_eq__((const char *const *)(actual), (const char *const *)(expected), (n), \
                              &__bdd_fi__)) {                                                      \
      __BDD_CHECK__(0, "array mismatch at [%zu]: expected \"%s\" but got \"%s\"", __bdd_fi__,      \
                    ((const char *const *)(expected))[__bdd_fi__]                                  \
                        ? ((const char *const *)(expected))[__bdd_fi__]                            \
                        : "(null)",                                                                \
                    ((const char *const *)(actual))[__bdd_fi__]                                    \
                        ? ((const char *const *)(actual))[__bdd_fi__]                              \
                        : "(null)");                                                               \
    }                                                                                              \
  } while (0)
#define check_str_array_eq_warn(actual, expected, n)                                               \
  do {                                                                                             \
    size_t __bdd_fi__ = 0;                                                                         \
    if (!__bdd_str_array_eq__((const char *const *)(actual), (const char *const *)(expected), (n), \
                              &__bdd_fi__)) {                                                      \
      __BDD_WARN__(0, "array mismatch at [%zu]: expected \"%s\" but got \"%s\"", __bdd_fi__,       \
                   ((const char *const *)(expected))[__bdd_fi__]                                   \
                       ? ((const char *const *)(expected))[__bdd_fi__]                             \
                       : "(null)",                                                                 \
                   ((const char *const *)(actual))[__bdd_fi__]                                     \
                       ? ((const char *const *)(actual))[__bdd_fi__]                               \
                       : "(null)");                                                                \
    }                                                                                              \
  } while (0)

/* --- Non-fatal assertion --- */
/* Internal implementation that takes file and line */
#define __BDD_WARN_IMPL__(condition, file, line, ...)                                              \
  do {                                                                                             \
    if (__bdd_active_config__) {                                                                   \
      if (!__bdd_eval_bool__(!!(condition))) {                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
        char *__bdd_message__ = __bdd_format__(__VA_ARGS__);                                       \
        ++__bdd_active_config__->warn_count;                                                       \
        __bdd_indent__(stdout, __bdd_active_config__->current_test                                 \
                                   ? __bdd_active_config__->current_test->level + 1                \
                                   : 1);                                                           \
        if (__bdd_active_config__->use_color) {                                                    \
          printf(__BDD_COLOR_YELLOW__ "Warning:" __BDD_COLOR_RESET__ " %s", __bdd_message__);      \
        } else {                                                                                   \
          printf("Warning: %s", __bdd_message__);                                                  \
        }                                                                                          \
        printf(" at %s:%s\n", file, line);                                                         \
        free(__bdd_message__);                                                                     \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    }                                                                                              \
  } while (0)

/* Wrapper that captures __FILE__ and __LINE__ */
#define __BDD_WARN__(condition, ...)                                                               \
  __BDD_WARN_IMPL__(condition, __FILE__, __STRING__LINE__, __VA_ARGS__)

#define __BDD_WARN_ONE__(condition)                                                                \
  __BDD_WARN_IMPL__(condition, __FILE__, __STRING__LINE__, #condition)

#define check_warn(...) __BDD_MACRO__(__BDD_WARN_, __VA_ARGS__)

/* --- Info context --- */
#define info(...)                                                                                  \
  do {                                                                                             \
    char *__bdd_info_msg__ = __bdd_format__(__VA_ARGS__);                                          \
    size_t __bdd_info_msg_len__ = strlen(__bdd_info_msg__);                                        \
    if (__bdd_active_config__->info_len + __bdd_info_msg_len__ + 2 <                               \
        sizeof(__bdd_active_config__->info_buffer)) {                                              \
      if (__bdd_active_config__->info_len > 0) {                                                   \
        __bdd_active_config__->info_buffer[__bdd_active_config__->info_len++] = ' ';               \
      }                                                                                            \
      memcpy(__bdd_active_config__->info_buffer + __bdd_active_config__->info_len,                 \
             __bdd_info_msg__, __bdd_info_msg_len__);                                              \
      __bdd_active_config__->info_len += __bdd_info_msg_len__;                                     \
      __bdd_active_config__->info_buffer[__bdd_active_config__->info_len] = '\0';                  \
    }                                                                                              \
    free(__bdd_info_msg__);                                                                        \
  } while (0)

#define capture(var, fmt) info(#var "=" fmt, (var))

/* --- BDD keywords (given/when/then) --- */
#define given(...) describe("Given " __VA_ARGS__)
#define when(...) describe("When " __VA_ARGS__)
#define then(...) it("Then " __VA_ARGS__)
#define and_given(...) describe("And given " __VA_ARGS__)
#define and_when(...) describe("And when " __VA_ARGS__)
#define and_then(...) it("And then " __VA_ARGS__)

/* --- Section (TDD SECTION equivalent) --- */
#define section(...) describe(__VA_ARGS__)

/* --- Benchmarking --- */
/* Usage:
 *   benchmark("case", iterations, scale) { code; }
 */
/* Compatibility shim: benchmark titles are no longer configurable. */
#define benchmark_titles(name_title, input_title, iters_title, avg_title, ns_title, min_title,    \
                         max_title, ops_title, size_title, bw_title)                               \
  if (1)

#define benchmark(title, iters, scale)                                                             \
  for (                                                                                            \
      struct {                                                                                     \
        int __done;                                                                                \
        size_t __n;                                                                                \
        double __min;                                                                              \
        double __max;                                                                              \
        double __sum;                                                                              \
        double __scale;                                                                            \
      } __bdd_bm__ = {0, __BDD_CAST(size_t, (iters)), 1e18, 0.0, 0.0,                              \
                      __BDD_CAST(double, (scale))};                                                \
      !__bdd_bm__.__done; __bdd_bm__.__done = 1,                                                   \
        __bdd_bench_print__(                                                                       \
            __bdd_active_config__, (title), __bdd_bm__.__n, __bdd_bm__.__sum, __bdd_bm__.__min,  \
            __bdd_bm__.__max, __bdd_bm__.__scale,                                                 \
            __bdd_active_config__->current_test ? __bdd_active_config__->current_test->level + 1  \
                                                : 1,                                               \
            __bdd_active_config__->use_color))                                                     \
    for (size_t __bdd_bm_i__ = 0; __bdd_bm_i__ < __bdd_bm__.__n; ++__bdd_bm_i__)                   \
      for (double __bdd_bm_t0__ = __bdd_get_time_ms__(), __bdd_bm_t1__ = 0; __bdd_bm_t1__ == 0;    \
           __bdd_bm_t1__ = __bdd_get_time_ms__() - __bdd_bm_t0__,                                  \
                  __bdd_bm__.__sum += __bdd_bm_t1__,                                               \
                  __bdd_bm__.__min = __bdd_bm_t1__ < __bdd_bm__.__min ? __bdd_bm_t1__              \
                                                                      : __bdd_bm__.__min,          \
                  __bdd_bm__.__max = __bdd_bm_t1__ > __bdd_bm__.__max ? __bdd_bm_t1__              \
                                                                      : __bdd_bm__.__max)

/* --- C++ container assertions (only available in C++ mode) --- */
#ifdef __cplusplus
  #include <algorithm>
  #include <iterator>
  #include <sstream>
  #include <stdexcept>

namespace __bdd_cpp__ {

  template <typename T, typename U> inline bool __bdd_equal__(const T &a, const U &b) {
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"
  #endif
    return a == b;
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
  #endif
  }

  template <typename T, typename U> inline bool __bdd_not_equal__(const T &a, const U &b) {
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"
  #endif
    return a != b;
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
  #endif
  }

  template <typename Container>
  std::string container_to_string(const Container &c, size_t max_items = 8) {
    std::ostringstream os;
    os << "[";
    size_t i = 0;
    for (auto it = c.begin(); it != c.end() && i < max_items; ++it, ++i) {
      if (i > 0) os << ", ";
      os << *it;
    }
    if (c.size() > max_items) os << ", ...(" << c.size() << " total)";
    os << "]";
    return os.str();
  }

  template <typename Container>
  bool containers_equal(const Container &actual, const Container &expected, size_t &fail_idx,
                        bool &size_mismatch) {
    size_mismatch = (actual.size() != expected.size());
    if (size_mismatch) return false;
    auto a = actual.begin();
    auto e = expected.begin();
    for (fail_idx = 0; a != actual.end(); ++a, ++e, ++fail_idx) {
      if (!(*a == *e)) return false;
    }
    return true;
  }

  template <typename Map>
  bool maps_equal(const Map &actual, const Map &expected, std::string &detail) {
    if (actual.size() != expected.size()) {
      std::ostringstream os;
      os << "size mismatch: expected " << expected.size() << " but got " << actual.size();
      detail = os.str();
      return false;
    }
    for (auto it = expected.begin(); it != expected.end(); ++it) {
      auto found = actual.find(it->first);
      if (found == actual.end()) {
        std::ostringstream os;
        os << "missing key: " << it->first;
        detail = os.str();
        return false;
      }
      if (!(found->second == it->second)) {
        std::ostringstream os;
        os << "value mismatch at key " << it->first << ": expected " << it->second << " but got "
           << found->second;
        detail = os.str();
        return false;
      }
    }
    return true;
  }

} /* namespace __bdd_cpp__ */

  #define check_eq(actual, expected)                                                               \
    do {                                                                                           \
      size_t __bdd_fi__ = 0;                                                                       \
      bool __bdd_sm__ = false;                                                                     \
      if (!__bdd_cpp__::containers_equal((actual), (expected), __bdd_fi__, __bdd_sm__)) {          \
        if (__bdd_sm__) {                                                                          \
          std::string __a = __bdd_cpp__::container_to_string((actual));                            \
          std::string __e = __bdd_cpp__::container_to_string((expected));                          \
          __BDD_CHECK__(                                                                           \
              0,                                                                                   \
              "size mismatch: expected %zu elements but got %zu\n  expected: %s\n  actual:   %s",  \
              (expected).size(), (actual).size(), __e.c_str(), __a.c_str());                       \
        } else {                                                                                   \
          __BDD_CHECK__(0, "mismatch at index %zu", __bdd_fi__);                                   \
        }                                                                                          \
      }                                                                                            \
    } while (0)
  #define check_eq_warn(actual, expected)                                                          \
    do {                                                                                           \
      size_t __bdd_fi__ = 0;                                                                       \
      bool __bdd_sm__ = false;                                                                     \
      if (!__bdd_cpp__::containers_equal((actual), (expected), __bdd_fi__, __bdd_sm__)) {          \
        if (__bdd_sm__) {                                                                          \
          std::string __a = __bdd_cpp__::container_to_string((actual));                            \
          std::string __e = __bdd_cpp__::container_to_string((expected));                          \
          __BDD_WARN__(                                                                            \
              0,                                                                                   \
              "size mismatch: expected %zu elements but got %zu\n  expected: %s\n  actual:   %s",  \
              (expected).size(), (actual).size(), __e.c_str(), __a.c_str());                       \
        } else {                                                                                   \
          __BDD_WARN__(0, "mismatch at index %zu", __bdd_fi__);                                    \
        }                                                                                          \
      }                                                                                            \
    } while (0)

  #define check_map_eq(actual, expected)                                                           \
    do {                                                                                           \
      std::string __bdd_detail__;                                                                  \
      if (!__bdd_cpp__::maps_equal((actual), (expected), __bdd_detail__)) {                        \
        __BDD_CHECK__(0, "%s", __bdd_detail__.c_str());                                            \
      }                                                                                            \
    } while (0)
  #define check_map_eq_warn(actual, expected)                                                      \
    do {                                                                                           \
      std::string __bdd_detail__;                                                                  \
      if (!__bdd_cpp__::maps_equal((actual), (expected), __bdd_detail__)) {                        \
        __BDD_WARN__(0, "%s", __bdd_detail__.c_str());                                             \
      }                                                                                            \
    } while (0)

  #define check_contains(container, value)                                                         \
    __BDD_CHECK__(std::find((container).begin(), (container).end(), (value)) != (container).end(), \
                  "container does not contain expected value")
  #define check_contains_warn(container, value)                                                    \
    __BDD_WARN__(std::find((container).begin(), (container).end(), (value)) != (container).end(),  \
                 "container does not contain expected value")

  #define check_not_contains(container, value)                                                     \
    __BDD_CHECK__(std::find((container).begin(), (container).end(), (value)) == (container).end(), \
                  "container contains unexpected value")
  #define check_not_contains_warn(container, value)                                                \
    __BDD_WARN__(std::find((container).begin(), (container).end(), (value)) == (container).end(),  \
                 "container contains unexpected value")

  #define check_size(container, expected_size)                                                     \
    __BDD_CHECK__((container).size() == __BDD_CAST(size_t, (expected_size)),                       \
                  "expected size %zu but got %zu", __BDD_CAST(size_t, (expected_size)),            \
                  __BDD_CAST(size_t, (container).size()))
  #define check_size_warn(container, expected_size)                                                \
    __BDD_WARN__((container).size() == __BDD_CAST(size_t, (expected_size)),                        \
                 "expected size %zu but got %zu", __BDD_CAST(size_t, (expected_size)),             \
                 __BDD_CAST(size_t, (container).size()))

  #define check_empty(container)                                                                   \
    __BDD_CHECK__((container).empty(), "expected empty but got %zu elements",                      \
                  __BDD_CAST(size_t, (container).size()))
  #define check_empty_warn(container)                                                              \
    __BDD_WARN__((container).empty(), "expected empty but got %zu elements",                       \
                 __BDD_CAST(size_t, (container).size()))

  #define check_not_empty(container)                                                               \
    __BDD_CHECK__(!(container).empty(), "expected non-empty container")
  #define check_not_empty_warn(container)                                                          \
    __BDD_WARN__(!(container).empty(), "expected non-empty container")

  #define check_map_has_key(map, key)                                                              \
    __BDD_CHECK__((map).find((key)) != (map).end(), "map does not contain expected key")
  #define check_map_has_key_warn(map, key)                                                         \
    __BDD_WARN__((map).find((key)) != (map).end(), "map does not contain expected key")

  #define check_map_not_has_key(map, key)                                                          \
    __BDD_CHECK__((map).find((key)) == (map).end(), "map contains unexpected key")
  #define check_map_not_has_key_warn(map, key)                                                     \
    __BDD_WARN__((map).find((key)) == (map).end(), "map contains unexpected key")

/* std::string assertions */
namespace __bdd_cpp__ {

  template <typename T> std::string to_string_safe(const T &val) {
    std::ostringstream os;
    os << val;
    return os.str();
  }

  /* Safely convert to std::string for assertions, handling NULL pointers */
  inline std::string stringify_safe(const char *s) { return s ? s : "(null)"; }
  inline std::string stringify_safe(const std::string &s) { return s; }
  inline std::string stringify_safe(std::nullptr_t) { return "(null)"; }

} /* namespace __bdd_cpp__ */

  #define check_string_eq(actual, expected)                                                        \
    do {                                                                                           \
      std::string __a = __bdd_cpp__::stringify_safe(actual);                                       \
      std::string __e = __bdd_cpp__::stringify_safe(expected);                                     \
      __BDD_CHECK__(__a == __e, "expected \"%s\" but got \"%s\"", __e.c_str(), __a.c_str());       \
    } while (0)
  #define check_string_eq_warn(actual, expected)                                                   \
    do {                                                                                           \
      std::string __a = __bdd_cpp__::stringify_safe(actual);                                       \
      std::string __e = __bdd_cpp__::stringify_safe(expected);                                     \
      __BDD_WARN__(__a == __e, "expected \"%s\" but got \"%s\"", __e.c_str(), __a.c_str());        \
    } while (0)

  #define check_string_ne(actual, expected)                                                        \
    do {                                                                                           \
      std::string __a = __bdd_cpp__::stringify_safe(actual);                                       \
      std::string __e = __bdd_cpp__::stringify_safe(expected);                                     \
      __BDD_CHECK__(__a != __e, "expected != \"%s\" but got \"%s\"", __e.c_str(), __a.c_str());    \
    } while (0)
  #define check_string_ne_warn(actual, expected)                                                   \
    do {                                                                                           \
      std::string __a = __bdd_cpp__::stringify_safe(actual);                                       \
      std::string __e = __bdd_cpp__::stringify_safe(expected);                                     \
      __BDD_WARN__(__a != __e, "expected != \"%s\" but got \"%s\"", __e.c_str(), __a.c_str());     \
    } while (0)

  #define check_string_contains(haystack, needle)                                                  \
    do {                                                                                           \
      std::string __h = __bdd_cpp__::stringify_safe(haystack);                                     \
      std::string __n = __bdd_cpp__::stringify_safe(needle);                                       \
      __BDD_CHECK__(__h.find(__n) != std::string::npos, "expected \"%s\" to contain \"%s\"",       \
                    __h.c_str(), __n.c_str());                                                     \
    } while (0)
  #define check_string_contains_warn(haystack, needle)                                             \
    do {                                                                                           \
      std::string __h = __bdd_cpp__::stringify_safe(haystack);                                     \
      std::string __n = __bdd_cpp__::stringify_safe(needle);                                       \
      __BDD_WARN__(__h.find(__n) != std::string::npos, "expected \"%s\" to contain \"%s\"",        \
                   __h.c_str(), __n.c_str());                                                      \
    } while (0)

  #define check_string_starts_with(str, prefix)                                                    \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      std::string __p = __bdd_cpp__::stringify_safe(prefix);                                       \
      __BDD_CHECK__(__s.size() >= __p.size() && __s.compare(0, __p.size(), __p) == 0,              \
                    "expected \"%s\" to start with \"%s\"", __s.c_str(), __p.c_str());             \
    } while (0)
  #define check_string_starts_with_warn(str, prefix)                                               \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      std::string __p = __bdd_cpp__::stringify_safe(prefix);                                       \
      __BDD_WARN__(__s.size() >= __p.size() && __s.compare(0, __p.size(), __p) == 0,               \
                   "expected \"%s\" to start with \"%s\"", __s.c_str(), __p.c_str());              \
    } while (0)

  #define check_string_ends_with(str, suffix)                                                      \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      std::string __x = __bdd_cpp__::stringify_safe(suffix);                                       \
      __BDD_CHECK__(__s.size() >= __x.size() &&                                                    \
                        __s.compare(__s.size() - __x.size(), __x.size(), __x) == 0,                \
                    "expected \"%s\" to end with \"%s\"", __s.c_str(), __x.c_str());               \
    } while (0)
  #define check_string_ends_with_warn(str, suffix)                                                 \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      std::string __x = __bdd_cpp__::stringify_safe(suffix);                                       \
      __BDD_WARN__(__s.size() >= __x.size() &&                                                     \
                       __s.compare(__s.size() - __x.size(), __x.size(), __x) == 0,                 \
                   "expected \"%s\" to end with \"%s\"", __s.c_str(), __x.c_str());                \
    } while (0)

  #define check_string_empty(str)                                                                  \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      __BDD_CHECK__(__s.empty(), "expected empty string but got \"%s\"", __s.c_str());             \
    } while (0)
  #define check_string_empty_warn(str)                                                             \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      __BDD_WARN__(__s.empty(), "expected empty string but got \"%s\"", __s.c_str());              \
    } while (0)

  #define check_string_not_empty(str)                                                              \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      __BDD_CHECK__(!__s.empty(), "expected non-empty string");                                    \
    } while (0)
  #define check_string_not_empty_warn(str)                                                         \
    do {                                                                                           \
      std::string __s = __bdd_cpp__::stringify_safe(str);                                          \
      __BDD_WARN__(!__s.empty(), "expected non-empty string");                                     \
    } while (0)

  /* --- Template-based generic assertions --- */

  #define check_equal(actual, expected)                                                            \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_cpp__::__bdd_equal__(__bdd_a_ref__, __bdd_e_ref__)))) {      \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_CHECK__(0, "expected %s but got %s", __e.c_str(), __a.c_str());                      \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)
  #define check_equal_warn(actual, expected)                                                       \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_cpp__::__bdd_equal__(__bdd_a_ref__, __bdd_e_ref__)))) {      \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_WARN__(0, "expected %s but got %s", __e.c_str(), __a.c_str());                       \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

  #define check_not_equal(actual, expected)                                                        \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_cpp__::__bdd_not_equal__(__bdd_a_ref__, __bdd_e_ref__)))) {  \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_CHECK__(0, "expected != %s but got %s", __e.c_str(), __a.c_str());                   \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)
  #define check_not_equal_warn(actual, expected)                                                   \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_cpp__::__bdd_not_equal__(__bdd_a_ref__, __bdd_e_ref__)))) {  \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_WARN__(0, "expected != %s but got %s", __e.c_str(), __a.c_str());                    \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

  #define check_greater(actual, expected)                                                          \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_a_ref__ > __bdd_e_ref__))) {                                 \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_CHECK__(0, "expected > %s but got %s", __e.c_str(), __a.c_str());                    \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)
  #define check_greater_warn(actual, expected)                                                     \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_a_ref__ > __bdd_e_ref__))) {                                 \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_WARN__(0, "expected > %s but got %s", __e.c_str(), __a.c_str());                     \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

  #define check_less(actual, expected)                                                             \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_a_ref__ < __bdd_e_ref__))) {                                 \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_CHECK__(0, "expected < %s but got %s", __e.c_str(), __a.c_str());                    \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)
  #define check_less_warn(actual, expected)                                                        \
    do {                                                                                           \
      const auto &__bdd_a_ref__ = (actual);                                                        \
      const auto &__bdd_e_ref__ = (expected);                                                      \
      if (!__bdd_eval_bool__(!!(__bdd_a_ref__ < __bdd_e_ref__))) {                                 \
        std::string __a = __bdd_cpp__::to_string_safe(__bdd_a_ref__);                              \
        std::string __e = __bdd_cpp__::to_string_safe(__bdd_e_ref__);                              \
        __BDD_WARN__(0, "expected < %s but got %s", __e.c_str(), __a.c_str());                     \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

  /* --- Exception testing macros --- */

  /* check_throws(expr) — must throw any exception */
  #define check_throws(expr)                                                                       \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
      }                                                                                            \
      __BDD_CHECK__(__bdd_threw__, "expected exception but none was thrown");                      \
    } while (0)

  /* check_throws_as(expr, ExType) — must throw specific type */
  #define check_throws_as(expr, ExType)                                                            \
    do {                                                                                           \
      bool __bdd_threw_correct__ = false;                                                          \
      bool __bdd_threw_other__ = false;                                                            \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const ExType &) {                                                                   \
        __bdd_threw_correct__ = true;                                                              \
      } catch (...) {                                                                              \
        __bdd_threw_other__ = true;                                                                \
      }                                                                                            \
      if (__bdd_threw_other__) {                                                                   \
        __BDD_CHECK__(0, "expected " #ExType " but got a different exception");                    \
      } else {                                                                                     \
        __BDD_CHECK__(__bdd_threw_correct__, "expected " #ExType " but no exception was thrown");  \
      }                                                                                            \
    } while (0)

  /* check_throws_with(expr, msg) — must throw with what() containing msg */
  #define check_throws_with(expr, msg)                                                             \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      std::string __bdd_what__;                                                                    \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const std::exception &__e) {                                                        \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = __e.what();                                                                 \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = "(non-std exception)";                                                      \
      }                                                                                            \
      if (!__bdd_threw__) {                                                                        \
        __BDD_CHECK__(0, "expected exception with message \"%s\" but none was thrown", (msg));     \
      } else {                                                                                     \
        __BDD_CHECK__(__bdd_what__.find(msg) != std::string::npos,                                 \
                      "expected exception message containing \"%s\" but got \"%s\"", (msg),        \
                      __bdd_what__.c_str());                                                       \
      }                                                                                            \
    } while (0)

  /* check_nothrow(expr) — must not throw */
  #define check_nothrow(expr)                                                                      \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      std::string __bdd_what__;                                                                    \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const std::exception &__e) {                                                        \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = __e.what();                                                                 \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = "(non-std exception)";                                                      \
      }                                                                                            \
      if (__bdd_threw__) {                                                                         \
        __BDD_CHECK__(0, "expected no exception but got: %s", __bdd_what__.c_str());               \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

  /* Non-fatal versions */
  #define check_throws_warn(expr)                                                                  \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
      }                                                                                            \
      __BDD_WARN__(__bdd_threw__, "expected exception but none was thrown");                       \
    } while (0)

  #define check_throws_as_warn(expr, ExType)                                                       \
    do {                                                                                           \
      bool __bdd_threw_correct__ = false;                                                          \
      bool __bdd_threw_other__ = false;                                                            \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const ExType &) {                                                                   \
        __bdd_threw_correct__ = true;                                                              \
      } catch (...) {                                                                              \
        __bdd_threw_other__ = true;                                                                \
      }                                                                                            \
      if (__bdd_threw_other__) {                                                                   \
        __BDD_WARN__(0, "expected " #ExType " but got a different exception");                     \
      } else {                                                                                     \
        __BDD_WARN__(__bdd_threw_correct__, "expected " #ExType " but no exception was thrown");   \
      }                                                                                            \
    } while (0)

  #define check_throws_with_warn(expr, msg)                                                        \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      std::string __bdd_what__;                                                                    \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const std::exception &__e) {                                                        \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = __e.what();                                                                 \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = "(non-std exception)";                                                      \
      }                                                                                            \
      if (!__bdd_threw__) {                                                                        \
        __BDD_WARN__(0, "expected exception with message \"%s\" but none was thrown", (msg));      \
      } else {                                                                                     \
        __BDD_WARN__(__bdd_what__.find(msg) != std::string::npos,                                  \
                     "expected exception message containing \"%s\" but got \"%s\"", (msg),         \
                     __bdd_what__.c_str());                                                        \
      }                                                                                            \
    } while (0)

  #define check_nothrow_warn(expr)                                                                 \
    do {                                                                                           \
      bool __bdd_threw__ = false;                                                                  \
      std::string __bdd_what__;                                                                    \
      try {                                                                                        \
        (void)(expr);                                                                              \
      } catch (const std::exception &__e) {                                                        \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = __e.what();                                                                 \
      } catch (...) {                                                                              \
        __bdd_threw__ = true;                                                                      \
        __bdd_what__ = "(non-std exception)";                                                      \
      }                                                                                            \
      if (__bdd_threw__) {                                                                         \
        __BDD_WARN__(0, "expected no exception but got: %s", __bdd_what__.c_str());                \
      } else {                                                                                     \
        ++__bdd_active_config__->assertion_count;                                                  \
      }                                                                                            \
    } while (0)

#endif /* __cplusplus */

/* Use before_all()/after_all() as the cross-language names for one-time setup/teardown hooks. */
#define before_all()                                                                               \
  __BDD_NODE__(__bdd_node_flags_none__, list_before, __BDD_NODE_INTERIM__, "before")
#define after_all() __BDD_NODE__(__bdd_node_flags_none__, list_after, __BDD_NODE_INTERIM__, "after")

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

#endif /*TINYTEST_H*/
