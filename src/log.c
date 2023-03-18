/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <locale.h>

#include "common.h"
#include "log.h"
#include "pl_thread.h"

struct priv {
    pl_mutex lock;
    enum pl_log_level log_level_cap;
    pl_str logbuffer;
};

pl_log pl_log_create(int api_ver, const struct pl_log_params *params)
{
    (void) api_ver;
    struct pl_log_t *log = pl_zalloc_obj(NULL, log, struct priv);
    struct priv *p = PL_PRIV(log);
    log->params = *PL_DEF(params, &pl_log_default_params);
    pl_mutex_init(&p->lock);
    pl_info(log, "Initialized libplacebo %s (API v%d)", PL_VERSION, PL_API_VER);
    return log;
}

const struct pl_log_params pl_log_default_params = {0};

void pl_log_destroy(pl_log *plog)
{
    pl_log log = *plog;
    if (!log)
        return;

    struct priv *p = PL_PRIV(log);
    pl_mutex_destroy(&p->lock);
    pl_free((void *) log);
    *plog = NULL;
}

struct pl_log_params pl_log_update(pl_log ptr, const struct pl_log_params *params)
{
    struct pl_log_t *log = (struct pl_log_t *) ptr;
    if (!log)
        return pl_log_default_params;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    struct pl_log_params prev_params = log->params;
    log->params = *PL_DEF(params, &pl_log_default_params);
    pl_mutex_unlock(&p->lock);

    return prev_params;
}

enum pl_log_level pl_log_level_update(pl_log ptr, enum pl_log_level level)
{
    struct pl_log_t *log = (struct pl_log_t *) ptr;
    if (!log)
        return PL_LOG_NONE;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    enum pl_log_level prev_level = log->params.log_level;
    log->params.log_level = level;
    pl_mutex_unlock(&p->lock);

    return prev_level;
}

void pl_log_level_cap(pl_log log, enum pl_log_level cap)
{
    if (!log)
        return;

    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);
    p->log_level_cap = cap;
    pl_mutex_unlock(&p->lock);
}

static FILE *default_stream(void *stream, enum pl_log_level level)
{
    return PL_DEF(stream, level <= PL_LOG_WARN ? stderr : stdout);
}

void pl_log_simple(void *stream, enum pl_log_level level, const char *msg)
{
    static const char *prefix[] = {
        [PL_LOG_FATAL] = "fatal",
        [PL_LOG_ERR]   = "error",
        [PL_LOG_WARN]  = "warn",
        [PL_LOG_INFO]  = "info",
        [PL_LOG_DEBUG] = "debug",
        [PL_LOG_TRACE] = "trace",
    };

    FILE *h = default_stream(stream, level);
    fprintf(h, "%5s: %s\n", prefix[level], msg);
    if (level <= PL_LOG_WARN)
        fflush(h);
}

void pl_log_color(void *stream, enum pl_log_level level, const char *msg)
{
    static const char *color[] = {
        [PL_LOG_FATAL] = "31;1", // bright red
        [PL_LOG_ERR]   = "31",   // red
        [PL_LOG_WARN]  = "33",   // yellow/orange
        [PL_LOG_INFO]  = "32",   // green
        [PL_LOG_DEBUG] = "34",   // blue
        [PL_LOG_TRACE] = "30;1", // bright black
    };

    FILE *h = default_stream(stream, level);
    fprintf(h, "\033[%sm%s\033[0m\n", color[level], msg);
    if (level <= PL_LOG_WARN)
        fflush(h);
}

static void pl_msg_va(pl_log log, enum pl_log_level lev,
                      const char *fmt, va_list va)
{
    // Test log message without taking the lock, to avoid thrashing the
    // lock for thousands of trace messages unless those are actually
    // enabled. This may be a false negative, in which case log messages may
    // be lost as a result. But this shouldn't be a big deal, since any
    // situation leading to lost log messages would itself be a race condition.
    if (!pl_msg_test(log, lev))
        return;

    // Re-test the log message level with held lock to avoid false positives,
    // which would be a considerably bigger deal than false negatives
    struct priv *p = PL_PRIV(log);
    pl_mutex_lock(&p->lock);

    // Apply this cap before re-testing the log level, to avoid giving users
    // messages that should have been dropped by the log level.
    lev = PL_MAX(lev, p->log_level_cap);
    if (!pl_msg_test(log, lev))
        goto done;

    p->logbuffer.len = 0;
    pl_str_append_vasprintf((void *) log, &p->logbuffer, fmt, va);
    log->params.log_cb(log->params.log_priv, lev, (char *) p->logbuffer.buf);

done:
    pl_mutex_unlock(&p->lock);
}

void pl_msg(pl_log log, enum pl_log_level lev, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    pl_msg_va(log, lev, fmt, va);
    va_end(va);
}

void pl_msg_source(pl_log log, enum pl_log_level lev, const char *src)
{
    if (!pl_msg_test(log, lev) || !src)
        return;

    int line = 1;
    while (*src) {
        const char *end = strchr(src, '\n');
        if (!end) {
            pl_msg(log, lev, "[%3d] %s", line, src);
            break;
        }

        pl_msg(log, lev, "[%3d] %.*s", line, (int)(end - src), src);
        src = end + 1;
        line++;
    }
}

#ifdef PL_HAVE_DBGHELP

#include <windows.h>
#include <dbghelp.h>
#include <shlwapi.h>

// https://github.com/llvm/llvm-project/blob/f03cd763384bbb67ddfa12957859ed58841d4b34/compiler-rt/lib/sanitizer_common/sanitizer_stacktrace.h#L85-L106
static inline uintptr_t get_prev_inst_pc(uintptr_t pc) {
#if defined(__arm__)
  // T32 (Thumb) branch instructions might be 16 or 32 bit long,
  // so we return (pc-2) in that case in order to be safe.
  // For A32 mode we return (pc-4) because all instructions are 32 bit long.
  return (pc - 3) & (~1);
#elif defined(__x86_64__) || defined(__i386__)
  return pc - 1;
#else
  return pc - 4;
#endif
}

static DWORD64 get_preferred_base(const char *module)
{
    DWORD64 image_base = 0;
    HANDLE file_mapping = NULL;
    HANDLE file_view = NULL;

    HANDLE file = CreateFile(module, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE)
        goto done;

    file_mapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file_mapping == NULL)
        goto done;

    file_view = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
    if (file_view == NULL)
        goto done;

    PIMAGE_DOS_HEADER dos_header = (PIMAGE_DOS_HEADER) file_view;
    if (dos_header->e_magic != IMAGE_DOS_SIGNATURE)
        goto done;

    PIMAGE_NT_HEADERS pe_header = (PIMAGE_NT_HEADERS) ((char *) file_view +
                                                                dos_header->e_lfanew);
    if (pe_header->Signature != IMAGE_NT_SIGNATURE)
        goto done;

    if (pe_header->FileHeader.SizeOfOptionalHeader != sizeof(pe_header->OptionalHeader))
        goto done;

    image_base = pe_header->OptionalHeader.ImageBase;

done:
    if (file_view)
        UnmapViewOfFile(file_view);
    if (file_mapping)
        CloseHandle(file_mapping);
    if (file != INVALID_HANDLE_VALUE)
        CloseHandle(file);

    return image_base;
}

void pl_log_stack_trace(pl_log log, enum pl_log_level lev)
{
    if (!pl_msg_test(log, lev))
        return;

    void *tmp = pl_tmp(NULL);
    PL_ARRAY(void *) frames = {0};

    size_t capacity = 16;
    do {
        capacity *= 2;
        PL_ARRAY_RESIZE(tmp, frames, capacity);
        // Skip first frame, we don't care about this function
        frames.num = CaptureStackBackTrace(1, capacity, frames.elem, NULL);
    } while (capacity == frames.num);

    if (!frames.num) {
        pl_free(tmp);
        return;
    }

    // Load dbghelp on demand. While it is available on all Windows versions,
    // no need to keep it loaded all the time as stack trace printing function,
    // in theory should be used repetitively rarely.
    HANDLE process = GetCurrentProcess();
    HANDLE dbghelp = LoadLibrary("dbghelp.dll");
    DWORD options;
    SYMBOL_INFO *symbol;
    BOOL use_dbghelp = !!dbghelp;

#define DBGHELP_SYM(sym)                                                        \
    __typeof__(&sym) p##sym = (__typeof__(&sym))(void *) GetProcAddress(dbghelp, #sym); \
    use_dbghelp &= !!p##sym

    DBGHELP_SYM(SymCleanup);
    DBGHELP_SYM(SymFromAddr);
    DBGHELP_SYM(SymGetLineFromAddr64);
    DBGHELP_SYM(SymGetModuleInfo64);
    DBGHELP_SYM(SymGetOptions);
    DBGHELP_SYM(SymGetSearchPathW);
    DBGHELP_SYM(SymInitialize);
    DBGHELP_SYM(SymSetOptions);
    DBGHELP_SYM(SymSetSearchPathW);

#undef DBGHELP_SYM

    struct priv *p = PL_PRIV(log);
    PL_ARRAY(wchar_t) base_search = { .num = 1024 };

    if (use_dbghelp) {
        // DbgHelp is not thread-safe. Note that on Windows mutex is recursive,
        // so no need to unlock before calling pl_msg.
        pl_mutex_lock(&p->lock);

        options = pSymGetOptions();
        pSymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS |
                       SYMOPT_LOAD_LINES | SYMOPT_FAVOR_COMPRESSED);
        use_dbghelp &= pSymInitialize(process, NULL, TRUE);

        if (use_dbghelp) {
            symbol = pl_alloc(tmp, sizeof(SYMBOL_INFO) + 512);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 512;

            PL_ARRAY_RESIZE(tmp, base_search, base_search.num);
            BOOL ret = pSymGetSearchPathW(process, base_search.elem,
                                          base_search.num);
            base_search.num = ret ? wcslen(base_search.elem) : 0;
            PL_ARRAY_APPEND(tmp, base_search, L'\0');
        } else {
            pSymSetOptions(options);
            pl_mutex_unlock(&p->lock);
        }
    }

    for (int n = 0; n < frames.num; n++) {
        uintptr_t pc = get_prev_inst_pc((uintptr_t) frames.elem[n]);
        pl_str out = {0};
        pl_str_append_asprintf(tmp, &out, "    #%-2d 0x%"PRIxPTR, n, pc);

        MEMORY_BASIC_INFORMATION meminfo = {0};
        char module_path[MAX_PATH] = {0};
        if (VirtualQuery((LPCVOID) pc, &meminfo, sizeof(meminfo))) {
            DWORD sz = GetModuleFileNameA(meminfo.AllocationBase, module_path,
                                          sizeof(module_path));
            if (sz == sizeof(module_path))
                pl_msg(log, PL_LOG_ERR, "module path truncated");

            if (use_dbghelp) {
                // According to documentation it should search in "The directory
                // that contains the corresponding module.", but it doesn't appear
                // to work, so manually set the path to module path.
                // https://learn.microsoft.com/windows/win32/debug/symbol-paths
                PL_ARRAY(wchar_t) mod_search = { .num = MAX_PATH };
                PL_ARRAY_RESIZE(tmp, mod_search, mod_search.num);

                sz = GetModuleFileNameW(meminfo.AllocationBase,
                                        mod_search.elem, mod_search.num);

                if (sz > 0 && sz != MAX_PATH &&
                    // TODO: Replace with PathCchRemoveFileSpec once mingw-w64
                    // >= 8.0.1 is commonly available, at the time of writing
                    // there are a few high profile Linux distributions that ship
                    // 8.0.0.
                    PathRemoveFileSpecW(mod_search.elem))
                {
                    mod_search.num = wcslen(mod_search.elem);
                    PL_ARRAY_APPEND(tmp, mod_search, L';');
                    PL_ARRAY_CONCAT(tmp, mod_search, base_search);
                    pSymSetSearchPathW(process, mod_search.elem);
                }
            }
        }

        DWORD64 sym_displacement;
        if (use_dbghelp && pSymFromAddr(process, pc, &sym_displacement, symbol))
            pl_str_append_asprintf(tmp, &out, " in %s+0x%llx",
                                   symbol->Name, sym_displacement);

        DWORD line_displacement;
        IMAGEHLP_LINE64 line = {sizeof(line)};
        if (use_dbghelp &&
            pSymGetLineFromAddr64(process, pc, &line_displacement, &line))
        {
            pl_str_append_asprintf(tmp, &out, " %s:%lu+0x%lx", line.FileName,
                                   line.LineNumber, line_displacement);
            goto done;
        }

        // LLVM tools by convention use absolute addresses with "prefered" base
        // image offset. We need to read this offset from binary, because due to
        // ASLR we are not loaded at this base. While Windows tools like WinDbg
        // expect relative offset to image base. So to be able to easily use it
        // with both worlds, print both values.
        DWORD64 module_base = get_preferred_base(module_path);
        pl_str_append_asprintf(tmp, &out, " (%s+0x%"PRIxPTR") (0x%llx)", module_path,
                               pc - (uintptr_t) meminfo.AllocationBase,
                               module_base + (pc - (uintptr_t) meminfo.AllocationBase));

done:
        pl_msg(log, lev, "%s", out.buf);
    }

    if (use_dbghelp) {
        pSymSetOptions(options);
        pSymCleanup(process);
        pl_mutex_unlock(&p->lock);
    }
    // Unload dbghelp. Maybe it is better to keep it loaded?
    if (dbghelp)
        FreeLibrary(dbghelp);
    pl_free(tmp);
}

#elif defined(PL_HAVE_UNWIND)
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <dlfcn.h>

void pl_log_stack_trace(pl_log log, enum pl_log_level lev)
{
    if (!pl_msg_test(log, lev))
        return;

    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip, off;
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);

    int depth = 0;
    pl_msg(log, lev, "  Backtrace:");
    while (unw_step(&cursor) > 0) {
        char symbol[256] = "<unknown>";
        Dl_info info = {
            .dli_fname = "<unknown>",
        };

        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_proc_name(&cursor, symbol, sizeof(symbol), &off);
        dladdr((void *) (uintptr_t) ip, &info);
        pl_msg(log, lev, "    #%-2d 0x%016" PRIxPTR " in %s+0x%" PRIxPTR" at %s+0x%" PRIxPTR,
               depth++, ip, symbol, off, info.dli_fname, ip - (uintptr_t) info.dli_fbase);
    }
}

#elif defined(PL_HAVE_EXECINFO) && !defined(MSAN)
#include <execinfo.h>

void pl_log_stack_trace(pl_log log, enum pl_log_level lev)
{
    if (!pl_msg_test(log, lev))
        return;

    PL_ARRAY(void *) buf = {0};
    size_t buf_avail = 16;
    do {
        buf_avail *= 2;
        PL_ARRAY_RESIZE(NULL, buf, buf_avail);
        buf.num = backtrace(buf.elem, buf_avail);
    } while (buf.num == buf_avail);

    pl_msg(log, lev, "  Backtrace:");
    char **strings = backtrace_symbols(buf.elem, buf.num);
    for (int i = 1; i < buf.num; i++)
        pl_msg(log, lev, "    #%-2d %s", i - 1, strings[i]);

    free(strings);
    pl_free(buf.elem);
}

#else
void pl_log_stack_trace(pl_log log, enum pl_log_level lev) { }
#endif
