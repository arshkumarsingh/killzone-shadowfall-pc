/* SPDX-License-Identifier: GPL-2.0-or-later
 * gow3 (Killzone): on hybrid CPUs (Intel 12th generation and later: performance and efficiency
 * cores) the threads a frame waits on prefer the performance cores. Killzone runs about 1000
 * jobs a frame on six workers that all meet at a barrier between the entity update and drawing
 * (Guerrilla, "Threading the Entity Update on PS4", 2014), so one worker on an efficiency core
 * holds up the whole frame; the GPU command thread is the port's own limit. The port's helper
 * threads (draw preparation, copies) keep every core.
 *
 * Plain C and C++ (runtime_thread.c, probe.c and liverpool.cpp include it). Windows CPU Sets:
 * a thread's selected sets are where the scheduler runs it. Only when the performance cores
 * offer 8 or more logical processors (room for the main thread, six workers and the GPU thread);
 * GOW3_PCORE_THREADS=0 turns it off, =1 forces it on any hybrid CPU. */
#ifndef GOW3_PCORES_H
#define GOW3_PCORES_H
#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>

/* SYSTEM_CPU_SET_INFORMATION (winnt.h), spelled out for headers without it. Entries are read
 * by their own size field. */
typedef struct {
    DWORD size, type;
    DWORD id;
    WORD group;
    BYTE logical_index, core_index, cache_index, numa_index, efficiency_class, flags;
    DWORD reserved;
    DWORD64 allocation_tag;
} Gow3CpuSet;
typedef BOOL (WINAPI *Gow3GetCpuSets)(void *, ULONG, PULONG, HANDLE, ULONG);
typedef BOOL (WINAPI *Gow3SetThreadCpuSets)(HANDLE, const ULONG *, ULONG);
enum { GOW3_PCORE_MAX = 256, GOW3_PCORE_MIN_LOGICAL = 8 };
typedef struct {
    ULONG ids[GOW3_PCORE_MAX];
    ULONG count, total;
    Gow3SetThreadCpuSets set;
} Gow3PCores;

static inline BOOL CALLBACK gow3_pcores_init(PINIT_ONCE once, PVOID parameter, PVOID *context) {
    (void)once; (void)context;
    Gow3PCores *p = (Gow3PCores *)parameter;
    const char *setting = getenv("GOW3_PCORE_THREADS");
    if (setting && setting[0] == '0') return TRUE;
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    if (!kernel) return TRUE;
    Gow3GetCpuSets get = (Gow3GetCpuSets)(void *)GetProcAddress(kernel, "GetSystemCpuSetInformation");
    p->set = (Gow3SetThreadCpuSets)(void *)GetProcAddress(kernel, "SetThreadSelectedCpuSets");
    if (!get || !p->set) return TRUE;
    ULONG length = 0;
    get(NULL, 0, &length, GetCurrentProcess(), 0);
    unsigned char *buffer = length ? (unsigned char *)malloc(length) : NULL;
    if (!buffer) return TRUE;
    if (get(buffer, length, &length, GetCurrentProcess(), 0)) {
        BYTE low = 255, high = 0;
        for (ULONG at = 0; at + sizeof(Gow3CpuSet) <= length;) {
            const Gow3CpuSet *s = (const Gow3CpuSet *)(buffer + at);
            if (!s->size) break;
            if (s->type == 0) {
                low = s->efficiency_class < low ? s->efficiency_class : low;
                high = s->efficiency_class > high ? s->efficiency_class : high;
                ++p->total;
            }
            at += s->size;
        }
        /* Hybrid: more than one efficiency class; the highest is the performance cores. */
        for (ULONG at = 0; high > low && at + sizeof(Gow3CpuSet) <= length;) {
            const Gow3CpuSet *s = (const Gow3CpuSet *)(buffer + at);
            if (!s->size) break;
            if (s->type == 0 && s->efficiency_class == high && p->count < GOW3_PCORE_MAX)
                p->ids[p->count++] = s->id;
            at += s->size;
        }
    }
    free(buffer);
    if (p->count < GOW3_PCORE_MIN_LOGICAL && !(setting && setting[0] == '1')) p->count = 0;
    return TRUE;
}

/* The calling thread prefers the performance cores. Returns their logical processor count, or 0
 * when nothing changed (not hybrid, too few, turned off, or the call failed); `total` (optional)
 * receives every logical processor. */
static inline unsigned gow3_prefer_performance_cores(unsigned *total) {
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    static Gow3PCores cores;
    InitOnceExecuteOnce(&once, gow3_pcores_init, &cores, NULL);
    if (total) *total = (unsigned)cores.total;
    if (!cores.count || !cores.set(GetCurrentThread(), cores.ids, cores.count)) return 0;
    return (unsigned)cores.count;
}
#else
static inline unsigned gow3_prefer_performance_cores(unsigned *total) {
    if (total) *total = 0;
    return 0;
}
#endif
#endif
