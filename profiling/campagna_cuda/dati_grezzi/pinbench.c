/* Proxy locale del costo di cudaHostAlloc: allocare e bloccare in memoria N MiB.
   cudaHostAlloc fa allocazione + page lock + registrazione presso il driver; qui
   si misurano i primi due, che sono la parte proporzionale alla dimensione. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(int argc, char **argv) {
    const size_t mib = argc > 1 ? (size_t)atoll(argv[1]) : 576;
    const size_t bytes = mib << 20;
    struct timespec t0, t1, t2, t3;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const int locked = mlock(p, bytes) == 0;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    memset(p, 0, bytes);                 /* first touch, se mlock non c'e' */
    clock_gettime(CLOCK_MONOTONIC, &t3);

    printf("%4zu MiB | mmap %7.1f ms | mlock %7.1f ms (%s) | first touch %7.1f ms\n",
           mib, ms(t0, t1), ms(t1, t2), locked ? "ok" : "NEGATO", ms(t2, t3));
    munlock(p, bytes); munmap(p, bytes);
    return 0;
}
