#pragma once

// Unico slot d'errore del backend, condiviso da tutte le TU GPU.
// Un buffer per processo, scritto dall'ultima chiamata CUDA fallita e riletto
// da last_error(). Interno a src/gpu: il confine pubblico resta align_gpu.h.

#include <cuda_runtime.h>

namespace theseus {
namespace gpu {

// Registra err insieme a context, sovrascrivendo il messaggio precedente.
void set_error(const char *context, cudaError_t err);

// Svuota lo slot, cosi' un entry point che riesce non riporta errori vecchi.
void clear_error();

}  // namespace gpu
}  // namespace theseus
