// Tutto il codice device in una sola translation unit: i registri dipendono
// dall'inlining dell'intera catena. [cat. 1 privatization] Un blocco possiede
// una query; le fasi che definiscono prev_pos restano sul thread 0.

#include "gpu/align_gpu.h"
#include "query_state.h"
#include "gpu/align_core.h"
#include "gpu/kernel_launch.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace theseus {
namespace gpu {

namespace {

// Tetto, non taglia: oltre 1 KB la query si legge da global come prima.
// [cat. 5 tiling] E' l'unico dato che ogni cella M rilegge di continuo:
// copiarlo una volta per blocco ha portato i registri da 239 a 226.
constexpr int32_t kQueryTileBytes = 1024;

// Byte di shared memory dinamica per blocco: i buffer di staging tengono un
// solo tile, quindi scala con la larghezza del blocco e non con
// kScopeWavefrontCapacity. Il layout deve combaciare con quello del kernel.
size_t kernel_shared_bytes(int32_t threads_per_block) {
    return static_cast<size_t>(threads_per_block) *
               (sizeof(Cell) + 2 * sizeof(int)) +
           static_cast<size_t>(kQueryTileBytes);
}

// Una corsa contigua di candidati che una meta' sparsify riversa nella
// ScratchPad, descritta in modo che ogni thread ricostruisca da solo il
// candidato l.
struct SparsifyPart {
    const Cell *wf;            // wavefront sorgente
    const int64_t *positions;  // posizioni dei salti, null in una corsa densa
    int64_t start;             // primo indice di una corsa densa
    int32_t count;
    int32_t offset_increase;
    int32_t shift_factor;
    bool rewrite_prev;
    Cell::Matrix from_matrix;
};

// Le corse che una meta' sparsify visita, nell'ordine in cui le visita.
struct SparsifyPlan {
    SparsifyPart parts[4];
    int32_t nparts;
    int32_t total;
    int32_t query_len;
    int32_t upper_bound;
};

struct ICandidateRanges {
    Range i_dense;
    int32_t i_jumps_size;
    Range m_dense;
    int32_t m_jumps_size;
    int64_t *i_jumps;
    int64_t *m_jumps;
    int32_t pos_prev_i;
    int32_t pos_prev_m;
    int32_t upper_bound;
    int32_t total;
};


// Sonda di controllo: ogni thread riporta la lunghezza di una sequenza, per
// dimostrare che gli offset sono arrivati interi sul device.
__global__ void seq_length_kernel(const int32_t *offsets, int32_t num_seqs,
                                  int32_t *out_seq_lengths) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int bx = blockIdx.x;
    const int32_t tid = bx * ntx + tx;
    if (tid >= num_seqs) {
        return;
    }
    out_seq_lengths[tid] = offsets[tid + 1] - offsets[tid];
}

__device__ Range empty_range() {
    return Range{0, 0};
}

__device__ int32_t range_len(Range range) {
    return static_cast<int32_t>(range.end - range.start);
}

__device__ ICandidateRanges prepare_i_candidate_ranges(
                                                       QueryState &qs, const AlignScoring &scoring,
                                                       const GraphCsrView &graph, int32_t score, int32_t v) {
    ICandidateRanges ranges;
    ranges.i_dense = empty_range();
    ranges.i_jumps_size = 0;
    ranges.m_dense = empty_range();
    ranges.m_jumps_size = 0;
    ranges.i_jumps = nullptr;
    ranges.m_jumps = nullptr;
    ranges.pos_prev_i = score - scoring.gape;
    ranges.pos_prev_m = score - (scoring.gapo + scoring.gape);
    ranges.upper_bound = vertex_len(graph, v);
    ranges.total = 0;

    const int32_t v_id = vd_get_id(qs, v);
    if (v_id < 0) {
        return ranges;
    }

    if (ranges.pos_prev_i >= 0) {
        if (sc_i_pos_size(qs, ranges.pos_prev_i) > v_id) {
            ranges.i_dense = sc_i_pos(qs, ranges.pos_prev_i)[v_id];
        }
        const int32_t pos_prev_i_scope = vd_get_pos(qs, ranges.pos_prev_i);
        ranges.i_jumps = vd_i_jumps(qs, v, pos_prev_i_scope);
        ranges.i_jumps_size = vd_i_jumps_size(qs, v, pos_prev_i_scope);
    }
    if (ranges.pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, ranges.pos_prev_m) > v_id) {
            ranges.m_dense = sc_m_pos(qs, ranges.pos_prev_m)[v_id];
        }
        const int32_t pos_prev_m_scope = vd_get_pos(qs, ranges.pos_prev_m);
        ranges.m_jumps = vd_m_jumps(qs, v, pos_prev_m_scope);
        ranges.m_jumps_size = vd_m_jumps_size(qs, v, pos_prev_m_scope);
    }

    ranges.total = range_len(ranges.i_dense) + ranges.i_jumps_size +
                   range_len(ranges.m_dense) + ranges.m_jumps_size;
    return ranges;
}

__device__ bool make_i_candidate(QueryState &qs,
                                 const ICandidateRanges &ranges,
                                 int32_t idx,
                                 int32_t query_len,
                                 Cell &candidate) {
    const int32_t i_dense_count = range_len(ranges.i_dense);
    const int32_t i_jumps_begin = i_dense_count;
    const int32_t m_dense_begin = i_jumps_begin + ranges.i_jumps_size;
    const int32_t m_jumps_begin = m_dense_begin + range_len(ranges.m_dense);

    if (idx < i_dense_count) {
        candidate = sc_i_wf(qs, ranges.pos_prev_i)[ranges.i_dense.start + idx];
        candidate.diag += 1;
    } else if (idx < m_dense_begin) {
        const int32_t local = idx - i_jumps_begin;
        const int64_t pos = ranges.i_jumps[local];
        candidate = qs.bs_i_jumps_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::IJumps;
        candidate.diag += 1;
    } else if (idx < m_jumps_begin) {
        const int32_t local = idx - m_dense_begin;
        const int64_t pos = ranges.m_dense.start + local;
        candidate = qs.bs_m_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::M;
        candidate.diag += 1;
    } else {
        const int32_t local = idx - m_jumps_begin;
        const int64_t pos = ranges.m_jumps[local];
        candidate = qs.bs_m_jumps_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::MJumps;
        candidate.diag += 1;
    }

    const int32_t new_col = candidate.offset + candidate.diag;
    return candidate.offset <= query_len && new_col <= ranges.upper_bound;
}

constexpr int32_t kMaxWarps = 8;  // 256 thread, il blocco piu' largo ammesso

// Tutto quello che un blocco condivide mentre allinea la sua query.
// [cat. 6 occupancy] Erano diciotto `__shared__` separate: raggrupparle vale
// quasi 2x, ma solo insieme a __launch_bounds__ (da sole non servono).
struct BlockShared {
    // Controllo del loop, scritto dal thread 0.
    int block_continue;
    int block_end;
    int block_score;
    Cell block_end_cell;

    // Vertice corrente e celle M che ha prodotto.
    int num_active;
    int vertex;
    int64_t range_start;
    int64_t range_end;

    // Finestra della ScratchPad, fissata una volta per allineamento.
    int32_t span;
    int32_t clear_start;

    // Costruiti dal thread 0, letti dal blocco dopo una barriera.
    ICandidateRanges i_ranges;
    SparsifyPlan sparsify_plan;
    int i_count;

    // Scratch per prefix sum e riduzione max di blocco.
    int32_t warp_base[kMaxWarps];
    int32_t accum;

    // Un elemento per thread, in shared dinamica. `tile` e' l'unico staging di
    // Cell: sparsify e candidati I lo usano in sequenza, mai insieme. `m_valid`
    // e' separato perche' lavora mentre un tile e' ancora in volo.
    Cell *tile;
    int *tile_valid;
    int *m_valid;
};


// Assegna uno slot per ogni thread con flag settato, in ordine di thread: gli
// indici sono quelli di un append seriale. [cat. 1 privatization] Ballot di
// warp piu' somma sul thread 0, niente atomiche. Contiene barriere.
__device__ int32_t block_prefix_alloc(int flag, int32_t *shared_warp_base,
                                      int32_t &shared_running) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int32_t lane = tx & 31;
    const int32_t warp = tx >> 5;
    const int32_t nwarps = (ntx + 31) >> 5;

    // Al ballot arrivano tutti, anche i flag 0: il warp resta convergente.
    const unsigned ballot = __ballot_sync(0xffffffffu, flag != 0);
    const int32_t warp_prefix = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) {
        shared_warp_base[warp] = __popc(ballot);
    }
    __syncthreads();

    if (tx == 0) {
        int32_t acc = shared_running;
        for (int32_t w = 0; w < nwarps; ++w) {
            const int32_t count = shared_warp_base[w];
            shared_warp_base[w] = acc;
            acc += count;
        }
        shared_running = acc;
    }
    __syncthreads();

    const int32_t slot = shared_warp_base[warp] + warp_prefix;
    __syncthreads();   // shared_warp_base torna scratch per il chiamante
    return slot;
}

// Merge di un tile di candidati: riproduce sia il vincitore (offset massimo, a
// parita' indice minimo) sia l'ordine di append di sp_diags, che decide Range e
// prev_pos. [cat. 1 privatization] Scansione O(tile), non atomiche. Barriere.
__device__ void merge_candidate_tile(QueryState &qs, int32_t tile_len,
                                     BlockShared &sh) {
    const int tx = threadIdx.x;
    const bool mine = tx < tile_len && sh.tile_valid[tx] != 0;
    const int32_t my_diag = mine ? sh.tile[tx].diag : 0;
    const int32_t my_off = mine ? sh.tile[tx].offset : 0;

    // Come sp_merge_candidate: fuori finestra e' un fallimento di capacita'.
    const int32_t idx = my_diag - qs.sp_min_diag;
    const bool in_range = mine && idx >= 0 && idx < kScratchpadSpan;
    if (mine && !in_range) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1,
                 kScratchpadSpan);
    }
    if (mine && my_off < 0) {
        // Un offset e' una posizione nella query, quindi non e' mai negativo:
        // il caso resta controllato invece che assunto.
        cap_fail(qs, kCapScratchpadSpan, -1, kScratchpadSpan);
    }

    bool is_winner = in_range;
    bool is_first = in_range;
    if (in_range) {
        for (int32_t j = 0; j < tile_len; ++j) {
            if (j == tx || sh.tile_valid[j] == 0 ||
                sh.tile[j].diag != my_diag) {
                continue;
            }
            const int32_t oj = sh.tile[j].offset;
            if (oj > my_off || (oj == my_off && j < tx)) {
                is_winner = false;
            }
            if (j < tx) {
                is_first = false;
            }
        }
    }

    // Stato a riposo, letto prima di ogni scrittura del tile: la diagonale si
    // accoda solo se la cella era inattiva. Il flag e' sp_off, non l'offset.
    const bool needs_append = is_first && qs.sp_off[idx] == -1;

    // Qui sono state tolte due __syncthreads(): non ordinavano niente. sp_off e'
    // gia' separato dalla sua unica scrittura dalla barriera dopo l'append, e
    // sh.accum lo scrive e rilegge solo il thread 0.
    if (tx == 0) {
        sh.accum = qs.sp_ndiags;
    }
    const int32_t slot =
        block_prefix_alloc(needs_append ? 1 : 0, sh.warp_base, sh.accum);
    if (needs_append) {
        if (slot < kMaxActiveDiags) {
            qs.sp_diags[slot] = my_diag;
        } else {
            cap_fail(qs, kCapScratchpadDiags, slot + 1, kMaxActiveDiags);
        }
    }
    if (tx == 0) {
        qs.sp_ndiags =
            sh.accum < kMaxActiveDiags ? sh.accum : kMaxActiveDiags;
    }
    __syncthreads();

    // Un vincitore per diagonale: due thread non scrivono mai la stessa cella.
    if (is_winner && qs.sp_off[idx] < my_off) {
        qs.sp_wf[idx] = sh.tile[tx];
        qs.sp_off[idx] = my_off;
    }
    __syncthreads();
}

__device__ void plan_add(SparsifyPlan &plan, const Cell *wf,
                         const int64_t *positions, int64_t start, int32_t count,
                         int32_t offset_increase, int32_t shift_factor,
                         bool rewrite_prev, Cell::Matrix from_matrix) {
    if (count <= 0) {
        return;
    }
    SparsifyPart &part = plan.parts[plan.nparts];
    part.wf = wf;
    part.positions = positions;
    part.start = start;
    part.count = count;
    part.offset_increase = offset_increase;
    part.shift_factor = shift_factor;
    part.rewrite_prev = rewrite_prev;
    part.from_matrix = from_matrix;
    ++plan.nparts;
    plan.total += count;
}

// Le corse che percorre core_next_d_sparsify, nel suo ordine esatto.
__device__ SparsifyPlan prepare_d_sparsify_plan(QueryState &qs,
                                                const AlignScoring &scoring,
                                                int32_t query_len,
                                                const GraphCsrView &graph,
                                                int32_t score, int32_t v) {
    SparsifyPlan plan;
    plan.nparts = 0;
    plan.total = 0;
    plan.query_len = query_len;
    plan.upper_bound = vertex_len(graph, v);

    const int32_t pos_prev_m = score - (scoring.gapo + scoring.gape);
    const int32_t pos_prev_d = score - scoring.gape;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (pos_prev_d >= 0 && sc_d_pos_size(qs, pos_prev_d) > v_id) {
        const Range r = sc_d_pos(qs, pos_prev_d)[v_id];
        plan_add(plan, sc_d_wf(qs, pos_prev_d), nullptr, r.start,
                 range_len(r), 1, -1, false,
                 Cell::Matrix::None);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            const Range r = sc_m_pos(qs, pos_prev_m)[v_id];
            plan_add(plan, qs.bs_m_wf, nullptr, r.start,
                     range_len(r), 1, -1, true,
                     Cell::Matrix::M);
        }
        plan_add(plan, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope), 0,
                 vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, -1, true,
                 Cell::Matrix::MJumps);
    }
    return plan;
}

// Le corse che percorre core_next_m_sparsify, nel suo ordine.
__device__ SparsifyPlan prepare_m_sparsify_plan(QueryState &qs,
                                                const AlignScoring &scoring,
                                                int32_t query_len,
                                                const GraphCsrView &graph,
                                                int32_t score, int32_t v) {
    SparsifyPlan plan;
    plan.nparts = 0;
    plan.total = 0;
    plan.query_len = query_len;
    plan.upper_bound = vertex_len(graph, v);

    const int32_t pos_prev_m = score - scoring.mism;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (sc_d_pos_size(qs, score) > v_id) {
        const Range r = sc_d_pos(qs, score)[v_id];
        plan_add(plan, sc_d_wf(qs, score), nullptr, r.start,
                 range_len(r), 0, 0, false,
                 Cell::Matrix::None);
    }
    if (sc_i_pos_size(qs, score) > v_id) {
        const Range r = sc_i_pos(qs, score)[v_id];
        plan_add(plan, sc_i_wf(qs, score), nullptr, r.start,
                 range_len(r), 0, 0, false,
                 Cell::Matrix::None);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            const Range r = sc_m_pos(qs, pos_prev_m)[v_id];
            plan_add(plan, qs.bs_m_wf, nullptr, r.start,
                     range_len(r), 1, 0, true,
                     Cell::Matrix::M);
        }
        plan_add(plan, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope), 0,
                 vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, 0, true,
                 Cell::Matrix::MJumps);
    }
    return plan;
}

// Ricostruisce il candidato idx di un piano e dice se passa il filtro che ogni
// core_sparsify_* applica prima di toccare la ScratchPad.
__device__ bool make_sparsify_candidate(const SparsifyPlan &plan, int32_t idx,
                                        Cell &out) {
    int32_t local = idx;
    for (int32_t p = 0; p < plan.nparts; ++p) {
        const SparsifyPart &part = plan.parts[p];
        if (local < part.count) {
            const int64_t pos =
                part.positions != nullptr ? part.positions[local]
                                          : part.start + local;
            Cell cell = part.wf[pos];
            if (part.rewrite_prev) {
                cell.prev_pos = pos;
                cell.from_matrix = part.from_matrix;
            }
            cell.diag += part.shift_factor;
            cell.offset += part.offset_increase;
            out = cell;
            const int32_t new_col = cell.offset + cell.diag;
            return cell.offset <= plan.query_len && new_col <= plan.upper_bound;
        }
        local -= part.count;
    }
    out = Cell{-1, -1, -1, -1, Cell::Matrix::None};
    return false;
}

// Massimo di blocco, restituito a tutti i thread: max e' associativo, quindi
// l'ordine della riduzione da' lo stesso numero della scansione seriale.
// Riusa lo scratch shared_warp_base. Contiene barriere.
__device__ int32_t block_reduce_max(int32_t value, int32_t *shared_warp_base) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int32_t lane = tx & 31;
    const int32_t warp = tx >> 5;
    const int32_t nwarps = (ntx + 31) >> 5;

    // ntx e' 64, 128 o 256: ogni warp e' pieno e la maschera piena e' giusta.
    for (int32_t delta = 16; delta > 0; delta >>= 1) {
        const int32_t other = __shfl_down_sync(0xffffffffu, value, delta);
        if (other > value) {
            value = other;
        }
    }
    if (lane == 0) {
        shared_warp_base[warp] = value;
    }
    __syncthreads();

    if (tx == 0) {
        int32_t acc = shared_warp_base[0];
        for (int32_t w = 1; w < nwarps; ++w) {
            if (shared_warp_base[w] > acc) {
                acc = shared_warp_base[w];
            }
        }
        shared_warp_base[0] = acc;
    }
    __syncthreads();

    const int32_t result = shared_warp_base[0];
    __syncthreads();   // shared_warp_base torna scratch per il chiamante
    return result;
}

// Densify di I, D e M in parallelo: filtro indipendente per diagonale piu' una
// prefix sum esclusiva, che riproduce le posizioni seriali e tiene l'output
// byte-identico. In overflow si scarta e cap_fail registra il buffer.
__device__ Range densify(QueryState &qs, Cell::Matrix matrix, int32_t v,
                         Cell *wf, int32_t &wf_size, int32_t capacity,
                         CapBuffer cap_buffer, bool track_peak,
                         BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int32_t ndiags = qs.sp_ndiags;   // uniforme: densify non lo tocca mai
    const int32_t range_start = wf_size;   // letto prima che il thread 0 lo aggiorni

    // ScratchPad vuota nel 46-47 % delle chiamate, 100 % su c4_exact: uscire
    // subito da' lo stesso Range con due barriere in meno. Escono tutti i
    // thread o nessuno, perche' ndiags e' uniforme all'ingresso.
    if (ndiags == 0) {
        Range empty;
        empty.start = range_start;
        empty.end = range_start;
        return empty;
    }

    if (tx == 0) {
        sh.accum = 0;
    }
    __syncthreads();

    for (int32_t tile = 0; tile < ndiags; tile += ntx) {
        const int32_t di = tile + tx;
        int32_t flag = 0;
        Cell value{-1, -1, -1, -1, Cell::Matrix::None};
        if (di < ndiags) {
            const int32_t diag = qs.sp_diags[di];
            if (vd_valid_diagonal(qs, matrix, v, diag)) {
                flag = 1;
                value = sp_at(qs, diag);
            }
        }

        const int32_t slot =
            block_prefix_alloc(flag, sh.warp_base, sh.accum);
        if (flag != 0) {
            const int32_t pos = range_start + slot;
            if (pos < capacity) {
                wf[pos] = value;
            }
        }
        // Nessuna barriera a chiudere il tile: block_prefix_alloc finisce gia'
        // con una, e wf non lo legge nessuno prima della barriera dopo il loop.
    }

    if (tx == 0) {
        const int32_t total = sh.accum;
        const int32_t room = capacity - range_start;
        const int32_t fits = room > 0 ? room : 0;
        const int32_t written = total < fits ? total : fits;
        if (total > fits) {
            cap_fail(qs, cap_buffer, capacity + 1, capacity);
        }
        wf_size = range_start + written;
        if (track_peak && wf_size > qs.sc_peak_wf) {
            qs.sc_peak_wf = wf_size;
        }
    }
    __syncthreads();

    Range r;
    r.start = range_start;
    r.end = wf_size;
    return r;
}

// Riempie base[begin, end) con value. [cat. 2 thread coarsening] Quattro parole
// per thread in una sola store int4: -38 % istruzioni e 1,27x su c4_exact.
// L'allineamento e' verificato, non assunto: uno stato su due parte sfasato.
__device__ inline void fill_words(int32_t *base, int32_t begin, int32_t end,
                                  int32_t value, int32_t tx, int32_t ntx) {
    if (begin >= end) {
        return;
    }
    // Parole da saltare prima del primo confine a 16 byte. L'array e' int32,
    // quindi il disallineamento e' sempre multiplo di 4 e il conto e' esatto.
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base + begin);
    const int32_t skip = static_cast<int32_t>((16u - (addr & 15u)) & 15u) >> 2;
    int32_t vec_begin = begin + skip;
    if (vec_begin > end) {
        vec_begin = end;
    }
    const int32_t nvec = (end - vec_begin) >> 2;
    const int32_t vec_end = vec_begin + (nvec << 2);

    for (int32_t i = begin + tx; i < vec_begin; i += ntx) {
        base[i] = value;
    }
    const int4 quad = make_int4(value, value, value, value);
    int4 *vbase = reinterpret_cast<int4 *>(base + vec_begin);
    for (int32_t i = tx; i < nvec; i += ntx) {
        vbase[i] = quad;
    }
    for (int32_t i = vec_end + tx; i < end; i += ntx) {
        base[i] = value;
    }
}

// sp_reset in versione parallela di blocco. sp_diags contiene ogni diagonale
// toccata una volta sola, quindi le entry indirizzano celle distinte.
// Lo chiamano tutti i thread del blocco: contiene barriere.
__device__ void sp_reset_block(QueryState &qs) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int32_t ndiags = qs.sp_ndiags;   // uniforme: qui non lo scrive nessuno

    // Stessa uscita anticipata di densify, e qui pesa di piu': process_vertex
    // chiama questa funzione tre volte per vertice per score.
    if (ndiags == 0) {
        return;
    }

    for (int32_t i = tx; i < ndiags; i += ntx) {
        sp_reset_one(qs, i);
    }
    __syncthreads();
    if (tx == 0) {
        qs.sp_ndiags = 0;
    }
    __syncthreads();
}

// Riversa un piano di sparsify nella ScratchPad un tile alla volta: i buffer di
// staging sono dimensionati sul blocco, non sui candidati, e lo stato passa da
// un tile al successivo. Contiene barriere.
__device__ void run_sparsify_plan(QueryState &qs, const SparsifyPlan &plan,
                                  BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int32_t count = plan.total;

    for (int32_t tile_start = 0; tile_start < count; tile_start += ntx) {
        const int32_t idx = tile_start + tx;
        if (idx < count) {
            Cell candidate{-1, -1, -1, -1, Cell::Matrix::None};
            const bool valid = make_sparsify_candidate(plan, idx, candidate);
            sh.tile[tx] = candidate;
            sh.tile_valid[tx] = valid ? 1 : 0;
        } else {
            sh.tile_valid[tx] = 0;
        }
        __syncthreads();

        const int32_t remaining = count - tile_start;
        const int32_t lanes = remaining < ntx ? remaining : ntx;
        merge_candidate_tile(qs, lanes, sh);
    }
}

__device__ Range finish_i_wavefront(QueryState &qs, int32_t score, int32_t v,
                                    BlockShared &sh) {
    const int tx = threadIdx.x;
    const Range new_range =
        densify(qs, Cell::Matrix::I, v, sc_i_wf(qs, score),
                sc_i_wf_size(qs, score), kScopeWavefrontCapacity,
                kCapScopeWavefront, true, sh);
    if (tx == 0) {
        sc_pos_push(qs, sc_i_pos(qs, score), sc_i_pos_size(qs, score), new_range);
    }
    // Nessuna barriera dopo la push: sc_i_pos lo scrive e lo rilegge il thread 0,
    // e le celle di sc_i_wf le pubblica gia' la barriera finale di densify.
    return new_range;
}

// Candidati I di un vertice, fusi un tile di ntx alla volta: toglie il tetto
// che abortiva sopra kScopeWavefrontCapacity candidati. Possono collidere su
// una diagonale, quindi il merge non e' uno scatter: lo fa merge_candidate_tile.
__device__ void generate_and_merge_i_candidates(QueryState &qs,
                                                const AlignScoring &scoring,
                                                const char *query,
                                                int32_t query_len,
                                                const GraphCsrView &graph,
                                                int32_t score, int32_t v,
                                                BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;

    if (tx == 0) {
        sh.i_ranges = prepare_i_candidate_ranges(qs, scoring, graph,
                                                     score, v);
        sh.i_count = sh.i_ranges.total;
    }
    __syncthreads();

    // Uniforme nel blocco: tutti fanno lo stesso numero di tile e di barriere.
    const int32_t count = sh.i_count;

    for (int32_t tile_start = 0; tile_start < count; tile_start += ntx) {
        const int32_t idx = tile_start + tx;
        if (idx < count) {
            Cell candidate{-1, -1, -1, -1, Cell::Matrix::None};
            const bool valid = make_i_candidate(qs, sh.i_ranges, idx,
                                                query_len, candidate);
            sh.tile[tx] = candidate;
            sh.tile_valid[tx] = valid ? 1 : 0;
        } else {
            sh.tile_valid[tx] = 0;
        }
        __syncthreads();

        const int32_t remaining = count - tile_start;
        const int32_t lanes = remaining < ntx ? remaining : ntx;
        merge_candidate_tile(qs, lanes, sh);
    }

    const Range new_range = finish_i_wavefront(qs, score, v, sh);
    if (tx == 0) {
        // Stessa coda della next_I della CPU: le celle I arrivate all'ultima
        // colonna aprono i salti verso i vicini. Mancava nel port, ed era il bug
        // che teneva bloccato il tier complex.
        if (edge_begin(graph, v) < edge_end(graph, v)) {
            bool end = sh.block_end != 0;
            Cell end_cell = sh.block_end_cell;
            core_check_and_store_jumps(qs, query, query_len, graph, score, v,
                                       sc_i_wf(qs, score), new_range, end,
                                       end_cell);
            sh.block_end = end ? 1 : 0;
            sh.block_end_cell = end_cell;
        }
    }
    __syncthreads();
}

// core_lcp con un warp per cella. [cat. 4 divergenza] 32 caratteri per ballot,
// __ffs sul complemento: stesso offset e stesso j del seriale, perche' conta la
// corsa iniziale. Lane uniformi all'ingresso: contiene __ballot_sync.
__device__ inline void warp_lcp(const char *query, int32_t query_len,
                                const GraphCsrView &graph, int32_t v,
                                int32_t &offset, int32_t &j) {
    const int32_t n = vertex_len(graph, v);
    const int lane = threadIdx.x & 31;
    for (;;) {
        const int32_t room_q = query_len - offset;
        const int32_t room_v = n - j;
        const int32_t remaining = room_q < room_v ? room_q : room_v;
        if (remaining <= 0) {
            return;
        }
        const int32_t chunk = remaining < 32 ? remaining : 32;
        const bool match = lane < chunk &&
                           query[offset + lane] == vertex_char(graph, v, j + lane);
        const unsigned int matches = __ballot_sync(0xFFFFFFFFu, match);
        const unsigned int wanted =
            chunk == 32 ? 0xFFFFFFFFu : ((1u << chunk) - 1u);
        const unsigned int mismatches = (~matches) & wanted;
        const int32_t advance =
            mismatches != 0u ? (__ffs(static_cast<int>(mismatches)) - 1) : chunk;
        offset += advance;
        j += advance;
        if (advance < chunk) {
            return;
        }
    }
}

// Estensione del seed a score 0 sul warp 0: sposta solo la *misura*, la catena
// ordinata resta identica. [cat. 4 divergenza] 1,37-1,38x sul tier simple.
// [cat. 6 occupancy] __noinline__ stretta, altrimenti 138 -> 170 registri.
__device__ __noinline__ int32_t warp_seed_offset(const char *query,
                                                 int32_t query_len,
                                                 const GraphCsrView &graph,
                                                 int32_t vertex_id, int32_t diag,
                                                 int32_t offset) {
    int32_t j = diag + offset;
    warp_lcp(query, query_len, graph, vertex_id, offset, j);
    return offset;
}

__device__ void extend_and_consume_m_cells(QueryState &qs,
                                           const char *query,
                                           int32_t query_len,
                                           const GraphCsrView &graph,
                                           int32_t score,
                                           int64_t range_start,
                                           int64_t range_end,
                                           BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;

    // [cat. 4 divergenza] Un warp per cella invece di un thread per cella: il
    // tile e' largo nwarps e le 32 lane si dividono l'LCP di una cella sola.
    // La cella e' uniforme sul warp, quindi non diverge dentro warp_lcp.
    const int32_t lane = tx & 31;
    const int32_t warp_id = tx >> 5;
    const int32_t nwarps = ntx >> 5;

    for (int64_t chunk_start = range_start; chunk_start < range_end;
         chunk_start += nwarps) {
        const int64_t idx = chunk_start + warp_id;
        if (idx < range_end) {
            Cell cell = qs.bs_m_wf[idx];
            int32_t j = cell.diag + cell.offset;
            warp_lcp(query, query_len, graph, cell.vertex_id, cell.offset, j);
            // Il warp che ha esteso riscrive la sua cella, dalla lane 0.
            // Scrivere il tile in anticipo non si vede: il loop seriale qui
            // sotto non legge bs_m_wf.
            if (lane == 0) {
                qs.bs_m_wf[idx] = cell;
                sh.m_valid[warp_id] = 1;
            }
        } else if (lane == 0) {
            sh.m_valid[warp_id] = 0;
        }
        __syncthreads();

        if (tx == 0) {
            bool end = sh.block_end != 0;
            Cell end_cell = sh.block_end_cell;
            // Stesse celle nello stesso ordine di indice della versione seriale.
            for (int32_t w = 0; w < nwarps; ++w) {
                if (sh.m_valid[w] == 0) {
                    continue;
                }
                const int64_t cell_idx = chunk_start + w;
                Cell &cell = qs.bs_m_wf[cell_idx];
                core_check_end(cell, query_len, end, end_cell);
                const int32_t j = cell.diag + cell.offset;
                if (j == vertex_len(graph, cell.vertex_id) && cell.offset <= query_len &&
                    edge_begin(graph, cell.vertex_id) < edge_end(graph, cell.vertex_id)) {
                    core_store_m_jump(qs, query, query_len, graph, score,
                                      cell.vertex_id, cell, cell_idx,
                                      Cell::Matrix::M, end, end_cell);
                }
            }
            sh.block_end = end ? 1 : 0;
            sh.block_end_cell = end_cell;
        }
        __syncthreads();
    }
}

// [cat. 6 occupancy] Il __noinline__ vale 88 registri: sm_75, inlinata 226,
// tagliata 138, 0 spill in entrambi i casi e +21 % su c4_err_2k. Il taglio va
// esattamente qui: sulle fasi sotto da' 168-176. Chiama __syncthreads().
__device__ __noinline__ void process_vertex(QueryState &qs,
                               const AlignScoring &scoring,
                               const char *query,
                               int32_t query_len,
                               const GraphCsrView &graph,
                               int32_t score,
                               int32_t v,
                               BlockShared &sh) {
    const int tx = threadIdx.x;

    // Senza barriera: sh.range_* lo legge solo extend_and_consume_m_cells in
    // fondo, dietro una barriera che pubblica anche questo default.
    if (tx == 0) {
        sh.range_start = 0;
        sh.range_end = 0;
    }

    generate_and_merge_i_candidates(qs, scoring, query, query_len, graph,
                                    score, v, sh);

    sp_reset_block(qs);
    if (tx == 0) {
        sh.sparsify_plan =
            prepare_d_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    run_sparsify_plan(qs, sh.sparsify_plan, sh);
    {
        const Range d_range =
            densify(qs, Cell::Matrix::D, v, sc_d_wf(qs, score),
                    sc_d_wf_size(qs, score), kScopeWavefrontCapacity,
                    kCapScopeWavefront, true, sh);
        if (tx == 0) {
            sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), d_range);
        }
    }
    // Nessuna barriera a chiudere la fase D, ne' la M qui sotto: le celle le
    // pubblica la barriera finale di densify, sc_d_pos lo tocca solo il thread 0
    // e ogni tile di run_sparsify_plan finisce gia' su una barriera.

    sp_reset_block(qs);
    if (tx == 0) {
        sh.sparsify_plan =
            prepare_m_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    run_sparsify_plan(qs, sh.sparsify_plan, sh);
    {
        const Range m_range =
            densify(qs, Cell::Matrix::M, v, qs.bs_m_wf, qs.bs_m_wf_size,
                    kBeyondScopeCapacity, kCapBeyondScope, false, sh);
        if (tx == 0) {
            sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), m_range);
        }
    }

    sp_reset_block(qs);
    if (tx == 0) {
        const int32_t v_pos = vd_get_id(qs, v);
        if (!qs.capacity_exceeded && v_pos >= 0 &&
            sc_m_pos_size(qs, score) > v_pos) {
            const Range cells_range = sc_m_pos(qs, score)[v_pos];
            sh.range_start = cells_range.start;
            sh.range_end = cells_range.end;
        }
    }
    __syncthreads();

    extend_and_consume_m_cells(qs, query, query_len, graph, score,
                               sh.range_start, sh.range_end, sh);
    __syncthreads();
}

// vd_expand + vd_compact in versione parallela di blocco: toccano solo gli array
// del vertice `a`, quindi i vertici sono indipendenti, ne basta uno per thread e
// le due passate si possono fondere.
__device__ void expand_and_compact(QueryState &qs, BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int num_active = qs.vd_num_active;
    const int g = qs.vd_gape;
    for (int a = tx; a < num_active; a += ntx) {
        vd_expand_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
        qs.vd_m_invalid_size[a] =
            vd_compact_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        qs.vd_i_invalid_size[a] =
            vd_compact_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        qs.vd_d_invalid_size[a] =
            vd_compact_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
    __syncthreads();
    if (tx == 0) {
        sh.num_active = vd_num_active_vertices(qs);
    }
    __syncthreads();
}

__device__ void compute_new_wave(QueryState &qs,
                                 const AlignScoring &scoring,
                                 const char *query,
                                 int32_t query_len,
                                 const GraphCsrView &graph,
                                 int32_t score,
                                 BlockShared &sh) {
    const int tx = threadIdx.x;

    expand_and_compact(qs, sh);

    for (int32_t l = 0; l < sh.num_active; ++l) {
        if (tx == 0) {
            sh.vertex = vd_get_vertex_id(qs, l);
        }
        __syncthreads();
        process_vertex(qs, scoring, query, query_len, graph, score,
                       sh.vertex, sh);
    }
}

__device__ void align_one(QueryState &qs, const AlignScoring &scoring,
                          const char *query, int32_t query_len,
                          const GraphCsrView &graph,
                          int32_t start_node,
                          int32_t start_offset,
                          AlignResult &result,
                          BlockShared &sh) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;

    // Il vertice piu' lungo del grafo dimensiona la finestra della ScratchPad:
    // e' un max su tutto il grafo, quindi lo riduce il blocco, non il thread 0.
    int32_t local_max_diag = 0;
    for (int32_t v = tx; v < graph.num_vertices; v += ntx) {
        const int32_t n = vertex_len(graph, v);
        if (n > local_max_diag) {
            local_max_diag = n;
        }
    }
    const int32_t max_diag = block_reduce_max(local_max_diag, sh.warp_base);

    if (tx == 0) {
        qs.capacity_exceeded = false;
        // cap_fail tiene il *primo* fallimento e confronta cap_reason con
        // kCapNone: lasciarci quello del batch prima nasconderebbe un fallimento
        // vero. Lo azzerava il cudaMemset per batch, che non c'e' piu'.
        qs.cap_reason = kCapNone;
        qs.cap_required = 0;
        qs.cap_available = 0;
        // Solo le meta' scalari: i due loop di pulizia li fa il blocco qui sotto.
        sh.span = sp_init_window(qs, -query_len, max_diag);
        sh.clear_start = sp_clear_start(qs, sh.span);
        sc_init(qs, scoring.nscores);
        vd_init_scalar(qs, scoring.gapo, scoring.gape, scoring.nscores,
                       graph.num_vertices);
        sc_new_alignment(qs);
        bs_new_alignment(qs);
        vd_new_alignment_scalar(qs);
    }
    __syncthreads();

    // Limiti uniformi e store indipendenti che scrivono tutte lo stesso valore:
    // non conta chi pulisce cosa. vd_init e vd_new_alignment azzeravano lo stesso
    // prefisso allo stesso -1, quindi qui collassano in una passata sola.
    const int32_t vd_fill = vd_map_fill_count(graph.num_vertices);
    // [cat. 3 coalescing] Il clear era il 44-71 % dei cicli: ora stabilisce solo
    // sp_off, da 6 x span a 1 x span per query, fino a 7,86x. Sopra ci sta il
    // clear pigro (sp_clear_start): 5,1x sul tier simple.
    fill_words(qs.sp_off, sh.clear_start, sh.span, -1, tx, ntx);
    fill_words(qs.vd_vertex_to_idx, 0, vd_fill, -1, tx, ntx);
    __syncthreads();

    if (tx == 0) {
        sh.block_score = 0;
        sh.block_end = 0;
        sh.block_end_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};

        sc_new_score(qs, sh.block_score);
        Cell init_condition;
        init_condition.offset = 0;
        init_condition.vertex_id = start_node;
        init_condition.diag = start_offset;
        init_condition.prev_pos = -1;
        init_condition.from_matrix = Cell::Matrix::None;

        bs_push_back(qs, qs.bs_m_jumps_wf, qs.bs_m_jumps_wf_size,
                     init_condition);
        vd_activate_vertex(qs, start_node);
        vd_jumps_push(qs, vd_m_jumps(qs, start_node, 0),
                      vd_m_jumps_size(qs, start_node, 0), 0);
    }
    __syncthreads();

    while (true) {
        if (tx == 0) {
            sh.block_continue = (sh.block_end == 0 && !qs.capacity_exceeded) ? 1 : 0;
        }
        __syncthreads();
        const int continue_now = sh.block_continue;
        __syncthreads();
        if (continue_now == 0) {
            break;
        }

        // Seed a score 0: warp 0 avanza la cella (warp_seed_offset), poi il
        // thread 0 esegue la catena ordinata com'era. tx < 32 e' uniforme e
        // tutte le lane leggono la stessa cella: warp_lcp vuole entrambe.
        if (sh.block_score == 0 && tx < 32) {
            const Cell seed = qs.bs_m_jumps_wf[0];
            const int32_t new_offset =
                warp_seed_offset(query, query_len, graph, seed.vertex_id,
                                 seed.diag, seed.offset);
            if (tx == 0) {
                // La lane 0 e' il thread 0: stesso thread, niente barriera.
                qs.bs_m_jumps_wf[0].offset = new_offset;

                bool end = sh.block_end != 0;
                Cell end_cell = sh.block_end_cell;
                core_extend_diagonal(qs, query, query_len, graph, sh.block_score,
                                     qs.bs_m_jumps_wf[0], qs.bs_m_jumps_wf[0], 0,
                                     Cell::Matrix::MJumps, end, end_cell);
                sh.block_end = end ? 1 : 0;
                sh.block_end_cell = end_cell;
            }
        }
        __syncthreads();

        compute_new_wave(qs, scoring, query, query_len, graph, sh.block_score,
                         sh);

        if (tx == 0) {
            ++sh.block_score;
            sc_new_score(qs, sh.block_score);
        }
        __syncthreads();

        // vd_new_score, un vertice attivo per thread: ogni entry e' di un solo
        // vertice e prendono tutte lo stesso 0, quindi non conta chi pulisce cosa.
        {
            const int pos = vd_new_score_slot(qs, sh.block_score);
            for (int a = tx; a < qs.vd_num_active; a += ntx) {
                vd_new_score_one(qs, a, pos);
            }
        }
        __syncthreads();
    }

    if (tx == 0) {
        // Oltre kMaxActiveDiags l'append e' scartato ma la cella vincente si
        // scrive lo stesso, quindi puo' restarne una attiva che nessuno resetta.
        // Il risultato e' gia' buttato: sp_cleared a 0 non passa lo sporco oltre.
        if (qs.capacity_exceeded) {
            qs.sp_cleared = 0;
        }
        --sh.block_score;
        result.score = sh.block_score;
        result.end_vertex_id = sh.block_end_cell.vertex_id;
        result.end_offset = sh.block_end_cell.offset;
        result.end_diag = sh.block_end_cell.diag;
        result.end_prev_pos = sh.block_end_cell.prev_pos;
        result.end_from_matrix = static_cast<int8_t>(sh.block_end_cell.from_matrix);
        result.reached_end = sh.block_end != 0 ? 1 : 0;
        result.capacity_exceeded = qs.capacity_exceeded ? 1 : 0;
        result.reserved = 0;
    }
    __syncthreads();
}

// [cat. 6 occupancy] Due blocchi da 256 per SM tengono ptxas a 128 registri:
// senza tetto ne prende 175 senza servirgli (0 spill), 11 warp invece di 16.
// Vale solo con BlockShared: 2,14 ms contro 4,17 su c4_err_2k, T4.
__global__ __launch_bounds__(256, 2) void theseus_align_batch_kernel(
                                           BatchView batch, GraphCsrView graph,
                                           const int32_t *start_node_ids,
                                           const int32_t *start_offsets,
                                           AlignScoring scoring,
                                           QueryState *states,
                                           AlignResult *results) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int bx = blockIdx.x;

    const int32_t query_id = bx;   // un blocco per query
    if (query_id >= batch.num_seqs) {
        return;
    }

    QueryState *state = &states[query_id];

    __shared__ BlockShared sh;

    // Un tile per buffer di staging, dimensionato al lancio su ntx. Cell per
    // prima perche' ha l'allineamento piu' stretto; combacia con kernel_shared_bytes().
    extern __shared__ unsigned char smem[];
    Cell *tile = reinterpret_cast<Cell *>(smem);
    int *tile_valid = reinterpret_cast<int *>(tile + ntx);
    int *m_valid = tile_valid + ntx;
    char *shared_query = reinterpret_cast<char *>(m_valid + ntx);

    if (tx == 0) {
        sh.block_continue = 0;
        sh.block_end = 0;
        sh.block_score = 0;
        sh.num_active = 0;
        sh.vertex = 0;
        sh.range_start = 0;
        sh.range_end = 0;
        sh.i_count = 0;
        sh.block_end_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};
        sh.tile = tile;
        sh.tile_valid = tile_valid;
        sh.m_valid = m_valid;
    }

    __syncthreads();

    const int32_t begin = batch.offsets[query_id];
    const int32_t end = batch.offsets[query_id + 1];

    // [cat. 5 tiling] Se ci sta, la query si copia in shared e ad align_one si
    // passa quel puntatore. A valle e' un const char * in sola lettura, quindi la
    // sostituzione e' invisibile: stessi byte, stessi indici.
    const int32_t query_len = end - begin;
    const char *query = batch.chars + begin;
    if (query_len <= kQueryTileBytes) {
        for (int32_t i = tx; i < query_len; i += ntx) {
            shared_query[i] = query[i];
        }
        query = shared_query;
    }
    __syncthreads();

    align_one(*state, scoring, query, query_len, graph,
              start_node_ids[query_id], start_offsets[query_id],
              results[query_id], sh);
}

__global__ void traceback_meta_kernel(const QueryState *states, int32_t count,
                                      TracebackMeta *metadata) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const QueryState &state = states[i];
    metadata[i] = TracebackMeta{state.bs_m_wf_size,
                                state.bs_m_jumps_wf_size,
                                state.bs_i_jumps_wf_size,
                                state.sc_peak_wf,
                                state.cap_required,
                                state.cap_available,
                                static_cast<int8_t>(state.capacity_exceeded),
                                state.cap_reason,
                                {0, 0}};
}

// Compatta le celle del backtrace host: un blocco per query scrive i suoi tre
// prefissi (M, M jumps, I jumps) da base[i], prefix sum esclusiva dell'host.
// Porta la D2H da 288 KB fissi per query ai byte prodotti: 24 celle su 12 288.
__global__ void pack_traceback_kernel(const QueryState *states, int32_t count,
                                      const int32_t *base, Cell *packed) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const QueryState &state = states[i];
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;

    Cell *out = packed + base[i];
    const int32_t m = state.bs_m_wf_size;
    const int32_t mj = state.bs_m_jumps_wf_size;
    const int32_t ij = state.bs_i_jumps_wf_size;

    for (int32_t k = tx; k < m; k += ntx) {
        out[k] = state.bs_m_wf[k];
    }
    for (int32_t k = tx; k < mj; k += ntx) {
        out[m + k] = state.bs_m_jumps_wf[k];
    }
    for (int32_t k = tx; k < ij; k += ntx) {
        out[m + mj + k] = state.bs_i_jumps_wf[k];
    }
}

// Rilegge il CSR, un blocco per vertice. Ogni blocco arriva al proprio testo e ai
// propri archi dagli array di offset, cioe' con la stessa traversata del kernel di
// allineamento: leggerlo in altro modo verificherebbe l'upload ma non quella.
__global__ void graph_readback_kernel(GraphCsrView graph,
                                      char *out_vertex_chars,
                                      int32_t *out_vertex_offsets,
                                      int32_t *out_edge_targets,
                                      int32_t *out_edge_overlaps,
                                      int32_t *out_edge_offsets) {
    const int tx = threadIdx.x;
    const int ntx = blockDim.x;
    const int bx = blockIdx.x;

    const int32_t v = bx;   // un blocco per vertice
    if (v >= graph.num_vertices) {
        return;
    }

    const int32_t text_begin = graph.vertex_offsets[v];
    const int32_t text_end = graph.vertex_offsets[v + 1];
    for (int32_t i = text_begin + tx; i < text_end; i += ntx) {
        out_vertex_chars[i] = graph.vertex_chars[i];
    }

    const int32_t edge_begin = graph.edge_offsets[v];
    const int32_t edge_end = graph.edge_offsets[v + 1];
    for (int32_t e = edge_begin + tx; e < edge_end; e += ntx) {
        out_edge_targets[e] = graph.edge_targets[e];
        out_edge_overlaps[e] = graph.edge_overlaps[e];
    }

    if (tx == 0) {
        out_vertex_offsets[v] = text_begin;
        out_edge_offsets[v] = edge_begin;
        if (v == graph.num_vertices - 1) {
            out_vertex_offsets[v + 1] = text_end;
            out_edge_offsets[v + 1] = edge_end;
        }
    }
}

}  // namespace

// I wrapper di lancio dichiarati in kernel_launch.h. Ognuno possiede la geometria
// del proprio kernel (blocchi e, per l'allineamento, la shared dinamica), cosi'
// nessun chiamante puo' contraddirlo. Lanciano e riportano, non aspettano.
cudaError_t launch_seq_length(int32_t threads_per_block, const int32_t *offsets,
                              int32_t num_seqs, int32_t *out_seq_lengths) {
    const int blocks = (num_seqs + threads_per_block - 1) / threads_per_block;
    seq_length_kernel<<<blocks, threads_per_block>>>(offsets, num_seqs,
                                                    out_seq_lengths);
    return cudaGetLastError();
}

cudaError_t launch_align_batch(int32_t threads_per_block, const BatchView &batch,
                               const GraphCsrView &graph,
                               const int32_t *start_node_ids,
                               const int32_t *start_offsets,
                               AlignScoring scoring, QueryState *states,
                               AlignResult *results) {
    theseus_align_batch_kernel<<<batch.num_seqs, threads_per_block,
                                 kernel_shared_bytes(threads_per_block)>>>(
        batch, graph, start_node_ids, start_offsets, scoring, states, results);
    return cudaGetLastError();
}

cudaError_t launch_traceback_meta(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  TracebackMeta *metadata) {
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    traceback_meta_kernel<<<blocks, threads_per_block>>>(states, count, metadata);
    return cudaGetLastError();
}

cudaError_t launch_pack_traceback(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  const int32_t *base, Cell *packed) {
    pack_traceback_kernel<<<count, threads_per_block>>>(states, count, base,
                                                        packed);
    return cudaGetLastError();
}

cudaError_t launch_graph_readback(const GraphCsrView &graph,
                                  char *out_vertex_chars,
                                  int32_t *out_vertex_offsets,
                                  int32_t *out_edge_targets,
                                  int32_t *out_edge_overlaps,
                                  int32_t *out_edge_offsets) {
    // Un blocco per vertice, 32 thread: testo e archi di un vertice sono corti.
    graph_readback_kernel<<<graph.num_vertices, 32>>>(
        graph, out_vertex_chars, out_vertex_offsets, out_edge_targets,
        out_edge_overlaps, out_edge_offsets);
    return cudaGetLastError();
}

}  // namespace gpu
}  // namespace theseus
