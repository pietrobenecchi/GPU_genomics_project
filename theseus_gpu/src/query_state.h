/*
 *                             The MIT License
 *
 * Copyright (c) 2024 by Albert Jimenez-Blanco
 *
 * This file is part of #################### Theseus Library ####################.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#pragma once

#include <cstdint>

#include "cell.h"

#ifdef __CUDACC__
#define THESEUS_HD __host__ __device__
#else
#define THESEUS_HD
#endif

// Il working set per query, array POD piatti a capacita' fissa: il device non
// puo' allocare a kernel avviato, quindi ne possiede uno per query. Le capacita'
// sono PROVVISORIE e l'overflow non e' mai silenzioso, lo registra cap_fail.

namespace theseus {

// Diagonali della ScratchPad, derivate: max_vertex_len + max_query_len + 1,
// cioe' 9 164 su ebola e 52 107 su c4. Crescere costa 28 byte per diagonale e si
// paga sulla taglia del batch; un grafo con vertici piu' lunghi ne vuole di piu'.
constexpr int kScratchpadSpan = 52224;  // vertice c4 da 52 006 bp + read da 100 bp

// Celle di ogni wavefront BeyondScope: crescono per tutto l'allineamento e le
// rilegge il backtrace, quindi e' il buffer per query dominante. PROVVISORIO.
constexpr int kBeyondScopeCapacity = 4096;

// Lunghezza dell'anello dello Scope. La CPU la ricava a runtime dalle penalita'
// (5 con quelle di default), gli array device vogliono un bound a compile time.
constexpr int kMaxScores = 8;

// Vertici che un allineamento puo' toccare. Sta qui, prima degli altri bound di
// VerticesData, perche' i vettori di posizione dello Scope ne discendono.
constexpr int kMaxActiveVertices = 256;

// Celle di ogni wavefront I e D dello Scope. PROVVISORIO, tenuto uguale a
// kProvisionalWavefrontCapacity, che sta in un header non includibile da qui.
constexpr int kScopeWavefrontCapacity = 1024;

// Derivato: a uno score ogni vettore di posizione riceve una push per vertice
// attivo, quindi non supera kMaxActiveVertices. Era 1024, quattro volte tanto:
// 384 KB per QueryState invece di 96. Il guard in sc_pos_push resta.
constexpr int kScopePosCapacity = kMaxActiveVertices;

// Un range semiaperto in un wavefront, uno per vertice attivo. Copia il POD
// Scope::range, cosi' lo Scope appiattito non dipende dalla classe Scope.
struct Range {
    int64_t start;
    int64_t end;
};

// Bound fissi di VerticesData, tutti PROVVISORI e guardati: taglia del grafo,
// segmenti di diagonali non valide per matrice per vertice, salti per score.
constexpr int kMaxVertices = 2048;
// kMaxActiveVertices sta sopra, insieme al bound dello Scope che ne discende.
constexpr int kMaxInvalidSegments = 64;
constexpr int kMaxJumpsPerScore = 32;

// Derivato dalle capacita' sopra, non dalla finestra: sp_diags non puo' essere
// piu' lunga dei candidati fusi in una fase di sparsify, e il tetto e' la fase M
// (due wavefront Scope, un range BeyondScope, una lista di salti). Era 52 224.
constexpr int kMaxActiveDiags = 2 * kScopeWavefrontCapacity +
                                kBeyondScopeCapacity + kMaxJumpsPerScore;
// Derivato: lo stack cresce solo su catene di vertici a lunghezza zero e il
// picco misurato sui quattro dataset GGBS e' 1; 8 copre una catena di sette.
// Overflow non silenzioso: cap_fail registra kCapIJumpStack e la profondita'.
constexpr int kMaxIJumpStack = 8;

// Una corsa di diagonali non valide per vertice/matrice, coi contatori che la
// fanno crescere. Copia VerticesData::InvalidData appiattito in un POD.
struct InvalidSeg {
    int32_t start_d;   // incluso
    int32_t end_d;     // incluso
    int32_t rem_up;    // score che mancano prima che cresca di una diagonale in su
    int32_t rem_down;  // score che mancano prima che cresca di una in giu'
};

struct QueryState {
    // ---- ScratchPad -------------------------------------------------------

    // Wavefront per diagonale su [sp_min_diag, sp_max_diag], piu' la lista delle
    // diagonali toccate dall'ultimo reset (la vista sparsa che percorre
    // densify). Una cella e' inattiva quando il suo offset e' -1.

    // L'offset sta in sp_off, array denso suo, ed e' l'autorita' per attivita' e
    // confronti. Il clear per query e' una parola per diagonale invece delle sei
    // di una Cell, su tutta la finestra (52 107 su c4) di cui se ne tocca poche.

    // sp_wf e' la meta' fredda: payload delle celle attive, mai pulito, letto
    // solo via sp_diags. Il suo `offset` viaggia col resto della cella, quindi
    // una attiva concorda con sp_off; una inattiva tiene l'ultimo valore.
    Cell    sp_wf[kScratchpadSpan];
    int32_t sp_off[kScratchpadSpan];
    Cell    sp_overflow_cell;
    int32_t sp_diags[kMaxActiveDiags];
    // Quanta parte di sp_off e' mai stata messa a -1, in entry da 0. La
    // ScratchPad torna da sola a riposo (process_vertex chiude con sp_reset),
    // quindi un prefisso pulito resta pulito. 0 = niente pulito, come uno nuovo.
    int32_t sp_cleared;
    int32_t sp_min_diag;
    int32_t sp_max_diag;
    int32_t sp_ndiags;

    // ---- BeyondScope ------------------------------------------------------

    // I wavefront append-only tenuti fino al backtrace: uscita di densify M e
    // celle di salto M/I, array piatto piu' taglia. Fissi, quindi i riferimenti
    // non si spostano. Il wavefront I2-jumps, inutilizzato, non c'e'.
    Cell    bs_m_wf[kBeyondScopeCapacity];
    Cell    bs_m_jumps_wf[kBeyondScopeCapacity];
    Cell    bs_i_jumps_wf[kBeyondScopeCapacity];
    int32_t bs_m_wf_size;
    int32_t bs_m_jumps_wf_size;
    int32_t bs_i_jumps_wf_size;

    // ---- Scope (anello di nscores wavefront) ------------------------------

    // Indicizzato per score % sc_nscores: ogni slot tiene i wavefront densi I e D
    // (quello M sta in BeyondScope) e i range M/I/D, uno per vertice attivo.
    // sc_peak_wf e' il picco, per ricavarne un bound vero.
    Cell    sc_i_wf[kMaxScores][kScopeWavefrontCapacity];
    Cell    sc_d_wf[kMaxScores][kScopeWavefrontCapacity];
    int32_t sc_i_wf_size[kMaxScores];
    int32_t sc_d_wf_size[kMaxScores];
    Range   sc_m_pos[kMaxScores][kScopePosCapacity];
    Range   sc_i_pos[kMaxScores][kScopePosCapacity];
    Range   sc_d_pos[kMaxScores][kScopePosCapacity];
    int32_t sc_m_pos_size[kMaxScores];
    int32_t sc_i_pos_size[kMaxScores];
    int32_t sc_d_pos_size[kMaxScores];
    int32_t sc_nscores;
    int32_t sc_peak_wf;

    // ---- VerticesData -----------------------------------------------------

    // Attivi in [0, vd_num_active); vd_vertex_to_idx mappa un id del grafo sul
    // suo indice attivo (-1 se inattivo). Per vertice: segmenti di diagonali non
    // valide M/I/D e salti per score. vd_gapo/vd_gape sono le penalita'.
    int32_t vd_gapo;
    int32_t vd_gape;
    int32_t vd_nscores;
    int32_t vd_num_vertices;
    int32_t vd_num_active;
    int32_t vd_vertex_to_idx[kMaxVertices];
    int32_t vd_vertex_id[kMaxActiveVertices];

    InvalidSeg vd_m_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    InvalidSeg vd_i_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    InvalidSeg vd_d_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    int32_t vd_m_invalid_size[kMaxActiveVertices];
    int32_t vd_i_invalid_size[kMaxActiveVertices];
    int32_t vd_d_invalid_size[kMaxActiveVertices];

    int64_t vd_m_jumps_pos[kMaxActiveVertices][kMaxScores][kMaxJumpsPerScore];
    int64_t vd_i_jumps_pos[kMaxActiveVertices][kMaxScores][kMaxJumpsPerScore];
    int32_t vd_m_jumps_pos_size[kMaxActiveVertices][kMaxScores];
    int32_t vd_i_jumps_pos_size[kMaxActiveVertices][kMaxScores];

    // Alzato la prima volta che una capacita' fissa qui sopra non basta. L'host
    // lo controlla dopo l'allineamento: non viene mai ignorato in silenzio.
    bool capacity_exceeded;

    // Quale buffer e' finito per primo, e di quanto. Un booleano diceva solo che
    // un overflow c'era stato fra quattordici siti, che non basta a dimensionare
    // niente. Si registra solo il PRIMO: i seguenti di solito ne discendono.
    int8_t  cap_reason;    // a CapBuffer value
    int32_t cap_required;  // entries the caller asked for
    int32_t cap_available; // entries the buffer has
};

// Quello che serve all'host per ricostruire un allineamento, e nient'altro. Il
// backtrace legge solo i tre wavefront BeyondScope piu' la diagnostica: 288 KB
// per query invece di 4,2 MB, 147 MB per un batch da 512 invece di 2,1 GB.

// Il costruttore di default e' scritto a mano e vuoto apposta: rende il tipo non
// banalmente costruibile, cosi' std::vector<CompactTracebackState>(n) lo chiama
// invece di azzerare un buffer che align_batch riscrive per intero.
struct CompactTracebackState {
    CompactTracebackState() noexcept {}

    // Viste sul buffer compattato che riempie align_batch, non storage.

    // Erano tre Cell[kBeyondScopeCapacity], 288 KB per query comunque andasse.
    // Misurato sulla CPU, su c4_err_2k il wavefront M ha in media 24,3 celle su
    // 4096 e picco 438: si legge lo 0,21 % dei byte mossi.

    // Ora align_batch compatta i prefissi usati sul device e copia una volta: le
    // celle sono un prestito nello staging, valido fino al prossimo align_batch
    // (il backtrace gira subito dopo). Taglia 0 lascia il puntatore, mai nullo.
    const Cell *bs_m_wf;
    const Cell *bs_m_jumps_wf;
    const Cell *bs_i_jumps_wf;
    int32_t bs_m_wf_size;
    int32_t bs_m_jumps_wf_size;
    int32_t bs_i_jumps_wf_size;

    // Diagnostica, cosi' un overflow sul device lo riporta l'host, che il buffer
    // andato in overflow non lo vede mai.
    int32_t sc_peak_wf;
    bool    capacity_exceeded;
    int8_t  cap_reason;    // a CapBuffer value
    int32_t cap_required;
    int32_t cap_available;
};

// Vista in sola lettura sui tre wavefront che percorre il backtrace. Legarci qui
// le due sorgenti (QueryState della CPU e CompactTracebackState del kernel)
// tiene una traceback sola invece di un test a ogni passo.
struct TracebackWavefronts {
    const Cell *m;
    const Cell *m_jumps;
    const Cell *i_jumps;
};

THESEUS_HD inline TracebackWavefronts traceback_view(const QueryState &qs) {
    return TracebackWavefronts{qs.bs_m_wf, qs.bs_m_jumps_wf, qs.bs_i_jumps_wf};
}

THESEUS_HD inline TracebackWavefronts traceback_view(const CompactTracebackState &state) {
    return TracebackWavefronts{state.bs_m_wf, state.bs_m_jumps_wf, state.bs_i_jumps_wf};
}

// Quale buffer a capacita' fissa e' andato in overflow. I valori sono stabili:
// tornano dal device e li stampa l'host.
enum CapBuffer : int8_t {
    kCapNone = 0,
    kCapScratchpadSpan,   // finestra di diagonali della ScratchPad
    kCapScratchpadDiags,  // lista delle diagonali attive
    kCapBeyondScope,      // wavefront di backtrace che si accumulano
    kCapScores,           // lunghezza dell'anello dello Scope
    kCapScopeWavefront,   // celle dei wavefront I/D per score
    kCapScopePos,         // range di posizione M/I/D per score
    kCapVertices,         // dominio della mappa vertice -> indice attivo
    kCapActiveVertices,   // vertici toccati in un allineamento
    kCapInvalidSegments,  // segmenti non validi per matrice per vertice
    kCapJumpsPerScore,    // posizioni dei salti per vertice per score
    kCapIJumpStack,       // profondita' dello stack esplicito in store_I_jump
};

// Registra un overflow: alza il flag e tiene la prima causa.
THESEUS_HD inline void cap_fail(QueryState &qs, CapBuffer which, int32_t required,
                                int32_t available) {
    qs.capacity_exceeded = true;
    if (qs.cap_reason == kCapNone) {
        qs.cap_reason = which;
        qs.cap_required = required;
        qs.cap_available = available;
    }
}

// Nome del buffer a cui si riferisce @p reason, per i messaggi sull'host.
inline const char *cap_buffer_name(int8_t reason) {
    switch (reason) {
        case kCapScratchpadSpan:  return "scratchpad diagonal span";
        case kCapScratchpadDiags: return "scratchpad active-diagonal list";
        case kCapBeyondScope:     return "beyond-scope wavefront";
        case kCapScores:          return "scope ring length";
        case kCapScopeWavefront:  return "scope wavefront cells";
        case kCapScopePos:        return "scope position ranges";
        case kCapVertices:        return "vertex index map";
        case kCapActiveVertices:  return "active vertices";
        case kCapInvalidSegments: return "invalid segments";
        case kCapJumpsPerScore:   return "jumps per score";
        case kCapIJumpStack:      return "I-jump stack";
        default:                  return "none";
    }
}

// La meta' scalare di sp_init: fissa la finestra, svuota la lista attiva e
// controlla lo span. Torna le celle che il chiamante deve ancora pulire, 0 se lo
// span non ci sta. Separata perche' sul device il loop lo fa tutto il blocco.
THESEUS_HD inline int sp_init_window(QueryState &qs, int min_diag, int max_diag) {
    qs.sp_min_diag = min_diag;
    qs.sp_max_diag = max_diag;
    qs.sp_ndiags = 0;
    qs.sp_overflow_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};

    const int span = max_diag - min_diag + 1;
    if (span > kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, span, kScratchpadSpan);
        return 0;
    }
    return span;
}

// Entry di sp_off che questa query deve ancora pulire, come [start, span). Solo
// quelle mai pulite: la ScratchPad si rimette a posto da sola, quindi resta
// scoperto solo cio' che una finestra piu' lunga raggiunge per la prima volta.

// L'eccezione e' un fallimento di capacita': l'append viene scartato ma la cella
// si scrive, e resta sporca senza che nessuno lo registri. align_one risponde
// rimettendo sp_cleared a 0, cosi' la query dopo ripulisce tutto.
THESEUS_HD inline int sp_clear_start(QueryState &qs, int span) {
    const int start = qs.sp_cleared < span ? qs.sp_cleared : span;
    if (span > qs.sp_cleared) {
        qs.sp_cleared = span;
    }
    return start;
}

THESEUS_HD inline void sp_init(QueryState &qs, int min_diag, int max_diag) {
    const int span = sp_init_window(qs, min_diag, max_diag);
    for (int i = sp_clear_start(qs, span); i < span; ++i) {
        qs.sp_off[i] = -1;
    }
}

// La cella della ScratchPad sulla diagonale @p diag, senza marcarla attiva.
THESEUS_HD inline Cell &sp_at(QueryState &qs, int diag) {
    const int idx = diag - qs.sp_min_diag;
    if (idx < 0 || idx >= kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1, kScratchpadSpan);
        qs.sp_overflow_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};
        return qs.sp_overflow_cell;
    }
    return qs.sp_wf[idx];
}

// Fonde un candidato nella ScratchPad tenendo l'offset piu' grande: e'
// access_alloc piu' il confronto che ogni chiamante faceva da se'. Una primitiva
// sola perche' offset e payload ora stanno in due array, e vanno tenuti in passo.

// Semantica invariata: la diagonale si accoda solo se la cella era inattiva
// (dedup e ordine di primo tocco, che decide il wavefront denso), il confronto
// resta stretto, e fuori finestra e' un fallimento di capacita'.
THESEUS_HD inline void sp_merge_candidate(QueryState &qs, const Cell &c) {
    const int idx = c.diag - qs.sp_min_diag;
    if (idx < 0 || idx >= kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1, kScratchpadSpan);
        return;
    }
    if (qs.sp_off[idx] == -1) {
        if (qs.sp_ndiags >= kMaxActiveDiags) {
            cap_fail(qs, kCapScratchpadDiags, qs.sp_ndiags + 1, kMaxActiveDiags);
            return;
        }
        qs.sp_diags[qs.sp_ndiags] = c.diag;
        ++qs.sp_ndiags;
    }
    if (qs.sp_off[idx] < c.offset) {
        qs.sp_wf[idx] = c;
        qs.sp_off[idx] = c.offset;
    }
}

// Rimette la ScratchPad a riposo: ogni diagonale attiva torna inattiva e la
// lista si svuota. Si puliscono solo le celle toccate, come sulla CPU.
THESEUS_HD inline void sp_reset_one(QueryState &qs, int i) {
    qs.sp_off[qs.sp_diags[i] - qs.sp_min_diag] = -1;
}

THESEUS_HD inline void sp_reset(QueryState &qs) {
    for (int i = 0; i < qs.sp_ndiags; ++i) {
        sp_reset_one(qs, i);
    }
    qs.sp_ndiags = 0;
}

// Svuota tutti i wavefront BeyondScope per un nuovo allineamento.
THESEUS_HD inline void bs_new_alignment(QueryState &qs) {
    qs.bs_m_wf_size = 0;
    qs.bs_m_jumps_wf_size = 0;
    qs.bs_i_jumps_wf_size = 0;
}

// Accoda @p c a un wavefront BeyondScope e torna l'indice dove e' finita. In
// overflow la cella si scarta e si alza capacity_exceeded: l'allineamento e' gia'
// invalido, ma non si scrive oltre l'array e l'indice clampato tiene in bound.
THESEUS_HD inline int32_t bs_push_back(QueryState &qs, Cell *wf, int32_t &size,
                            const Cell &c) {
    if (size >= kBeyondScopeCapacity) {
        cap_fail(qs, kCapBeyondScope, size + 1, kBeyondScopeCapacity);
        return (size > 0) ? size - 1 : 0;
    }
    const int32_t pos = size;
    wf[pos] = c;
    size = pos + 1;
    return pos;
}

// ---- Scope ----------------------------------------------------------------

// Fissa la lunghezza dell'anello dello Scope e lo svuota. Un anello troppo lungo
// viene clampato e segnalato, non indicizzato fuori range.
THESEUS_HD inline void sc_init(QueryState &qs, int nscores) {
    if (nscores > kMaxScores) {
        cap_fail(qs, kCapScores, nscores, kMaxScores);
        nscores = kMaxScores;
    }
    qs.sc_nscores = nscores;
    qs.sc_peak_wf = 0;
    for (int s = 0; s < kMaxScores; ++s) {
        qs.sc_i_wf_size[s] = 0;
        qs.sc_d_wf_size[s] = 0;
        qs.sc_m_pos_size[s] = 0;
        qs.sc_i_pos_size[s] = 0;
        qs.sc_d_pos_size[s] = 0;
    }
}

// Svuota tutti gli slot dell'anello per un nuovo allineamento.
THESEUS_HD inline void sc_new_alignment(QueryState &qs) {
    for (int s = 0; s < qs.sc_nscores; ++s) {
        qs.sc_i_wf_size[s] = 0;
        qs.sc_d_wf_size[s] = 0;
        qs.sc_m_pos_size[s] = 0;
        qs.sc_i_pos_size[s] = 0;
        qs.sc_d_pos_size[s] = 0;
    }
}

// Azzera lo slot di @p score, il cui contenuto vecchio e' ormai fuori scope.
THESEUS_HD inline void sc_new_score(QueryState &qs, int score) {
    const int s = score % qs.sc_nscores;
    qs.sc_i_wf_size[s] = 0;
    qs.sc_d_wf_size[s] = 0;
    qs.sc_m_pos_size[s] = 0;
    qs.sc_i_pos_size[s] = 0;
    qs.sc_d_pos_size[s] = 0;
}

// Viste sugli slot, coi nomi degli accessor dello Scope: i call site non cambiano.
THESEUS_HD inline Cell *sc_i_wf(QueryState &qs, int score) { return qs.sc_i_wf[score % qs.sc_nscores]; }
THESEUS_HD inline Cell *sc_d_wf(QueryState &qs, int score) { return qs.sc_d_wf[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_i_wf_size(QueryState &qs, int score) { return qs.sc_i_wf_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_d_wf_size(QueryState &qs, int score) { return qs.sc_d_wf_size[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_m_pos(QueryState &qs, int score) { return qs.sc_m_pos[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_i_pos(QueryState &qs, int score) { return qs.sc_i_pos[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_d_pos(QueryState &qs, int score) { return qs.sc_d_pos[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_m_pos_size(QueryState &qs, int score) { return qs.sc_m_pos_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_i_pos_size(QueryState &qs, int score) { return qs.sc_i_pos_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_d_pos_size(QueryState &qs, int score) { return qs.sc_d_pos_size[score % qs.sc_nscores]; }

// Accoda una cella a un wavefront dello Scope, tenendo traccia del picco. In
// overflow la cella si scarta e si alza capacity_exceeded, senza scrivere oltre.
THESEUS_HD inline void sc_wf_push(QueryState &qs, Cell *wf, int32_t &size, const Cell &c) {
    if (size >= kScopeWavefrontCapacity) {
        cap_fail(qs, kCapScopeWavefront, size + 1, kScopeWavefrontCapacity);
        return;
    }
    wf[size] = c;
    ++size;
    if (size > qs.sc_peak_wf) {
        qs.sc_peak_wf = size;
    }
}

// Accoda un range a un vettore di posizione dello Scope, controllando la capacita'.
THESEUS_HD inline void sc_pos_push(QueryState &qs, Range *pos, int32_t &size, const Range &r) {
    if (size >= kScopePosCapacity) {
        cap_fail(qs, kCapScopePos, size + 1, kScopePosCapacity);
        return;
    }
    pos[size] = r;
    ++size;
}

// ---- VerticesData ---------------------------------------------------------

THESEUS_HD inline int32_t vd_min(int32_t a, int32_t b) { return a < b ? a : b; }
THESEUS_HD inline int32_t vd_max(int32_t a, int32_t b) { return a > b ? a : b; }

// Quante entry di vd_vertex_to_idx possiede un grafo di @p num_vertices. E'
// l'unico posto dove vive il clamp a kMaxVertices, cosi' le meta' scalari qui
// sotto e il fill parallelo del kernel non possono discordare sul bound.
THESEUS_HD inline int vd_map_fill_count(int num_vertices) {
    return num_vertices > kMaxVertices ? kMaxVertices : num_vertices;
}

// La meta' scalare di vd_init: penalita', anello, taglia del grafo e controllo di
// capacita'. Torna quante entry di vd_vertex_to_idx restano da mettere a -1.
// Separata come sp_init_window: sul device il loop lo fa tutto il blocco.
THESEUS_HD inline int vd_init_scalar(QueryState &qs, int gapo, int gape, int nscores,
                                     int num_vertices) {
    qs.vd_gapo = gapo;
    qs.vd_gape = gape;
    qs.vd_nscores = nscores;
    qs.vd_num_vertices = num_vertices;
    qs.vd_num_active = 0;
    if (num_vertices > kMaxVertices) {
        cap_fail(qs, kCapVertices, num_vertices, kMaxVertices);
    }
    return vd_map_fill_count(num_vertices);
}

// Fissa penalita', anello e taglia del grafo, e svuota. @p num_vertices e' il
// dominio della mappa vertice -> indice.
THESEUS_HD inline void vd_init(QueryState &qs, int gapo, int gape, int nscores,
                    int num_vertices) {
    const int n = vd_init_scalar(qs, gapo, gape, nscores, num_vertices);
    for (int i = 0; i < n; ++i) {
        qs.vd_vertex_to_idx[i] = -1;
    }
}

// La meta' scalare di vd_new_alignment. Vedi vd_init_scalar.
THESEUS_HD inline int vd_new_alignment_scalar(QueryState &qs) {
    qs.vd_num_active = 0;
    return vd_map_fill_count(qs.vd_num_vertices);
}

// Scarta tutti i vertici attivi e marca inattivo ogni vertice del grafo.
THESEUS_HD inline void vd_new_alignment(QueryState &qs) {
    const int n = vd_new_alignment_scalar(qs);
    for (int i = 0; i < n; ++i) {
        qs.vd_vertex_to_idx[i] = -1;
    }
}

THESEUS_HD inline int vd_get_pos(const QueryState &qs, int score) { return score % qs.vd_nscores; }
THESEUS_HD inline int vd_get_id(const QueryState &qs, int vtx) { return qs.vd_vertex_to_idx[vtx]; }
THESEUS_HD inline int vd_get_vertex_id(const QueryState &qs, int idx) { return qs.vd_vertex_id[idx]; }
THESEUS_HD inline int vd_num_active_vertices(const QueryState &qs) { return qs.vd_num_active; }

// Aggiunge @p vtx all'insieme attivo se non c'e' gia'. Un vertice appena attivo
// ha le liste di segmenti non validi e di salti vuote.
THESEUS_HD inline void vd_activate_vertex(QueryState &qs, int vtx) {
    if (vtx >= kMaxVertices) {
        cap_fail(qs, kCapVertices, vtx + 1, kMaxVertices);
        return;
    }
    if (qs.vd_vertex_to_idx[vtx] != -1) {
        return;
    }
    const int idx = qs.vd_num_active;
    if (idx >= kMaxActiveVertices) {
        cap_fail(qs, kCapActiveVertices, idx + 1, kMaxActiveVertices);
        return;
    }
    qs.vd_vertex_id[idx] = vtx;
    qs.vd_m_invalid_size[idx] = 0;
    qs.vd_i_invalid_size[idx] = 0;
    qs.vd_d_invalid_size[idx] = 0;
    for (int s = 0; s < qs.vd_nscores; ++s) {
        qs.vd_m_jumps_pos_size[idx][s] = 0;
        qs.vd_i_jumps_pos_size[idx][s] = 0;
    }
    qs.vd_vertex_to_idx[vtx] = idx;
    qs.vd_num_active = idx + 1;
}

// Azzera le posizioni dei salti di ogni vertice attivo nello slot di @p score.
THESEUS_HD inline int vd_new_score_slot(const QueryState &qs, int score) {
    return score % qs.vd_nscores;
}

// Azzera le posizioni dei salti di un vertice attivo nello slot @p pos.
THESEUS_HD inline void vd_new_score_one(QueryState &qs, int a, int pos) {
    qs.vd_m_jumps_pos_size[a][pos] = 0;
    qs.vd_i_jumps_pos_size[a][pos] = 0;
}

THESEUS_HD inline void vd_new_score(QueryState &qs, int score) {
    const int pos = vd_new_score_slot(qs, score);
    for (int a = 0; a < qs.vd_num_active; ++a) {
        vd_new_score_one(qs, a, pos);
    }
}

// Invecchia di uno score ogni segmento di una lista, allargandolo quando un
// contatore arriva a zero.
THESEUS_HD inline void vd_expand_vec(InvalidSeg *v, int size, int def_up, int def_down) {
    for (int l = 0; l < size; ++l) {
        v[l].rem_down -= 1;
        v[l].rem_up -= 1;
        if (v[l].rem_up == 0) {
            v[l].rem_up = def_up;
            v[l].end_d += 1;
        }
        if (v[l].rem_down == 0) {
            v[l].rem_down = def_down;
            v[l].start_d -= 1;
        }
    }
}

THESEUS_HD inline void vd_expand(QueryState &qs) {
    const int g = qs.vd_gape;
    for (int a = 0; a < qs.vd_num_active; ++a) {
        vd_expand_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
}

// Ordina una lista di segmenti per diagonale d'inizio e fonde quelli che si
// sovrappongono. La std::sort diventa un insertion sort (i segmenti per vertice
// sono pochi), l'aritmetica del merge e' verbatim. Torna la nuova taglia.
THESEUS_HD inline int vd_compact_vec(InvalidSeg *v, int size, int def_up, int def_down) {
    if (size == 0) {
        return 0;
    }

    // Insertion sort su start_d, crescente, stabile.
    for (int i = 1; i < size; ++i) {
        InvalidSeg key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j].start_d > key.start_d) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }

    int k = 0;
    for (int l = 1; l < size; ++l) {
        if (v[k].end_d + 1 >= v[l].start_d) {
            v[k].end_d = vd_max(v[l].end_d, v[k].end_d);

            v[k].rem_down = vd_min(v[k].rem_down,
                                   v[l].rem_down +
                                       (v[l].start_d - v[k].start_d) * def_down);

            if (v[l].end_d > v[k].end_d) {
                v[k].rem_up = vd_min(v[l].rem_up,
                                     v[k].rem_up +
                                         (v[l].end_d - v[k].end_d) * def_up);
            } else {
                v[k].rem_up = vd_min(v[k].rem_up,
                                     v[l].rem_up +
                                         (v[k].end_d - v[l].end_d) * def_up);
            }
        } else {
            k += 1;
            v[k] = v[l];
        }
    }
    return k + 1;
}

THESEUS_HD inline void vd_compact(QueryState &qs) {
    const int g = qs.vd_gape;
    for (int a = 0; a < qs.vd_num_active; ++a) {
        qs.vd_m_invalid_size[a] = vd_compact_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        qs.vd_i_invalid_size[a] = vd_compact_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        qs.vd_d_invalid_size[a] = vd_compact_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
}

THESEUS_HD inline void vd_invalid_push(QueryState &qs, InvalidSeg *v, int32_t &size,
                            const InvalidSeg &s) {
    if (size >= kMaxInvalidSegments) {
        cap_fail(qs, kCapInvalidSegments, size + 1, kMaxInvalidSegments);
        return;
    }
    v[size] = s;
    ++size;
}

// Registra le corse non valide che un salto I apre sul vertice attivo @p idx.
THESEUS_HD inline void vd_invalidate_i_jump(QueryState &qs, int idx, int diag) {
    const int gapo = qs.vd_gapo, gape = qs.vd_gape;
    InvalidSeg s;

    s.rem_down = gapo + gape;
    s.rem_up = gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_m_invalid[idx], qs.vd_m_invalid_size[idx], s);

    s.rem_down = 2 * gapo + 3 * gape;
    s.rem_up = gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_i_invalid[idx], qs.vd_i_invalid_size[idx], s);

    s.rem_down = gapo + gape;
    s.rem_up = gapo + 2 * gape;
    s.start_d = diag;
    s.end_d = diag - 1;
    vd_invalid_push(qs, qs.vd_d_invalid[idx], qs.vd_d_invalid_size[idx], s);
}

// Registra le corse non valide che un salto M apre sul vertice attivo @p idx.
THESEUS_HD inline void vd_invalidate_m_jump(QueryState &qs, int idx, int diag) {
    const int gapo = qs.vd_gapo, gape = qs.vd_gape;
    InvalidSeg s;

    s.rem_down = gapo + gape;
    s.rem_up = gapo + gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_m_invalid[idx], qs.vd_m_invalid_size[idx], s);

    s.rem_down = 2 * (gapo + gape);
    s.rem_up = gapo + gape;
    s.start_d = diag + 1;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_i_invalid[idx], qs.vd_i_invalid_size[idx], s);

    s.rem_down = gapo + gape;
    s.rem_up = 2 * (gapo + gape);
    s.start_d = diag;
    s.end_d = diag - 1;
    vd_invalid_push(qs, qs.vd_d_invalid[idx], qs.vd_d_invalid_size[idx], s);
}

// Se @p diag sta fuori da ogni corsa non valida della matrice data per @p vtx.
THESEUS_HD inline bool vd_valid_diagonal(const QueryState &qs, Cell::Matrix matrix, int vtx,
                              int diag) {
    const int idx = qs.vd_vertex_to_idx[vtx];
    const InvalidSeg *v;
    int size;
    if (matrix == Cell::Matrix::M) {
        v = qs.vd_m_invalid[idx];
        size = qs.vd_m_invalid_size[idx];
    } else if (matrix == Cell::Matrix::I) {
        v = qs.vd_i_invalid[idx];
        size = qs.vd_i_invalid_size[idx];
    } else {
        v = qs.vd_d_invalid[idx];
        size = qs.vd_d_invalid_size[idx];
    }
    for (int l = 0; l < size; ++l) {
        if (v[l].start_d <= diag && diag <= v[l].end_d) {
            return false;
        }
    }
    return true;
}

// Liste di posizioni dei salti di un vertice nello slot di uno score, e le taglie.
THESEUS_HD inline int64_t *vd_m_jumps(QueryState &qs, int vtx, int pos) { return qs.vd_m_jumps_pos[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int64_t *vd_i_jumps(QueryState &qs, int vtx, int pos) { return qs.vd_i_jumps_pos[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int32_t &vd_m_jumps_size(QueryState &qs, int vtx, int pos) { return qs.vd_m_jumps_pos_size[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int32_t &vd_i_jumps_size(QueryState &qs, int vtx, int pos) { return qs.vd_i_jumps_pos_size[qs.vd_vertex_to_idx[vtx]][pos]; }

THESEUS_HD inline void vd_jumps_push(QueryState &qs, int64_t *arr, int32_t &size, int64_t val) {
    if (size >= kMaxJumpsPerScore) {
        cap_fail(qs, kCapJumpsPerScore, size + 1, kMaxJumpsPerScore);
        return;
    }
    arr[size] = val;
    ++size;
}

}  // namespace theseus
