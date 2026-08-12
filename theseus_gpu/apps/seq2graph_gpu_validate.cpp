#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "theseus/alignment.h"
#include "theseus/penalties.h"
#include "theseus/theseus_aligner.h"

namespace {

struct QueryCase {
  std::string sequence;
  std::string start_node;
  int start_offset;
};

int score_alignment(theseus::Alignment alignment) {
  theseus::Penalties penalties(0, 2, 3, 1);
  return alignment.compute_affine_gap_score(penalties);
}

bool same_alignment(const theseus::Alignment &a, const theseus::Alignment &b) {
  return score_alignment(a) == score_alignment(b) &&
         a.start_offset == b.start_offset &&
         a.end_offset == b.end_offset &&
         a.query_length == b.query_length &&
         a.path == b.path &&
         a.edit_op == b.edit_op;
}

std::string graph_text() {
  return "S\t1\tACGT\n"
         "S\t2\tTGCA\n"
         "L\t1\t+\t2\t+\t0M\n";
}

std::string long_graph_text(int n) {
  std::string g = "S\t1\t";
  g.append(static_cast<size_t>(n), 'A');
  g += "\n";
  return g;
}

bool run_batch(const std::string &name, const std::vector<QueryCase> &cases,
               bool require_device, bool expect_workspace_overflow) {
  std::stringstream graph_cpu(graph_text());
  std::stringstream graph_gpu(graph_text());
  theseus::Penalties penalties(0, 2, 3, 1);
  theseus::TheseusAligner cpu(penalties, graph_cpu);
  theseus::TheseusAligner gpu(penalties, graph_gpu);

  std::vector<std::string> seqs;
  std::vector<std::string> starts;
  std::vector<int> offsets;
  seqs.reserve(cases.size());
  starts.reserve(cases.size());
  offsets.reserve(cases.size());
  for (const QueryCase &c : cases) {
    seqs.push_back(c.sequence);
    starts.push_back(c.start_node);
    offsets.push_back(c.start_offset);
  }

  std::vector<theseus::Alignment> cpu_alignments;
  cpu_alignments.reserve(cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    cpu_alignments.push_back(cpu.align(seqs[i], starts[i], offsets[i]));
  }

  theseus::GpuBatchReport report;
  std::vector<theseus::Alignment> gpu_alignments =
      gpu.align_batch_gpu(seqs, starts, offsets, &report, 128);

  // Run the identical batch again on the same aligner. Besides checking that
  // stale QueryState contents cannot affect correctness, this exercises the
  // persistent device workspace at stable query and batch capacities.
  theseus::GpuBatchReport reuse_report;
  std::vector<theseus::Alignment> reused_alignments =
      gpu.align_batch_gpu(seqs, starts, offsets, &reuse_report, 128);

  bool ok = true;
  if (require_device && !report.aligned_on_device) {
    std::cerr << name << ": CUDA alignment did not run: " << report.message
              << "\n";
    ok = false;
  }
  if (expect_workspace_overflow && report.aligned_on_device &&
      !report.query_state_capacity_exceeded) {
    std::cerr << name << ": expected QueryState capacity overflow, got: "
              << report.message << "\n";
    ok = false;
  }
  if (gpu_alignments.size() != cpu_alignments.size() ||
      reused_alignments.size() != cpu_alignments.size()) {
    std::cerr << name << ": result count mismatch\n";
    return false;
  }
  for (size_t i = 0; i < cpu_alignments.size(); ++i) {
    if (!same_alignment(cpu_alignments[i], gpu_alignments[i])) {
      std::cerr << name << ": alignment mismatch at query " << i
                << " cpu_score=" << score_alignment(cpu_alignments[i])
                << " gpu_score=" << score_alignment(gpu_alignments[i]) << "\n";
      ok = false;
    }
    if (!same_alignment(cpu_alignments[i], reused_alignments[i])) {
      std::cerr << name << ": reused-workspace alignment mismatch at query "
                << i << "\n";
      ok = false;
    }
  }
  std::cerr << name << ": " << report.message << "\n";
  return ok;
}

bool run_workspace_case(bool require_device) {
  // The point of this case is that an overflow is detected and reported, so the
  // input has to exceed whatever the current capacity is rather than a value
  // that happened to exceed it once. A vertex and a query of this length need
  // 2 * span + 1 diagonals, which cannot fit in span.
  const int kLong = theseus::scratchpad_span();
  std::stringstream graph_cpu(long_graph_text(kLong));
  std::stringstream graph_gpu(long_graph_text(kLong));
  theseus::Penalties penalties(0, 2, 3, 1);
  theseus::TheseusAligner cpu(penalties, graph_cpu);
  theseus::TheseusAligner gpu(penalties, graph_gpu);
  std::vector<std::string> seqs(1, std::string(kLong, 'A'));
  std::vector<std::string> starts(1, "1+");
  std::vector<int> offsets(1, 0);

  theseus::Alignment cpu_alignment = cpu.align(seqs[0], starts[0], offsets[0]);
  theseus::GpuBatchReport report;
  std::vector<theseus::Alignment> gpu_alignments =
      gpu.align_batch_gpu(seqs, starts, offsets, &report, 128);

  bool ok = same_alignment(cpu_alignment, gpu_alignments[0]);
  if (require_device && !report.aligned_on_device) {
    std::cerr << "workspace_overflow: CUDA alignment did not run: "
              << report.message << "\n";
    ok = false;
  }
  if (report.aligned_on_device && !report.query_state_capacity_exceeded) {
    std::cerr << "workspace_overflow: expected overflow, got: "
              << report.message << "\n";
    ok = false;
  }
  std::cerr << "workspace_overflow: " << report.message << "\n";
  return ok;
}

bool run_invalid_start_case() {
  std::stringstream graph_gpu(graph_text());
  theseus::Penalties penalties(0, 2, 3, 1);
  theseus::TheseusAligner gpu(penalties, graph_gpu);
  std::vector<std::string> seqs(1, "ACGT");
  std::vector<std::string> starts(1, "missing+");
  std::vector<int> offsets(1, 0);
  try {
    theseus::GpuBatchReport report;
    (void)gpu.align_batch_gpu(seqs, starts, offsets, &report);
  } catch (const std::out_of_range &) {
    return true;
  }
  std::cerr << "invalid_start: expected out_of_range\n";
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  const bool require_device = argc > 1 && std::string(argv[1]) == "--require-device";

  std::vector<QueryCase> basic = {
      {"ACGT", "1+", 0},      // exact
      {"AGGT", "1+", 0},      // mismatch
      {"ACGTT", "1+", 0},     // insertion relative to graph
      {"ACG", "1+", 0},       // deletion relative to graph
      {"ACGTTGCA", "1+", 0},  // crosses edge 1+ -> 2+
      {"TGCA", "2+", 0},
  };

  std::vector<QueryCase> large_batch;
  large_batch.reserve(129);
  for (int i = 0; i < 129; ++i) {
    large_batch.push_back((i % 2 == 0) ? QueryCase{"ACGT", "1+", 0}
                                       : QueryCase{"TGCA", "2+", 0});
  }

  bool ok = true;
  std::cerr << "CPU baseline versus GPU kernel validation\n";
  ok = run_batch("basic_cases", basic, require_device, false) && ok;
  ok = run_batch("batch_gt_block", large_batch, require_device, false) && ok;
  ok = run_workspace_case(require_device) && ok;
  ok = run_invalid_start_case() && ok;

  return ok ? 0 : 1;
}
