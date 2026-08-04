#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "theseus/penalties.h"
#include "theseus/theseus_aligner.h"

namespace {

struct BenchResult {
  double wall_ms = 0.0;
  double alignments_per_s = 0.0;
  double bases_per_s = 0.0;
  theseus::GpuBatchReport report;
};

std::vector<int> batch_sizes() { return {1, 8, 32, 128, 512, 1024}; }

std::vector<std::string> make_queries(int n) {
  std::vector<std::string> seqs;
  seqs.reserve(static_cast<size_t>(n));
  static const char *patterns[] = {"ACGT", "AGGT", "ACGTT", "ACG", "ACGTTGCA"};
  for (int i = 0; i < n; ++i) {
    seqs.emplace_back(patterns[i % 5]);
  }
  return seqs;
}

int total_bases(const std::vector<std::string> &seqs) {
  int total = 0;
  for (const std::string &seq : seqs) {
    total += static_cast<int>(seq.size());
  }
  return total;
}

template <typename Fn>
double time_ms(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

BenchResult run_gpu(theseus::TheseusAligner &aligner,
                    const std::vector<std::string> &seqs,
                    std::vector<std::string> starts,
                    std::vector<int> offsets) {
  BenchResult result;
  result.wall_ms = time_ms([&] {
    (void)aligner.align_batch_gpu(seqs, starts, offsets, &result.report, 128);
  });
  result.alignments_per_s = seqs.size() * 1000.0 / result.wall_ms;
  result.bases_per_s = total_bases(seqs) * 1000.0 / result.wall_ms;
  return result;
}

double run_cpu(theseus::TheseusAligner &aligner,
               const std::vector<std::string> &seqs,
               std::vector<std::string> starts,
               std::vector<int> offsets) {
  return time_ms([&] {
    for (size_t i = 0; i < seqs.size(); ++i) {
      (void)aligner.align(seqs[i], starts[i], offsets[i]);
    }
  });
}

}  // namespace

int main(int argc, char **argv) {
  const std::string graph_path = argc > 1 ? argv[1] : "theseus_gpu/data/sample_graph.gfa";
  std::ifstream graph_file(graph_path);
  if (!graph_file.is_open()) {
    std::cerr << "could not open graph: " << graph_path << "\n";
    return 1;
  }

  theseus::Penalties penalties(0, 2, 3, 1);
  std::cout << "batch,cpu_ms,gpu_wall_ms,gpu_kernel_ms,gpu_align_s,gpu_bases_s,"
               "gpu_status\n";

  for (int n : batch_sizes()) {
    std::vector<std::string> seqs = make_queries(n);
    std::vector<std::string> starts(static_cast<size_t>(n), "1+");
    std::vector<int> offsets(static_cast<size_t>(n), 0);

    std::ifstream cpu_graph(graph_path);
    std::ifstream gpu_graph(graph_path);
    theseus::TheseusAligner cpu(penalties, cpu_graph);
    theseus::TheseusAligner gpu(penalties, gpu_graph);

    const double cpu_ms = run_cpu(cpu, seqs, starts, offsets);
    const BenchResult gpu_result = run_gpu(gpu, seqs, starts, offsets);

    std::cout << n << ',' << cpu_ms << ',' << gpu_result.wall_ms << ','
              << gpu_result.report.kernel_ms << ','
              << gpu_result.alignments_per_s << ','
              << gpu_result.bases_per_s << ",\""
              << gpu_result.report.message << "\"\n";
  }

  return 0;
}
