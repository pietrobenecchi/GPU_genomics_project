#include <getopt.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "theseus/alignment.h"
#include "theseus/penalties.h"
#include "theseus/theseus_aligner.h"

enum class Backend { Cpu, Gpu };

struct CmdArgs {
  int match = 0;
  int mismatch = 2;
  int gapo = 3;
  int gape = 1;
  std::string graph_file;
  std::string sequences_file;
  std::string output_file;
  Backend backend = Backend::Cpu;
};

// Sequences to align, together with where each one starts in the graph.
struct Inputs {
  std::vector<std::string> sequences;
  std::vector<std::string> start_vertices;
  std::vector<int> start_offsets;
};

void help() {
  std::cout
      << "Usage: seq2graph_proxy [OPTIONS]\n"
      << "Options:\n"
      << "  -m, --match <int>            Match penalty (default 0)\n"
      << "  -x, --mismatch <int>         Mismatch penalty (default 2)\n"
      << "  -o, --gapo <int>             Gap-open penalty (default 3)\n"
      << "  -e, --gape <int>             Gap-extend penalty (default 1)\n"
      << "  -g, --graph_file <file>      Graph file in GFA format (required)\n"
      << "  -s, --sequences_file <file>  FASTA with start metadata (required)\n"
      << "  -f, --output_file <file>     Output GAF path (required)\n"
      << "  -b, --backend <cpu|gpu>      Alignment backend (default cpu)\n";
}

CmdArgs parse_args(int argc, char *const *argv) {
  static const option long_options[] = {
      {"match", required_argument, nullptr, 'm'},
      {"mismatch", required_argument, nullptr, 'x'},
      {"gapo", required_argument, nullptr, 'o'},
      {"gape", required_argument, nullptr, 'e'},
      {"graph_file", required_argument, nullptr, 'g'},
      {"sequences_file", required_argument, nullptr, 's'},
      {"output_file", required_argument, nullptr, 'f'},
      {"backend", required_argument, nullptr, 'b'},
      {nullptr, 0, nullptr, 0}};

  CmdArgs args;
  int option_index = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "m:x:o:e:g:s:f:b:", long_options,
                            &option_index)) != -1) {
    switch (opt) {
      case 'm':
        args.match = std::stoi(optarg);
        break;
      case 'x':
        args.mismatch = std::stoi(optarg);
        break;
      case 'o':
        args.gapo = std::stoi(optarg);
        break;
      case 'e':
        args.gape = std::stoi(optarg);
        break;
      case 'g':
        args.graph_file = optarg;
        break;
      case 's':
        args.sequences_file = optarg;
        break;
      case 'f':
        args.output_file = optarg;
        break;
      case 'b': {
        const std::string backend = optarg;
        if (backend == "cpu") {
          args.backend = Backend::Cpu;
        } else if (backend == "gpu") {
          args.backend = Backend::Gpu;
        } else {
          std::cerr << "Invalid backend: " << backend << " (expected cpu or gpu)\n";
          std::exit(1);
        }
        break;
      }
      default:
        std::cerr << "Invalid option\n";
        std::exit(1);
    }
  }

  return args;
}

void read_seq_pos_data(std::ifstream &sp_file, Inputs &inputs) {
  std::vector<std::string> &sequences = inputs.sequences;
  std::vector<std::string> &start_vertices = inputs.start_vertices;
  std::vector<int> &start_offsets = inputs.start_offsets;

  if (!sp_file.is_open()) {
    throw std::runtime_error("Could not open sequences file");
  }

  std::string sequence;
  std::string line;
  std::string vertex;
  std::string orientation;
  int offset = 0;
  int count = 0;

  while (std::getline(sp_file, line)) {
    if (line.empty()) {
      continue;
    }

    if (line[0] == '>') {
      std::istringstream iss(line.substr(1));
      if (!(iss >> vertex >> offset)) {
        throw std::runtime_error("Invalid FASTA metadata line: " + line);
      }
      if (!(iss >> orientation)) {
        orientation = "+";
      }
      if (orientation != "+" && orientation != "-") {
        throw std::runtime_error("Invalid orientation in line: " + line);
      }

      start_vertices.push_back(vertex + orientation);
      start_offsets.push_back(offset);

      if (count > 0) {
        sequences.push_back(sequence);
      }
      sequence.clear();
      ++count;
    } else {
      sequence += line;
    }
  }

  if (count > 0) {
    sequences.push_back(sequence);
  }

  if (sequences.size() != start_vertices.size() ||
      sequences.size() != start_offsets.size()) {
    throw std::runtime_error("Malformed sequence/start-position file");
  }
}

/**
 * @brief Align every sequence on the CPU, one at a time.
 */
void run_cpu(theseus::TheseusAligner &aligner, Inputs &inputs,
             std::ostream &out_stream) {
  for (size_t i = 0; i < inputs.sequences.size(); ++i) {
    theseus::Alignment aln = aligner.align(
        inputs.sequences[i], inputs.start_vertices[i], inputs.start_offsets[i]);
    aligner.print_alignment_as_gaf(aln, out_stream, "seq_" + std::to_string(i));
  }
}

/**
 * @brief Align the whole batch through the GPU backend.
 *
 * The backend falls back to the CPU when it cannot do the work, so the status it
 * reports is written to stderr rather than assumed.
 */
void run_gpu(theseus::TheseusAligner &aligner, Inputs &inputs,
             std::ostream &out_stream) {
  theseus::GpuBatchReport report;
  std::vector<theseus::Alignment> alignments = aligner.align_batch_gpu(
      inputs.sequences, inputs.start_vertices, inputs.start_offsets, &report);

  for (size_t i = 0; i < alignments.size(); ++i) {
    aligner.print_alignment_as_gaf(alignments[i], out_stream,
                                   "seq_" + std::to_string(i));
  }

  std::cerr << "GPU backend: " << report.message << "\n";
}

int main(int argc, char *const *argv) {
  CmdArgs args = parse_args(argc, argv);

  if (args.graph_file.empty() || args.sequences_file.empty() ||
      args.output_file.empty()) {
    std::cerr << "Missing required arguments\n";
    help();
    return 1;
  }

  std::ifstream graph_file(args.graph_file);
  std::ifstream seq_file(args.sequences_file);
  std::ofstream output_file(args.output_file);

  if (!graph_file.is_open()) {
    std::cerr << "Could not open graph file: " << args.graph_file << "\n";
    return 1;
  }
  if (!seq_file.is_open()) {
    std::cerr << "Could not open sequences file: " << args.sequences_file
              << "\n";
    return 1;
  }
  if (!output_file.is_open()) {
    std::cerr << "Could not open output file: " << args.output_file << "\n";
    return 1;
  }

  theseus::Penalties penalties(args.match, args.mismatch, args.gapo, args.gape);
  theseus::TheseusAligner aligner(penalties, graph_file);

  Inputs inputs;
  read_seq_pos_data(seq_file, inputs);

  auto start = std::chrono::steady_clock::now();

  if (args.backend == Backend::Gpu) {
    run_gpu(aligner, inputs, output_file);
  } else {
    run_cpu(aligner, inputs, output_file);
  }

  auto end = std::chrono::steady_clock::now();
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Aligned " << inputs.sequences.size() << " sequences in "
            << micros << " microseconds on "
            << (args.backend == Backend::Gpu ? "gpu" : "cpu") << "\n";

  return 0;
}
