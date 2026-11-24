// Time an FM-index as the source of a count prior: the build, the counts of
// held-out contexts, and the follower histograms the prior needs.
//
// The index is sdsl-lite's csa_wt (a Huffman-shaped wavelet tree over RRR bit
// vectors) built on the REVERSED text. In it, the suffixes that start with
// reverse(c) form one interval, and the symbols of the BWT in that interval
// are the characters that FOLLOW c in the text: the follower histogram of c
// is that interval's distinct symbols and their counts (interval_symbols).
// Extending c by one character at its end, c -> c a, is one backward-search
// step on the reversed text (a is prepended to reverse(c)), and the ranks the
// histogram returns for a give the new interval directly.
//
// So the held-out text is walked by start position: from each start j the
// context H[j..j+k-1] grows one character at a time, k = 1..depth, and each
// step yields the histogram that predicts H[j+k] at depth k. Every (position,
// depth) pair the prior needs is visited once, and the walk from j stops when
// the context no longer occurs.
//
// usage:
//   fm_time build <text> <index out> <tmp dir>
//   fm_time query <index> <held-out text> [depth=24] [reps=2]
//   fm_time check <index> <text> <held-out text> [samples=300] [depth=24]
//
// Output lines are "key value" pairs, one per line, for the README's tables.

#include <sdsl/suffix_arrays.hpp>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sdsl;
// SA and ISA samples are only for locate and extract, which the prior never
// does, so they are sampled very sparsely.
//
// The wavelet tree's bit vectors: RRR (compressed, mobius's design) by
// default; with -DFM_PLAIN, plain bit vectors with rank support, which are
// larger but answer a rank without decoding a block.
#ifdef FM_PLAIN
using csa_type = csa_wt<wt_huff<bit_vector>, 1u << 20, 1u << 20>;
static const char * const kVariant = "wt_huff<bit_vector>";
#else
using csa_type = csa_wt<wt_huff<rrr_vector<127>>, 1u << 20, 1u << 20>;
static const char * const kVariant = "wt_huff<rrr_vector<127>>";
#endif
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

static long peak_rss_kb()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

static std::string read_file(const std::string & path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "cannot read " << path << "\n"; std::exit(1); }
    return std::string(std::istreambuf_iterator<char>(in), {});
}

static int build(const std::string & text_path, const std::string & index_path, const std::string & tmp_dir)
{
    auto t0 = clk::now();
    std::string text = read_file(text_path);
    for (unsigned char ch : text)
        if (ch == 0) { std::cerr << "the text has a zero byte\n"; return 1; }
    std::reverse(text.begin(), text.end());
    std::string rev_path = tmp_dir + "/reversed.txt";
    {
        std::ofstream out(rev_path, std::ios::binary);
        out.write(text.data(), (std::streamsize)text.size());
    }
    size_t n = text.size();
    std::string().swap(text);
    auto t1 = clk::now();

    csa_type csa;
    cache_config config(true, tmp_dir, "fm");
    construct(csa, rev_path, config, 1);
    auto t2 = clk::now();
    store_to_file(csa, index_path);
    auto t3 = clk::now();
    std::remove(rev_path.c_str());

    double total = (double)size_in_bytes(csa);
    double wt = (double)size_in_bytes(csa.wavelet_tree);
    std::cout << "variant " << kVariant << "\n"
              << "text_chars " << n << "\n"
              << "read_reverse_s " << secs(t0, t1) << "\n"
              << "construct_s " << secs(t1, t2) << "\n"
              << "store_s " << secs(t2, t3) << "\n"
              << "peak_rss_mb " << peak_rss_kb() / 1024.0 << "\n"
              << "index_bytes " << (size_t)total << "\n"
              << "index_bits_per_char " << 8.0 * total / n << "\n"
              << "wt_bytes " << (size_t)wt << "\n"
              << "wt_bits_per_char " << 8.0 * wt / n << "\n"
              << "sigma " << csa.sigma << "\n";
    return 0;
}

struct Range { csa_type::size_type l, r; };  // inclusive; empty when l > r

// One backward-search step: the interval of a + pattern, from pattern's.
static inline bool step(const csa_type & csa, Range & g, unsigned char a)
{
    csa_type::size_type l, r;
    backward_search(csa, g.l, g.r, (csa_type::char_type)a, l, r);
    g = {l, r};
    return l <= r;
}

static int query(const std::string & index_path, const std::string & held_path, int depth, int reps)
{
    csa_type csa;
    auto t0 = clk::now();
    if (!load_from_file(csa, index_path)) { std::cerr << "cannot load " << index_path << "\n"; return 1; }
    auto t1 = clk::now();
    std::string h = read_file(held_path);
    const size_t n = h.size();
    const Range full{0, csa.size() - 1};

    std::cout << "variant " << kVariant << "\n"
              << "load_s " << secs(t0, t1) << "\n"
              << "heldout_chars " << n << "\n"
              << "depth " << depth << "\n";

    // Counts: backward steps only, as deep as the context occurs.
    std::vector<uint64_t> reached(depth + 1, 0);  // starts whose walk reached exactly depth k
    uint64_t steps = 0, checksum = 0;
    double best_counts = 1e30;
    for (int rep = 0; rep < reps; ++rep)
    {
        std::fill(reached.begin(), reached.end(), 0);
        steps = 0;
        checksum = 0;
        auto a = clk::now();
        for (size_t j = 0; j < n; ++j)
        {
            Range g = full;
            int k = 0;
            while (k < depth && j + k < n)
            {
                ++steps;
                if (!step(csa, g, (unsigned char)h[j + k])) break;
                ++k;
                checksum += g.r - g.l + 1;
            }
            ++reached[k];
        }
        best_counts = std::min(best_counts, secs(a, clk::now()));
    }
    std::cout << "counts_s " << best_counts << "\n"
              << "counts_us_per_char " << 1e6 * best_counts / n << "\n"
              << "counts_steps " << steps << "\n"
              << "counts_us_per_step " << 1e6 * best_counts / steps << "\n"
              << "counts_checksum " << checksum << "\n";
    double mean_depth = 0;
    for (int k = 0; k <= depth; ++k) mean_depth += (double)k * reached[k];
    std::cout << "mean_depth_reached " << mean_depth / n << "\n";
    for (int k = 0; k <= depth; ++k) std::cout << "reached_" << k << " " << reached[k] << "\n";

    // Histograms: at each depth k = 1..depth, the followers of the context
    // and their counts; the next interval comes from the follower's ranks.
    const size_t sigma = 256;
    std::vector<csa_type::wavelet_tree_type::value_type> cs(sigma);
    std::vector<csa_type::size_type> ri(sigma), rj(sigma);
    uint64_t histograms = 0, symbols = 0, hsum = 0;
    double best_hist = 1e30;
    for (int rep = 0; rep < reps; ++rep)
    {
        histograms = symbols = hsum = 0;
        auto a = clk::now();
        for (size_t j = 0; j < n; ++j)
        {
            Range g = full;
            if (!step(csa, g, (unsigned char)h[j])) continue;
            for (int k = 1; k <= depth && j + k < n; ++k)
            {
                csa_type::size_type m = 0;
                csa.wavelet_tree.interval_symbols(g.l, g.r + 1, m, cs, ri, rj);
                ++histograms;
                symbols += m;
                const unsigned char next = (unsigned char)h[j + k];
                bool found = false;
                for (csa_type::size_type s = 0; s < m; ++s)
                {
                    if (cs[s] == 0) continue;  // the sentinel: the context ends the text
                    hsum += rj[s] - ri[s];
                    if (cs[s] == next)
                    {
                        auto c_begin = csa.C[csa.char2comp[next]];
                        g = {c_begin + ri[s], c_begin + rj[s] - 1};
                        found = true;
                    }
                }
                if (!found || k == depth) break;
            }
        }
        best_hist = std::min(best_hist, secs(a, clk::now()));
    }
    std::cout << "hist_s " << best_hist << "\n"
              << "hist_us_per_char " << 1e6 * best_hist / n << "\n"
              << "histograms " << histograms << "\n"
              << "hist_us_per_histogram " << 1e6 * best_hist / histograms << "\n"
              << "mean_symbols_per_histogram " << (double)symbols / histograms << "\n"
              << "hist_checksum " << hsum << "\n"
              << "peak_rss_mb " << peak_rss_kb() / 1024.0 << "\n";
    return 0;
}

// Brute force: count every occurrence of ctx + a in the text by scanning.
static std::vector<uint64_t> scan_followers(const std::string & text, const std::string & ctx)
{
    std::vector<uint64_t> f(256, 0);
    const char * p = text.data();
    const char * end = p + text.size();
    while (true)
    {
        const void * hit = memmem(p, (size_t)(end - p), ctx.data(), ctx.size());
        if (!hit) break;
        const char * q = (const char *)hit;
        if (q + ctx.size() < end) ++f[(unsigned char)q[ctx.size()]];
        p = q + 1;
    }
    return f;
}

static int check(const std::string & index_path, const std::string & text_path, const std::string & held_path,
                 int samples, int depth)
{
    csa_type csa;
    if (!load_from_file(csa, index_path)) { std::cerr << "cannot load " << index_path << "\n"; return 1; }
    std::string text = read_file(text_path);
    std::string h = read_file(held_path);
    std::mt19937_64 rng(1);
    std::vector<csa_type::wavelet_tree_type::value_type> cs(256);
    std::vector<csa_type::size_type> ri(256), rj(256);
    int checked = 0, bad = 0, empty = 0;
    for (int s = 0; s < samples; ++s)
    {
        size_t j = rng() % (h.size() - depth - 1);
        // Half the samples are short, since most deep contexts don't occur in a small text.
        int k = 1 + (int)(rng() % (s % 2 ? depth : std::min(depth, 8)));
        std::string ctx = h.substr(j, (size_t)k);
        // The index's histogram, walked from j as query() does.
        Range g{0, csa.size() - 1};
        bool occurs = true;
        for (int i = 0; i < k && occurs; ++i) occurs = step(csa, g, (unsigned char)ctx[(size_t)i]);
        std::vector<uint64_t> got(256, 0);
        if (occurs)
        {
            csa_type::size_type m = 0;
            csa.wavelet_tree.interval_symbols(g.l, g.r + 1, m, cs, ri, rj);
            for (csa_type::size_type t = 0; t < m; ++t)
                if (cs[t] != 0) got[cs[t]] = rj[t] - ri[t];
        }
        else
            ++empty;
        std::vector<uint64_t> want = scan_followers(text, ctx);
        ++checked;
        if (got != want)
        {
            ++bad;
            if (bad <= 5) std::cout << "MISMATCH j=" << j << " k=" << k << "\n";
        }
    }
    std::cout << "checked " << checked << "\n"
              << "absent_contexts " << empty << "\n"
              << "mismatches " << bad << "\n";
    return bad == 0 ? 0 : 2;
}

int main(int argc, char ** argv)
{
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "build" && argc == 5) return build(argv[2], argv[3], argv[4]);
    if (mode == "query" && (argc >= 4 && argc <= 6))
        return query(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 24, argc > 5 ? std::atoi(argv[5]) : 2);
    if (mode == "check" && (argc >= 5 && argc <= 7))
        return check(argv[2], argv[3], argv[4], argc > 5 ? std::atoi(argv[5]) : 300, argc > 6 ? std::atoi(argv[6]) : 24);
    std::cerr << "usage: fm_time build <text> <index out> <tmp dir>\n"
                 "       fm_time query <index> <held-out text> [depth=24] [reps=2]\n"
                 "       fm_time check <index> <text> <held-out text> [samples=300] [depth=24]\n";
    return 1;
}
