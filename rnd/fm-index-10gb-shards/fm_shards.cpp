// The count prior from several FM-indexes at once: one index per subject
// shard, their counts added per symbol. Counts add, so the sum over the
// shards is exactly the histogram of the text they partition.
//
// Each index is sdsl-lite's csa_wt over plain bit vectors, as in
// rnd/fm-index-1gb/fm_time.cpp. A shard has two:
//   - rev, built on the REVERSED shard text: the FOLLOWERS of a context c are
//     the BWT symbols in the interval of reverse(c), and growing c by one
//     character at its end is one backward-search step;
//   - fwd, built on the text as it stands: the PREDECESSORS of c are the BWT
//     symbols in c's interval, and growing c by one character at its start
//     is one backward-search step.
// The code is the same for both: a predecessor pass walks the reversed
// held-out text in the fwd indexes. The wavelet tree's symbols are the
// text's own bytes; 0 is sdsl's end-of-text symbol.
//
// In a shard every book is preceded and followed by 0x01, a byte outside the
// alphabet, so no context runs across a book's edge, and 0x01 shows up only
// as a follower at a book's end or a predecessor at its start. The summed
// histograms drop it and the end-of-text symbol.
//
// usage:
//   fm_shards build <text> <index out> <tmp dir> [rev|fwd]
//   fm_shards query <index list> <held-out text> [depth=24] [reps=2] [per_shard=1] [followers|predecessors]
//   fm_shards check <index list> <whole text> <held-out text> [samples=300] [depth=24] [threads=4]
//                   [followers|predecessors]
//
// An index list is a file with one index path per line. Output lines are
// "key value" pairs, one per line, for the README's tables.

#include <sdsl/suffix_arrays.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace sdsl;
// SA and ISA samples are only for locate and extract, which the prior never
// does, so they are sampled very sparsely.
using csa_type = csa_wt<wt_huff<bit_vector>, 1u << 20, 1u << 20>;
static const char * const kVariant = "wt_huff<bit_vector>";
using clk = std::chrono::steady_clock;

static const unsigned char kSeparator = 0x01;  // after every book

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

static int build(const std::string & text_path, const std::string & index_path, const std::string & tmp_dir,
                 bool reversed)
{
    auto t0 = clk::now();
    std::string text = read_file(text_path);
    size_t separators = 0;
    for (unsigned char ch : text)
    {
        if (ch == 0) { std::cerr << "the text has a zero byte\n"; return 1; }
        separators += ch == kSeparator;
    }
    std::string build_path = text_path;
    if (reversed)
    {
        std::reverse(text.begin(), text.end());
        build_path = tmp_dir + "/reversed.txt";
        std::ofstream out(build_path, std::ios::binary);
        out.write(text.data(), (std::streamsize)text.size());
    }
    size_t n = text.size();
    std::string().swap(text);
    auto t1 = clk::now();

    csa_type csa;
    cache_config config(true, tmp_dir, "fm");
    construct(csa, build_path, config, 1);
    auto t2 = clk::now();
    store_to_file(csa, index_path);
    auto t3 = clk::now();
    if (reversed) std::remove(build_path.c_str());

    double total = (double)size_in_bytes(csa);
    std::cout << "variant " << kVariant << "\n"
              << "index_text " << (reversed ? "reversed" : "forward") << "\n"
              << "text_chars " << n << "\n"
              << "separators " << separators << "\n"
              << "read_reverse_s " << secs(t0, t1) << "\n"
              << "construct_s " << secs(t1, t2) << "\n"
              << "store_s " << secs(t2, t3) << "\n"
              << "build_s " << secs(t0, t3) << "\n"
              << "peak_rss_mb " << peak_rss_kb() / 1024.0 << "\n"
              << "index_bytes " << (size_t)total << "\n"
              << "index_bits_per_char " << 8.0 * total / n << "\n"
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

static std::vector<std::unique_ptr<csa_type>> load_list(const std::string & list_path,
                                                         std::vector<std::string> & paths)
{
    std::vector<std::unique_ptr<csa_type>> v;
    std::ifstream in(list_path);
    if (!in) { std::cerr << "cannot read " << list_path << "\n"; std::exit(1); }
    std::string p;
    while (std::getline(in, p))
    {
        if (p.empty()) continue;
        auto c = std::make_unique<csa_type>();
        if (!load_from_file(*c, p)) { std::cerr << "cannot load " << p << "\n"; std::exit(1); }
        v.push_back(std::move(c));
        paths.push_back(p);
    }
    return v;
}

// Scratch for interval_symbols, sized for a byte alphabet.
struct Scratch
{
    std::vector<csa_type::wavelet_tree_type::value_type> cs = std::vector<csa_type::wavelet_tree_type::value_type>(256);
    std::vector<csa_type::size_type> ri = std::vector<csa_type::size_type>(256);
    std::vector<csa_type::size_type> rj = std::vector<csa_type::size_type>(256);
};

// The summed histograms of one start: sum[k * 256 + symbol], and the entries
// made nonzero, so reading and clearing them touches only those.
struct Acc
{
    std::vector<uint64_t> sum;
    std::vector<uint32_t> touched;
    explicit Acc(int depth) : sum((size_t)(depth + 1) * 256, 0) { touched.reserve(4096); }
};

// The walk from start j in one shard: at each depth k = 1..depth, the
// followers of H[j..j+k-1] and their counts, each added to acc's sum[k][symbol]
// (when acc is given) and to count; the next interval comes from the
// follower's ranks. Returns the depth reached: the longest context from j, up
// to depth, that occurs in the shard. symbols gets the number of distinct
// followers seen.
static inline int walk(const csa_type & csa, const std::string & h, size_t j, int depth, Scratch & s,
                       Acc * acc, uint64_t & count, uint64_t & symbols)
{
    const size_t n = h.size();
    Range g{0, csa.size() - 1};
    if (!step(csa, g, (unsigned char)h[j])) return 0;
    int k = 1;
    for (; k <= depth && j + k < n; ++k)
    {
        csa_type::size_type m = 0;
        csa.wavelet_tree.interval_symbols(g.l, g.r + 1, m, s.cs, s.ri, s.rj);
        symbols += m;
        const unsigned char next = (unsigned char)h[j + k];
        const uint32_t base = (uint32_t)k * 256;
        bool found = false;
        for (csa_type::size_type t = 0; t < m; ++t)
        {
            const unsigned char c = (unsigned char)s.cs[t];
            if (c == 0 || c == kSeparator) continue;  // the text's end, a book's end
            count += s.rj[t] - s.ri[t];
            if (acc)
            {
                uint64_t & e = acc->sum[base + c];
                if (e == 0) acc->touched.push_back(base + c);
                e += s.rj[t] - s.ri[t];
            }
            if (c == next)
            {
                auto c_begin = csa.C[csa.char2comp[next]];
                g = {c_begin + s.ri[t], c_begin + s.rj[t] - 1};
                found = true;
            }
        }
        if (!found || k == depth) return k;
    }
    return k - 1;
}

static int query(const std::string & list_path, const std::string & held_path, int depth, int reps, bool per_shard,
                 bool predecessors)
{
    std::vector<std::string> paths;
    auto t0 = clk::now();
    auto shards = load_list(list_path, paths);
    auto t1 = clk::now();
    std::string h = read_file(held_path);
    if (predecessors) std::reverse(h.begin(), h.end());
    const size_t n = h.size();
    double index_bytes = 0;
    for (auto & c : shards) index_bytes += (double)size_in_bytes(*c);

    std::cout << "variant " << kVariant << "\n"
              << "histograms_of " << (predecessors ? "predecessors" : "followers") << "\n"
              << "shards " << shards.size() << "\n"
              << "index_bytes " << (size_t)index_bytes << "\n"
              << "load_s " << secs(t0, t1) << "\n"
              << "heldout_chars " << n << "\n"
              << "depth " << depth << "\n";

    Scratch s;
    Acc acc(depth);

    // Each shard alone, over all the held-out starts (one pass each): what a
    // process holding only that shard would spend, and how deep the shard's
    // contexts reach. The counts are totalled too, to compare with the pass below.
    uint64_t shard_total = 0;
    if (per_shard)
    {
        std::cout << "# shard_<i> text_chars hist_s us_per_char mean_depth_reached mean_symbols_per_histogram\n";
        double all_s = 0;
        for (size_t i = 0; i < shards.size(); ++i)
        {
            const csa_type & csa = *shards[i];
            uint64_t symbols = 0, histograms = 0;
            auto a = clk::now();
            for (size_t j = 0; j < n; ++j)
                histograms += (uint64_t)walk(csa, h, j, depth, s, nullptr, shard_total, symbols);
            double t = secs(a, clk::now());
            all_s += t;
            std::cout << "shard_" << i << " " << csa.size() - 1 << " " << t << " " << 1e6 * t / n << " "
                      << (double)histograms / n << " " << (histograms ? (double)symbols / histograms : 0.0) << "\n";
        }
        std::cout << "per_shard_total_s " << all_s << "\n"
                  << "per_shard_total_us_per_char " << 1e6 * all_s / n << "\n"
                  << "per_shard_count_total " << shard_total << "\n";
    }

    // The summed pass: for each start, every shard's walk, the counts added
    // per depth and symbol; then the summed histograms are read (folded into
    // a checksum) and cleared.
    uint64_t total = 0, walked = 0, checksum = 0, symbols = 0, reached_sum = 0, pooled_reached = 0;
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep)
    {
        total = walked = checksum = symbols = reached_sum = pooled_reached = 0;
        auto a = clk::now();
        for (size_t j = 0; j < n; ++j)
        {
            int maxk = 0;
            for (auto & c : shards)
            {
                int k = walk(*c, h, j, depth, s, &acc, walked, symbols);
                reached_sum += (uint64_t)k;
                maxk = std::max(maxk, k);
            }
            pooled_reached += (uint64_t)maxk;
            for (uint32_t e : acc.touched)
            {
                const uint64_t v = acc.sum[e];
                total += v;
                checksum += v * (uint64_t)((e & 255) + 1) * (uint64_t)(e >> 8);
                acc.sum[e] = 0;
            }
            acc.touched.clear();
        }
        best = std::min(best, secs(a, clk::now()));
    }
    std::cout << "sum_s " << best << "\n"
              << "sum_us_per_char " << 1e6 * best / n << "\n"
              << "sum_count_total " << total << "\n"
              << "sum_count_walked " << walked << "\n"
              << "sum_checksum " << checksum << "\n"
              << "mean_depth_reached_per_shard " << (double)reached_sum / ((double)n * shards.size()) << "\n"
              << "mean_depth_reached_pooled " << (double)pooled_reached / n << "\n"
              << "peak_rss_mb " << peak_rss_kb() / 1024.0 << "\n";
    return 0;
}

// Brute force: count every occurrence of ctx + a (or of a + ctx, with
// predecessors) in the text by scanning.
static std::vector<uint64_t> scan(const char * text, size_t size, const std::string & ctx, bool predecessors)
{
    std::vector<uint64_t> f(256, 0);
    const char * p = text;
    const char * end = text + size;
    while (true)
    {
        const void * hit = memmem(p, (size_t)(end - p), ctx.data(), ctx.size());
        if (!hit) break;
        const char * q = (const char *)hit;
        if (predecessors) { if (q > text) ++f[(unsigned char)q[-1]]; }
        else if (q + ctx.size() < end) ++f[(unsigned char)q[ctx.size()]];
        p = q + 1;
    }
    return f;
}

static int check(const std::string & list_path, const std::string & text_path, const std::string & held_path,
                 int samples, int depth, int threads, bool predecessors)
{
    auto t0 = clk::now();
    std::vector<std::string> paths;
    auto shards = load_list(list_path, paths);
    int fd = open(text_path.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "cannot read " << text_path << "\n"; return 1; }
    struct stat st{};
    fstat(fd, &st);
    const size_t size = (size_t)st.st_size;
    const char * text = (const char *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (text == MAP_FAILED) { std::cerr << "cannot map " << text_path << "\n"; return 1; }
    std::string h = read_file(held_path);
    // With predecessors the fwd indexes are walked along the reversed
    // held-out text: a drawn string is a context reversed.
    if (predecessors) std::reverse(h.begin(), h.end());

    // The contexts, drawn as rnd/fm-index-1gb's check draws them (same seed),
    // and the shards' summed histogram of each.
    std::mt19937_64 rng(1);
    Scratch s;
    std::vector<std::string> ctxs;
    std::vector<std::vector<uint64_t>> got;
    int absent = 0;
    for (int i = 0; i < samples; ++i)
    {
        size_t j = rng() % (h.size() - depth - 1);
        // Half the samples are short, since most deep contexts don't occur.
        int k = 1 + (int)(rng() % (i % 2 ? depth : std::min(depth, 8)));
        std::string ctx = h.substr(j, (size_t)k);
        std::vector<uint64_t> g(256, 0);
        bool anywhere = false;
        for (auto & c : shards)
        {
            Range r{0, c->size() - 1};
            bool occurs = true;
            for (int t = 0; t < k && occurs; ++t) occurs = step(*c, r, (unsigned char)ctx[(size_t)t]);
            if (!occurs) continue;
            anywhere = true;
            csa_type::size_type m = 0;
            c->wavelet_tree.interval_symbols(r.l, r.r + 1, m, s.cs, s.ri, s.rj);
            for (csa_type::size_type t = 0; t < m; ++t)
                if (s.cs[t] != 0) g[(unsigned char)s.cs[t]] += s.rj[t] - s.ri[t];
        }
        absent += !anywhere;
        ctxs.push_back(ctx);
        got.push_back(g);
    }

    // The scans, which take most of the time, split over threads.
    std::vector<std::vector<uint64_t>> want(ctxs.size());
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (size_t i = (size_t)t; i < ctxs.size(); i += (size_t)threads)
            {
                std::string c = ctxs[i];
                if (predecessors) std::reverse(c.begin(), c.end());
                want[i] = scan(text, size, c, predecessors);
            }
        });
    for (auto & th : pool) th.join();

    int bad = 0;
    uint64_t sep_followers = 0;
    for (size_t i = 0; i < ctxs.size(); ++i)
    {
        sep_followers += want[i][kSeparator];
        if (got[i] != want[i])
        {
            ++bad;
            if (bad <= 5) std::cout << "MISMATCH sample=" << i << " k=" << ctxs[i].size() << "\n";
        }
    }
    std::cout << "histograms_of " << (predecessors ? "predecessors" : "followers") << "\n"
              << "shards " << shards.size() << "\n"
              << "text_chars " << size << "\n"
              << "checked " << ctxs.size() << "\n"
              << "absent_contexts " << absent << "\n"
              << "separator_counts " << sep_followers << "\n"
              << "mismatches " << bad << "\n"
              << "check_s " << secs(t0, clk::now()) << "\n"
              << "peak_rss_mb " << peak_rss_kb() / 1024.0 << "\n";
    return bad == 0 ? 0 : 2;
}

int main(int argc, char ** argv)
{
    std::string mode = argc > 1 ? argv[1] : "";
    auto dir = [&](int i, const char * yes, const char * no) {
        if (argc <= i) return 0;
        if (std::strcmp(argv[i], yes) == 0) return 1;
        if (std::strcmp(argv[i], no) == 0) return 0;
        return -1;
    };
    if (mode == "build" && (argc == 5 || argc == 6) && dir(5, "fwd", "rev") >= 0)
        return build(argv[2], argv[3], argv[4], dir(5, "fwd", "rev") == 0);
    if (mode == "query" && argc >= 4 && argc <= 8 && dir(7, "predecessors", "followers") >= 0)
        return query(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 24, argc > 5 ? std::atoi(argv[5]) : 2,
                     argc > 6 ? std::atoi(argv[6]) != 0 : true, dir(7, "predecessors", "followers") == 1);
    if (mode == "check" && argc >= 5 && argc <= 9 && dir(8, "predecessors", "followers") >= 0)
        return check(argv[2], argv[3], argv[4], argc > 5 ? std::atoi(argv[5]) : 300, argc > 6 ? std::atoi(argv[6]) : 24,
                     argc > 7 ? std::atoi(argv[7]) : 4, dir(8, "predecessors", "followers") == 1);
    std::cerr << "usage: fm_shards build <text> <index out> <tmp dir> [rev|fwd]\n"
                 "       fm_shards query <index list> <held-out text> [depth=24] [reps=2] [per_shard=1]"
                 " [followers|predecessors]\n"
                 "       fm_shards check <index list> <whole text> <held-out text> [samples=300] [depth=24]"
                 " [threads=4] [followers|predecessors]\n";
    return 1;
}
