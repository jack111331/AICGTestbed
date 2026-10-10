#include "pch.h"
#include "Tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

namespace NeuralModelIntegrateTestbed {

namespace {

constexpr char kMagic[8] = {'P', 'T', 'F', 'D', 'T', 'O', 'K', '1'};
constexpr uint32_t kSupportedVersion = 1;

// Must match tools/prepare_flooddiffusion.py.
constexpr uint32_t kFlagWhitespaceSplit = 1u << 0;
constexpr uint32_t kFlagMetaspacePrepend = 1u << 1;
constexpr uint32_t kFlagByteFallback = 1u << 2;

// What an unmatched character scores, relative to the rarest token in the
// vocabulary. SentencePiece uses a fixed penalty below the minimum so that any
// real segmentation wins over falling back to unknown.
constexpr float kUnknownPenalty = 10.0f;

bool IsAsciiSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Length in bytes of the UTF-8 sequence starting at `text[0]`, clamped to what
// remains. Used only to advance past a character no token matched, so a
// malformed byte advancing by one is the right outcome rather than an error.
std::size_t Utf8Length(std::string_view text) {
    if (text.empty()) {
        return 0;
    }
    const unsigned char lead = static_cast<unsigned char>(text[0]);
    std::size_t length = 1;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
    }
    return std::min(length, text.size());
}

template <typename T>
bool ReadScalar(std::ifstream& stream, T* out) {
    stream.read(reinterpret_cast<char*>(out), sizeof(T));
    return static_cast<bool>(stream);
}

}  // namespace

bool UnigramTokenizer::Load(const std::filesystem::path& path, std::string* error) {
    const auto fail = [error](std::string message) {
        if (error != nullptr) {
            *error = std::move(message);
        }
        return false;
    };

    m_entries.clear();
    m_blob.clear();
    m_index.clear();

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail("cannot open " + path.string() +
                    "; run tools/prepare_flooddiffusion.py to produce it");
    }

    char magic[8] = {};
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        return fail(path.string() + " is not a baked tokenizer (bad magic)");
    }

    uint32_t version = 0;
    uint32_t vocabCount = 0;
    uint32_t unkId = 0;
    uint32_t eosId = 0;
    uint32_t maxTokenBytes = 0;
    float minScore = 0.0f;
    uint32_t stringBytes = 0;
    uint32_t flags = 0;
    if (!ReadScalar(stream, &version) || !ReadScalar(stream, &vocabCount) ||
        !ReadScalar(stream, &unkId) || !ReadScalar(stream, &eosId) ||
        !ReadScalar(stream, &maxTokenBytes) || !ReadScalar(stream, &minScore) ||
        !ReadScalar(stream, &stringBytes) || !ReadScalar(stream, &flags)) {
        return fail(path.string() + " ends inside its header");
    }

    if (version != kSupportedVersion) {
        return fail(path.string() + " is version " + std::to_string(version) +
                    "; this build understands version " +
                    std::to_string(kSupportedVersion));
    }
    if (vocabCount == 0 || maxTokenBytes == 0) {
        return fail(path.string() + " declares an empty vocabulary");
    }
    if (unkId >= vocabCount || eosId >= vocabCount) {
        return fail(path.string() + " has an unknown or end-of-sequence id "
                                    "outside its vocabulary");
    }

    // These are the stages Encode actually performs. Refusing anything else is
    // the point: a tokenizer needing a normaliser or a byte fallback would
    // otherwise encode subtly differently from the model's training, and the
    // only symptom would be motion that does not match the prompt.
    if ((flags & kFlagWhitespaceSplit) == 0 || (flags & kFlagMetaspacePrepend) == 0) {
        return fail(path.string() + " was baked from a tokenizer whose "
                                    "pre-tokenizer is not WhitespaceSplit + "
                                    "Metaspace(prepend always); this encoder "
                                    "implements only that combination");
    }
    if ((flags & kFlagByteFallback) != 0) {
        return fail(path.string() + " requires byte fallback, which this "
                                    "encoder does not implement");
    }

    m_entries.resize(vocabCount);
    stream.read(reinterpret_cast<char*>(m_entries.data()),
                static_cast<std::streamsize>(vocabCount * sizeof(Entry)));
    if (!stream) {
        return fail(path.string() + " ends inside its vocabulary table");
    }

    m_blob.resize(stringBytes);
    if (stringBytes != 0) {
        stream.read(m_blob.data(), static_cast<std::streamsize>(stringBytes));
        if (!stream) {
            return fail(path.string() + " ends inside its token text");
        }
    }

    m_index.reserve(vocabCount * 2);
    for (std::size_t i = 0; i < m_entries.size(); ++i) {
        const Entry& entry = m_entries[i];
        if (static_cast<std::size_t>(entry.offset) + entry.length > m_blob.size()) {
            m_entries.clear();
            m_blob.clear();
            m_index.clear();
            return fail(path.string() + " has a token reaching past its text "
                                        "block");
        }
        // First id wins, matching how a vocabulary list is indexed: a duplicate
        // string later in the list is unreachable there too.
        m_index.emplace(std::string_view(m_blob.data() + entry.offset, entry.length),
                        static_cast<int64_t>(i));
    }

    m_unkId = static_cast<int64_t>(unkId);
    m_eosId = static_cast<int64_t>(eosId);
    m_maxTokenBytes = maxTokenBytes;
    m_minScore = minScore;
    return true;
}

const UnigramTokenizer::Entry* UnigramTokenizer::Find(std::string_view piece,
                                                      int64_t* id) const {
    const auto found = m_index.find(piece);
    if (found == m_index.end()) {
        return nullptr;
    }
    if (id != nullptr) {
        *id = found->second;
    }
    return &m_entries[static_cast<std::size_t>(found->second)];
}

std::string_view UnigramTokenizer::Token(int64_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= m_entries.size()) {
        return {};
    }
    const Entry& entry = m_entries[static_cast<std::size_t>(id)];
    return std::string_view(m_blob.data() + entry.offset, entry.length);
}

void UnigramTokenizer::EncodePiece(std::string_view piece,
                                   std::vector<int64_t>& ids) const {
    const std::size_t n = piece.size();
    if (n == 0) {
        return;
    }

    // Viterbi over byte positions: best[i] is the score of the best
    // segmentation of piece[0, i), and `from`/`token` record the edge that
    // achieved it. Unigram's objective is the sum of per-token log
    // probabilities, so the best path is the highest total score.
    constexpr double kUnreachable = -std::numeric_limits<double>::infinity();
    std::vector<double> best(n + 1, kUnreachable);
    std::vector<std::size_t> from(n + 1, 0);
    std::vector<int64_t> token(n + 1, -1);
    best[0] = 0.0;

    const double unknownScore = static_cast<double>(m_minScore) - kUnknownPenalty;

    for (std::size_t i = 0; i < n; ++i) {
        if (best[i] == kUnreachable) {
            continue;
        }
        const std::size_t limit = std::min<std::size_t>(m_maxTokenBytes, n - i);
        bool matched = false;
        for (std::size_t length = 1; length <= limit; ++length) {
            int64_t id = 0;
            const Entry* entry = Find(piece.substr(i, length), &id);
            if (entry == nullptr) {
                continue;
            }
            matched = true;
            const double score = best[i] + static_cast<double>(entry->score);
            if (score > best[i + length]) {
                best[i + length] = score;
                from[i + length] = i;
                token[i + length] = id;
            }
        }
        if (!matched) {
            // Nothing in the vocabulary starts here, so consume one character
            // as unknown. Without this the path would dead-end and the rest of
            // the piece would be dropped silently.
            const std::size_t length = Utf8Length(piece.substr(i));
            const double score = best[i] + unknownScore;
            if (score > best[i + length]) {
                best[i + length] = score;
                from[i + length] = i;
                token[i + length] = m_unkId;
            }
        }
    }

    if (best[n] == kUnreachable) {
        // Only reachable if a matched edge jumped past every unknown fallback,
        // which the loop above cannot produce; treat the whole piece as unknown
        // rather than emitting nothing.
        ids.push_back(m_unkId);
        return;
    }

    const std::size_t start = ids.size();
    for (std::size_t i = n; i > 0; i = from[i]) {
        ids.push_back(token[i]);
    }
    std::reverse(ids.begin() + static_cast<std::ptrdiff_t>(start), ids.end());
}

std::vector<int64_t> UnigramTokenizer::Encode(std::string_view text,
                                              int maxLength) const {
    std::vector<int64_t> ids;
    if (!IsLoaded()) {
        return ids;
    }

    // WhitespaceSplit, then Metaspace with prepend_scheme "always": every
    // whitespace-separated run becomes its own piece carrying a leading
    // metaspace. An empty prompt yields no pieces at all, which is correct --
    // the null prompt used for classifier-free guidance is just the
    // end-of-sequence token.
    std::string piece;
    std::size_t i = 0;
    while (i <= text.size()) {
        const bool atEnd = i == text.size();
        if (atEnd || IsAsciiSpace(text[i])) {
            if (!piece.empty()) {
                std::string withMetaspace;
                withMetaspace.reserve(kMetaspace.size() + piece.size());
                withMetaspace.append(kMetaspace);
                withMetaspace.append(piece);
                EncodePiece(withMetaspace, ids);
                piece.clear();
            }
            if (atEnd) {
                break;
            }
        } else {
            piece.push_back(text[i]);
        }
        ++i;
    }

    // tokenizers reserves room for whatever the post-processor appends before
    // truncating, so the content is cut to maxLength - 1 and the sequence ends
    // up exactly maxLength long.
    if (maxLength > 0) {
        const std::size_t room = static_cast<std::size_t>(maxLength) - 1;
        if (ids.size() > room) {
            ids.resize(room);
        }
    }
    ids.push_back(m_eosId);
    return ids;
}

std::string UnigramTokenizer::Decode(const std::vector<int64_t>& ids) const {
    std::string out;
    for (const int64_t id : ids) {
        if (id == m_eosId) {
            continue;
        }
        const std::string_view piece = Token(id);
        std::size_t offset = 0;
        while (offset < piece.size()) {
            if (piece.compare(offset, kMetaspace.size(), kMetaspace) == 0) {
                out.push_back(' ');
                offset += kMetaspace.size();
            } else {
                out.push_back(piece[offset]);
                ++offset;
            }
        }
    }
    return out;
}

}  // namespace NeuralModelIntegrateTestbed
