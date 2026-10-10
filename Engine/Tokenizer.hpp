#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The SentencePiece Unigram tokenizer FloodDiffusion's umT5 text encoder
// expects, reading the flat table tools/prepare_flooddiffusion.py bakes out of
// tokenizer.json.
//
// Why this is implemented here rather than linked: text_encoder.onnx takes
// token ids, so SOMETHING has to turn a prompt into ids. The alternative was
// baking a fixed set of prompt embeddings offline, which would have meant the
// app could only ever produce motion for prompts chosen at build time. A
// Unigram decode is a Viterbi pass over a vocabulary, which is small enough to
// own outright and removes any build-time limit on what can be asked for.
//
// Deliberately free of D3D12 and ONNX Runtime so it can be tested on its own.
namespace NeuralModelIntegrateTestbed {

// U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's stand-in for a space.
inline constexpr std::string_view kMetaspace = "\xE2\x96\x81";

class UnigramTokenizer {
public:
    // Loads tokenizer.bin. Returns false and fills `error` on a missing file, a
    // bad magic or version, or a pre-tokenizer configuration this class does
    // not implement -- the baked flags exist so that a tokenizer which would be
    // applied WRONGLY is refused rather than silently mis-encoded.
    bool Load(const std::filesystem::path& path, std::string* error);

    bool IsLoaded() const { return !m_entries.empty(); }

    // Prompt -> token ids, with the end-of-sequence id appended, matching the
    // TemplateProcessing post-processor in tokenizer.json.
    //
    // `maxLength` caps the total including that final token, mirroring
    // tokenizers' own behaviour: truncation reserves room for the tokens the
    // post-processor will add, so content is cut to maxLength - 1. Zero or a
    // negative value disables the cap.
    std::vector<int64_t> Encode(std::string_view text, int maxLength) const;

    // Token text for an id, or an empty view when out of range. For diagnostics
    // and for the round-trip check in the tests.
    std::string_view Token(int64_t id) const;

    // Concatenates the tokens and turns the metaspace back into spaces. The
    // inverse of Encode up to the leading space and any unknown character, so
    // the tests can assert a prompt survives the round trip.
    std::string Decode(const std::vector<int64_t>& ids) const;

    std::size_t VocabSize() const { return m_entries.size(); }
    int64_t EndOfSequenceId() const { return m_eosId; }
    int64_t UnknownId() const { return m_unkId; }

private:
    struct Entry {
        uint32_t offset = 0;
        uint32_t length = 0;
        float score = 0.0f;
    };

    // Looks up the vocabulary entry whose text is exactly `piece`.
    const Entry* Find(std::string_view piece, int64_t* id) const;

    // One whitespace-separated piece, metaspace already prepended, appended to
    // `ids` as the highest-scoring segmentation.
    void EncodePiece(std::string_view piece, std::vector<int64_t>& ids) const;

    std::vector<Entry> m_entries;
    std::string m_blob;
    std::unordered_map<std::string_view, int64_t> m_index;

    int64_t m_unkId = 0;
    int64_t m_eosId = 0;
    uint32_t m_maxTokenBytes = 1;
    float m_minScore = 0.0f;
};

}  // namespace NeuralModelIntegrateTestbed
