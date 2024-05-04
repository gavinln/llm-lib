from process_openai_examples import get_remaining_tag, get_remaining_tags


def test_get_remaining_tags():
    default_tags = ["completions", "tiktoken"]
    missing_tag = "embeddings"
    tag_combos = ["".join(default_tags) + missing_tag]
    tag_rems = get_remaining_tags(default_tags, tag_combos)
    assert tag_rems == [missing_tag]


def test_remaining_tag():
    tag_combo = "completionstiktokenembeddings"
    tag_rem1 = get_remaining_tag(tag_combo, "completions")
    tag_rem2 = get_remaining_tag(tag_combo, "tiktoken")
    tag_rem3 = get_remaining_tag(tag_combo, "embeddings")
    assert (tag_rem1, tag_rem2, tag_rem3) == (
        "tiktokenembeddings",
        "completionsembeddings",
        "completionstiktoken",
    )
