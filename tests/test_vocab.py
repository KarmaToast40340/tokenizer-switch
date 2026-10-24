import torch

from toksw.vocab import _id_to_token, build_union, fvt_pieces, init_embeddings, prune_to_new


def test_union_keeps_old_ids_and_shares_rows(old_tok, new_tok):
    u = build_union(old_tok, new_tok)
    old_tokens, new_tokens = _id_to_token(old_tok), _id_to_token(new_tok)
    assert u.tokens[: len(old_tokens)] == old_tokens
    assert [u.tokens[i] for i in u.new2union] == new_tokens
    shared = set(old_tokens) & set(new_tokens)
    assert u.size == len(old_tokens) + len(new_tokens) - len(shared)


def test_fvt_pieces_spell_the_token(old_tok, new_tok):
    old_tokens = _id_to_token(old_tok)
    for s in _id_to_token(new_tok):
        pieces = fvt_pieces(old_tok, s)
        assert "".join(old_tokens[i] for i in pieces) == s


def test_init_and_prune(old_tok, new_tok):
    torch.manual_seed(0)
    old_emb = torch.randn(old_tok.get_vocab_size(), 8)
    u = build_union(old_tok, new_tok)
    emb = init_embeddings(old_emb, old_tok, u.tokens, "fvt")
    assert torch.equal(emb[: u.n_old], old_emb)
    # a new-only token is the mean of its pieces
    j = u.n_old
    assert torch.allclose(emb[j], old_emb[fvt_pieces(old_tok, u.tokens[j])].mean(0))
    pruned = prune_to_new(emb, u)
    assert pruned.shape == (new_tok.get_vocab_size(), 8)
    assert torch.equal(pruned, emb[torch.as_tensor(u.new2union)])
