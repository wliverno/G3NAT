from g3nat.data.splits import grouped_split

def test_no_sequence_shared_between_train_and_val():
    # 6 sequences, each duplicated 4x (mimics the 4 contact-variants per sequence)
    seqs = []
    for s in ['AAAA', 'CCCC', 'GGGG', 'TTTT', 'ACGT', 'TGCA']:
        seqs += [s] * 4
    train_idx, val_idx = grouped_split(seqs, test_size=0.34, seed=42)
    train_seqs = {seqs[i] for i in train_idx}
    val_seqs = {seqs[i] for i in val_idx}
    assert train_seqs.isdisjoint(val_seqs)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(seqs)

def test_deterministic_given_seed():
    seqs = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    assert grouped_split(seqs, seed=1) == grouped_split(seqs, seed=1)
