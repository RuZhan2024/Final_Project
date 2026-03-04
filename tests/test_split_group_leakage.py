from __future__ import annotations

from fall_detection.data.splits.make_splits import group_id_for, split_groups_to_match_targets


def test_caucafall_subject_group_extraction() -> None:
    stem = "Subject.07__Action__clip_001"
    gid = group_id_for(stem, mode="caucafall_subject", regex=None, group_map=None)
    assert gid.lower().startswith("subject")
    assert "07" in gid


def test_group_split_has_no_overlap_across_splits() -> None:
    group_sizes = {
        "Subject01": 10,
        "Subject02": 12,
        "Subject03": 8,
        "Subject04": 9,
        "Subject05": 11,
        "Subject06": 7,
    }
    gids = list(group_sizes.keys())
    # deterministic random wrapper from stdlib Random, as expected by the splitter.
    import random

    rng = random.Random(33724876)
    tr, va, te = split_groups_to_match_targets(
        group_ids=gids,
        group_sizes=group_sizes,
        train=0.8,
        val=0.1,
        test=0.1,
        rng=rng,
        balance_by="stems",
    )
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    assert s_tr | s_va | s_te == set(gids)

