from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bumbleclaw_v6.data.groups import (
    candidate_face_edges,
    estimate_pairwise_resources,
    exact_file_hash_groups,
    filename_only_groups,
)


def test_filename_only_groups_are_unique():
    result = filename_only_groups(["a", "b", "c"])
    assert result.method == "filename_only"
    assert result.group_id.tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError):
        filename_only_groups(["a", "a"])


def test_exact_file_hash_groups(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    samples = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "path": [str(a), str(b), str(c)]})
    result = exact_file_hash_groups(samples)
    assert result.group_id[0] == result.group_id[1]
    assert result.group_id[2] != result.group_id[0]
    assert len(result.edges) == 1


def test_candidate_edges_threshold_and_determinism():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    valid = np.array([True, True, True])
    edges = candidate_face_edges(x, valid, threshold=0.9)
    assert len(edges) == 1
    assert edges.iloc[0]["left"] == 0 and edges.iloc[0]["right"] == 1


def test_resource_estimate_positive():
    estimate = estimate_pairwise_resources(1000)
    assert estimate["pairs"] == 499_500
    assert estimate["chunk_matrix_bytes"] > 0
