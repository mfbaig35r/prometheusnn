"""LLM-friendly prediction summary builder. Pure functions, no state."""

from __future__ import annotations

from os.path import commonprefix

MAX_CODES = 5


def build_prediction_summary(
    predictions: list[dict],
    forked_completions: list[dict] | None,
    routing: dict,
    path_score: float,
    level_names: list[str],
    code_mapping: dict | None = None,
) -> dict:
    """Build an LLM-consumable summary from prediction results.

    Args:
        predictions: Per-level prediction dicts (depth, level, predicted, confidence, margin, top_candidates).
        forked_completions: Forked completion dicts (suffix, per_level_probs, score) or None.
        routing: Routing dict (decision, confident_depth, confident_path, review_reason).
        path_score: Log-probability score of the best path.
        level_names: Ordered level names from taxonomy.
        code_mapping: Loaded code mapping dict with "mapping" key, or None.

    Returns:
        Summary dict with candidates, classification_notes, verdict, routing.
    """
    candidates = _build_candidates(
        predictions,
        forked_completions,
        path_score,
        level_names,
        code_mapping,
    )
    notes = _build_classification_notes(predictions, candidates, forked_completions, code_mapping)
    verdict = _build_verdict(predictions, candidates, routing, code_mapping)

    return {
        "candidates": candidates,
        "classification_notes": notes,
        "verdict": verdict,
        "routing": routing["decision"],
    }


def lookup_codes(path: list[str], code_mapping: dict | None) -> list[str] | None:
    """Look up codes for a full taxonomy path.

    Args:
        path: List of labels forming the full path.
        code_mapping: Code mapping dict with "mapping" key, or None.

    Returns:
        Sorted deduplicated list of code strings, or None if no mapping or no match.
    """
    if code_mapping is None:
        return None
    mapping = code_mapping.get("mapping", {})
    key = "|".join(path)
    codes = mapping.get(key)
    if codes is None:
        return None
    return codes


def _code_prefix_for_partial_path(partial_path: list[str], code_mapping: dict | None) -> str | None:
    """Derive the longest common code prefix for a partial taxonomy path.

    Scans all entries in the mapping whose keys start with the given partial path
    and returns the longest common prefix of all their codes.
    """
    if code_mapping is None:
        return None
    mapping = code_mapping.get("mapping", {})
    prefix_key = "|".join(partial_path)

    all_codes = []
    for key, codes in mapping.items():
        if key == prefix_key or key.startswith(prefix_key + "|"):
            all_codes.extend(codes)

    if not all_codes:
        return None

    prefix = commonprefix(all_codes)
    return prefix if prefix else None


def _label_with_code(label: str, partial_path: list[str], code_mapping: dict | None) -> str:
    """Format a label with its code prefix: '40: Rubber' or just 'Rubber'."""
    if code_mapping is None:
        return label
    prefix = _code_prefix_for_partial_path(partial_path, code_mapping)
    if prefix:
        return f"{prefix}: {label}"
    return label


def _root_code_label(candidate: dict) -> str:
    """Format candidate's root label with its code_prefix if available."""
    path = candidate.get("path", [])
    if not path:
        return "?"
    root = path[0]
    label = root["label"]
    prefix = root.get("code_prefix")
    if prefix:
        return f"{prefix}: {label}"
    return label


def _build_structured_path(
    labels: list[str],
    level_names: list[str],
    confidence_per_level: dict[str, float],
    code_mapping: dict | None,
) -> list[dict]:
    """Build a structured path: list of dicts with level, label, confidence, and code info."""
    path_entries = []
    is_leaf = len(labels) - 1

    for i, (label, name) in enumerate(zip(labels, level_names)):
        entry: dict = {
            "level": name,
            "label": label,
            "confidence": confidence_per_level.get(name, 0.0),
        }

        if code_mapping is not None:
            if i == is_leaf:
                # Leaf level: attach full codes (capped)
                codes = lookup_codes(labels, code_mapping)
                if codes is not None:
                    if len(codes) <= MAX_CODES:
                        entry["codes"] = codes
                    else:
                        entry["codes"] = codes[:MAX_CODES]
                        entry["total_codes"] = len(codes)
            else:
                # Non-leaf: derive code prefix
                partial = labels[: i + 1]
                prefix = _code_prefix_for_partial_path(partial, code_mapping)
                if prefix:
                    entry["code_prefix"] = prefix

        path_entries.append(entry)

    return path_entries


def _build_candidates(
    predictions: list[dict],
    forked_completions: list[dict] | None,
    path_score: float,
    level_names: list[str],
    code_mapping: dict | None,
) -> list[dict]:
    """Build ranked candidate list from forked completions (or single best path)."""
    confident_path = []

    if forked_completions and len(forked_completions) > 0:
        best_path = [p["predicted"] for p in predictions]
        first_suffix = forked_completions[0]["suffix"]
        suffix_len = len(first_suffix)
        prefix_len = len(best_path) - suffix_len
        confident_path = best_path[:prefix_len]
    else:
        # No forks — single candidate (the best path)
        best_path = [p["predicted"] for p in predictions]
        confidence_per_level = {p["level"]: round(p["confidence"], 4) for p in predictions}
        structured_path = _build_structured_path(
            best_path, level_names, confidence_per_level, code_mapping
        )
        candidate = {
            "rank": 1,
            "path": structured_path,
            "path_score": round(path_score, 4),
            "min_confidence": round(min(e["confidence"] for e in structured_path), 4),
        }
        return [candidate]

    # Build candidates from forked completions
    candidates = []
    for i, fc in enumerate(forked_completions):
        if i >= 5:
            break
        full_path = confident_path + fc["suffix"]
        suffix_probs = fc["per_level_probs"]

        # Build per-level confidence: prefix from predictions, suffix from fc
        confidence_per_level = {}
        for j, name in enumerate(level_names):
            if j < len(confident_path):
                confidence_per_level[name] = round(predictions[j]["confidence"], 4)
            else:
                suffix_idx = j - len(confident_path)
                if suffix_idx < len(suffix_probs):
                    confidence_per_level[name] = round(suffix_probs[suffix_idx], 4)

        structured_path = _build_structured_path(
            full_path, level_names, confidence_per_level, code_mapping
        )
        candidate = {
            "rank": i + 1,
            "path": structured_path,
            "path_score": round(fc["score"], 4),
            "min_confidence": round(min(e["confidence"] for e in structured_path), 4),
        }
        candidates.append(candidate)

    return candidates


def _build_classification_notes(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> list[str]:
    """Generate classification notes from pattern detectors."""
    notes = []
    for detector in [
        _note_root_uncertainty,
        _note_heading_strength,
        _note_minority_leaf_confidence,
        _note_convergent_leaves,
        _note_confidence_gap,
        _note_bottleneck,
    ]:
        note = detector(predictions, candidates, forked_completions, code_mapping=code_mapping)
        if note:
            notes.append(note)
    if not notes:
        fallback = _fallback_confidence_summary(candidates)
        if fallback:
            notes.append(fallback)
    return notes


def _build_verdict(
    predictions: list[dict],
    candidates: list[dict],
    routing: dict,
    code_mapping: dict | None = None,
) -> str:
    """Synthesize observations into a 1-sentence recommendation."""
    decision = routing["decision"]

    if decision == "OUT_OF_TAXONOMY":
        if len(candidates) == 1:
            return "Item appears out-of-taxonomy; single path shown but treat with low confidence."
        root_conf = _root_confidence(candidates[0])
        best_label = _root_code_label(candidates[0])
        runner_label = _root_code_label(candidates[1])
        runner_root = _root_confidence(candidates[1])
        if root_conf is not None and root_conf >= 0.70:
            return (
                f"Item flagged out-of-taxonomy but candidate 1 ({best_label}) has strong root confidence "
                f"({root_conf:.0%}); review candidate 1 first."
            )
        if root_conf is not None and root_conf >= 0.50:
            return (
                f"Item flagged out-of-taxonomy; candidate 1 ({best_label}) has moderate root confidence "
                f"({root_conf:.0%}) — verify manually."
            )
        # root_conf < 50%
        if root_conf is not None and runner_root is not None:
            gap_pp = root_conf - runner_root
            if gap_pp < 0.15:
                return (
                    f"Item appears out-of-taxonomy; split between candidate 1 ({best_label}) "
                    f"and candidate 2 ({runner_label}) — no clear winner."
                )
        return "Item appears out-of-taxonomy; no candidate is clearly reliable."

    if decision == "AUTO_ACCEPT":
        if len(candidates) == 1:
            return "High confidence across all levels; candidate 1 is reliable."
        return "High confidence across all levels; candidate 1 is reliable."

    # HUMAN_REVIEW — this is where the verdict matters most
    if len(candidates) == 1:
        leaf_conf = _leaf_confidence(candidates[0])
        if leaf_conf is not None and leaf_conf < 0.50:
            return "Single candidate with low leaf confidence; manual review recommended."
        return "Single candidate needs review at uncertain levels."

    # Multiple candidates — compare them
    best = candidates[0]
    runner_up = candidates[1]
    best_score = best["path_score"]
    runner_score = runner_up["path_score"]
    gap = best_score - runner_score

    best_root = _root_confidence(best)
    runner_root = _root_confidence(runner_up)
    best_leaf = _leaf_confidence(best)
    runner_leaf = _leaf_confidence(runner_up)
    best_label = _root_code_label(best)
    runner_label = _root_code_label(runner_up)

    # Do they share the same root level?
    best_labels = _path_labels(best)
    runner_labels = _path_labels(runner_up)
    same_root = best_labels[0] == runner_labels[0] if best_labels and runner_labels else False

    if gap > 2.0 and best_root is not None and best_root >= 0.70:
        return (
            f"Candidate 1 ({best_label}) is the clear best ({best_root:.0%} root confidence, "
            f"{gap:.1f} log-prob gap); alternatives are unlikely."
        )

    if same_root:
        fork_level = _first_divergent_level(best, runner_up)
        if runner_leaf is not None and best_leaf is not None and runner_leaf > best_leaf + 0.10:
            return (
                f"Candidates share root but diverge at {fork_level}; "
                f"candidate {runner_up['rank']} has higher leaf confidence "
                f"({runner_leaf:.0%} vs {best_leaf:.0%}) — compare both."
            )
        return (
            f"Candidates share root but diverge at {fork_level}; "
            f"candidate 1 leads by {gap:.1f} log-prob."
        )

    # Different roots
    if best_root is not None and runner_root is not None:
        if best_root >= 0.70 and runner_root < 0.20:
            return (
                f"Candidate 1 ({best_label}) is strongly favored ({best_root:.0%} vs {runner_root:.0%} "
                f"root confidence); alternatives are in unrelated categories."
            )
        if best_root - runner_root < 0.20:
            return (
                f"Root-level split: candidate 1 ({best_label}) and candidate 2 ({runner_label}) are in different categories "
                f"with similar confidence ({best_root:.0%} vs {runner_root:.0%}); "
                f"review both paths."
            )

    return f"Candidate 1 leads by {gap:.1f} log-prob but review is recommended."


def _note_root_uncertainty(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect when top root-level labels are within 20pp of each other."""
    if not predictions:
        return None
    root = predictions[0]
    top_cands = root.get("top_candidates", [])
    if len(top_cands) < 2:
        return None

    top_conf = top_cands[0]["confidence"]
    close_count = sum(1 for c in top_cands[1:3] if top_conf - c["confidence"] <= 0.20)
    if close_count > 0:
        level_name = root["level"]
        labels = [
            _label_with_code(c["label"], [c["label"]], code_mapping)
            for c in top_cands[: close_count + 1]
        ]
        return (
            f"{level_name}-level uncertainty: top {len(labels)} labels within 20pp "
            f"({', '.join(labels)})"
        )
    return None


def _note_heading_strength(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect when mid-level (depth 1) confidence is strong (>90%) across fork branches."""
    if not forked_completions or len(forked_completions) < 2:
        return None
    if len(predictions) < 2:
        return None

    best_path = [p["predicted"] for p in predictions]
    first_suffix_len = len(forked_completions[0]["suffix"])
    prefix_len = len(best_path) - first_suffix_len

    if prefix_len >= 2:
        d1_conf = predictions[1]["confidence"]
        if d1_conf >= 0.90:
            return f"{predictions[1]['level']} confidence is strong ({d1_conf:.0%}) in confident prefix"
        return None

    depth_1_suffix_idx = 1 - prefix_len
    if depth_1_suffix_idx < 0:
        return None

    all_strong = all(
        len(fc["per_level_probs"]) > depth_1_suffix_idx
        and fc["per_level_probs"][depth_1_suffix_idx] >= 0.90
        for fc in forked_completions
    )
    if all_strong:
        level_name = predictions[1]["level"] if len(predictions) > 1 else "Level-1"
        return f"{level_name} confidence is strong (>90%) across all fork branches"
    return None


def _note_minority_leaf_confidence(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect when a non-best candidate has >10pp higher leaf confidence."""
    if len(candidates) < 2:
        return None

    best_leaf_conf = _leaf_confidence(candidates[0])
    if best_leaf_conf is None:
        return None

    for c in candidates[1:]:
        c_leaf = _leaf_confidence(c)
        if c_leaf is not None and c_leaf > best_leaf_conf + 0.10:
            leaf_entry = c["path"][-1]
            leaf_code = leaf_entry.get("code_prefix") or ""
            if leaf_entry.get("codes"):
                leaf_code = ", ".join(leaf_entry["codes"][:3])
            suffix = f" [{leaf_code}]" if leaf_code else ""
            return (
                f"Candidate {c['rank']} has higher leaf confidence "
                f"({c_leaf:.0%}) than best path ({best_leaf_conf:.0%}){suffix}"
            )
    return None


def _note_convergent_leaves(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect when multiple candidates converge on the same leaf label."""
    if len(candidates) < 2:
        return None

    leaf_to_ranks: dict[str, list[int]] = {}
    leaf_to_codes: dict[str, str] = {}
    for c in candidates:
        labels = _path_labels(c)
        leaf = labels[-1] if labels else None
        if leaf:
            leaf_to_ranks.setdefault(leaf, []).append(c["rank"])
            # Grab leaf codes from first candidate that has them
            if leaf not in leaf_to_codes:
                leaf_entry = c["path"][-1]
                if leaf_entry.get("codes"):
                    leaf_to_codes[leaf] = ", ".join(leaf_entry["codes"][:3])

    for leaf, ranks in leaf_to_ranks.items():
        if len(ranks) >= 2:
            rank_str = " and ".join(str(r) for r in ranks)
            code_suffix = f" [{leaf_to_codes[leaf]}]" if leaf in leaf_to_codes else ""
            return f'Candidates {rank_str} converge on same leaf "{leaf}"{code_suffix}'
    return None


def _note_confidence_gap(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect dominant (>2.0 log gap) vs competitive (<0.5 log gap) candidate spreads."""
    if len(candidates) < 2:
        return None

    best_score = candidates[0]["path_score"]
    second_score = candidates[1]["path_score"]
    gap = best_score - second_score

    if gap > 2.0:
        return "Best candidate is dominant (>2.0 log-prob gap over runner-up)"
    if gap < 0.5:
        return "Top candidates are competitive (<0.5 log-prob gap)"
    return None


def _note_bottleneck(
    predictions: list[dict],
    candidates: list[dict],
    forked_completions: list[dict] | None,
    code_mapping: dict | None = None,
) -> str | None:
    """Detect when one level is <50% while a neighbor is >70%."""
    if not candidates:
        return None
    path = candidates[0].get("path", [])
    if len(path) < 2:
        return None

    confs = [e["confidence"] for e in path]
    for i, conf in enumerate(confs):
        if conf < 0.50:
            has_strong_neighbor = False
            if i > 0 and confs[i - 1] > 0.70:
                has_strong_neighbor = True
            if i < len(confs) - 1 and confs[i + 1] > 0.70:
                has_strong_neighbor = True
            if has_strong_neighbor:
                level_name = path[i]["level"]
                return (
                    f"Confidence bottleneck at {level_name} ({conf:.0%}); other levels are strong."
                )
    return None


def _fallback_confidence_summary(candidates: list[dict]) -> str | None:
    """Produce a fallback note when no detectors fire."""
    if not candidates:
        return None
    path = candidates[0].get("path", [])
    if not path:
        return None

    confs = [e["confidence"] for e in path]
    if all(c >= 0.70 for c in confs):
        return "All levels confident (>70%)."

    # Check for monotonically decreasing
    if all(confs[i] >= confs[i + 1] for i in range(len(confs) - 1)) and confs[0] > confs[-1]:
        return f"Confidence decreases from root ({confs[0]:.0%}) to leaf ({confs[-1]:.0%})."

    # Check for monotonically increasing
    if all(confs[i] <= confs[i + 1] for i in range(len(confs) - 1)) and confs[-1] > confs[0]:
        return f"Confidence increases from root ({confs[0]:.0%}) to leaf ({confs[-1]:.0%})."

    avg = sum(confs) / len(confs)
    return f"Average confidence across levels: {avg:.0%}."


# ─── Path helpers (structured path: list[dict]) ───


def _path_labels(candidate: dict) -> list[str]:
    """Extract labels from a structured path."""
    return [entry["label"] for entry in candidate.get("path", [])]


def _leaf_confidence(candidate: dict) -> float | None:
    """Extract the leaf-level confidence from a candidate's structured path."""
    path = candidate.get("path", [])
    if not path:
        return None
    return path[-1].get("confidence")


def _root_confidence(candidate: dict) -> float | None:
    """Extract the root-level confidence from a candidate's structured path."""
    path = candidate.get("path", [])
    if not path:
        return None
    return path[0].get("confidence")


def _first_divergent_level(c1: dict, c2: dict) -> str:
    """Return the name of the first level where two candidates diverge."""
    path1 = c1.get("path", [])
    path2 = c2.get("path", [])
    for e1, e2 in zip(path1, path2):
        if e1["label"] != e2["label"]:
            return e1["level"]
    return path1[-1]["level"] if path1 else "leaf"
