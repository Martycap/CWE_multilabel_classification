"""Progressive bottom-up aggregation of rare CWE labels.
--------
1. Remove the pseudo-labels ``NVD-CWE-noinfo`` and ``NVD-CWE-Other``.
2. Process active CWE labels from the deepest level to the highest.
3. Keep a label when its current support is at least ``MIN_SUPPORT``.
4. Otherwise merge it progressively into:
   - its primary parent;
   - another direct parent;
   - a grandparent;
   - a more distant ancestor, when necessary.
5. Recompute supports after every merge using sets of CVE rows, so the same
   CVE is never counted twice when it already contains both parent and child.
6. Allow every non-Prohibited CWE to become a final class, including
   Discouraged, Class, and Pillar nodes, provided it reaches the threshold.
7. Remove final rare CWE labels that have no eligible ancestor.
8. Remove a CVE only when no CWE label remains after the previous steps.

The merge mapping remains progressive. For example:

    CWE-A -> CWE-B
    CWE-B -> CWE-C

The audit file preserves both steps instead of flattening them to CWE-A -> CWE-C.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CVE_DATASET_FILE = Path("data/cve_dataset.csv")
DEFAULT_CWE_HIERARCHY_FILE = Path("data/cwe_hierarchy_with_counts.csv")

DEFAULT_MERGED_CVE_FILE = Path("data/cve_dataset_hierarchy_merged.csv")
DEFAULT_MERGED_CWE_FILE = Path("data/cwe_hierarchy_merged_counts.csv")
DEFAULT_MAPPING_FILE = Path("data/cwe_merge_mapping.csv")
DEFAULT_REMOVED_CVE_FILE = Path("data/cve_removed_without_cwe.csv")
DEFAULT_REMOVED_LABELS_FILE = Path("data/cwe_removed_rare_labels.csv")

DEFAULT_MIN_SUPPORT = 100
PROHIBITED_MAPPING_USAGE = "prohibited"

PSEUDO_CWE_LABELS = {
    "NVD-CWE-noinfo",
    "NVD-CWE-Other",
}


@dataclass(frozen=True)
class CweNode:
    """Relevant information for one CWE hierarchy node."""

    cwe_id: str
    name: str
    abstraction: str
    status: str
    mapping_usage: str
    primary_parent: str | None
    other_parents: tuple[str, ...]
    children: tuple[str, ...]
    depth: int

    @property
    def can_be_final_class(self) -> bool:
        """Return whether the CWE is allowed as a final prediction class."""

        return self.mapping_usage.strip().lower() != PROHIBITED_MAPPING_USAGE


@dataclass(frozen=True)
class MergeStep:
    """One progressive merge performed by the algorithm."""

    step: int
    source_cwe: str
    target_cwe: str
    source_depth: int
    target_depth: int
    hierarchy_distance: int
    parent_selection: str
    source_count_before: int
    target_count_before: int
    target_count_after: int


@dataclass(frozen=True)
class RemovedRareLabel:
    """One final rare label removed because it could not be aggregated."""

    cwe_id: str
    cve_count: int
    reason: str


def parse_json_list(raw_value: str | None) -> list[str]:
    """Parse a JSON list stored in a CSV field."""

    if raw_value is None or not raw_value.strip():
        return []

    value = json.loads(raw_value)

    if not isinstance(value, list):
        raise ValueError(
            f"Expected a JSON list, received: {type(value).__name__}"
        )

    return [str(item).strip() for item in value if str(item).strip()]


def normalize_cwe_labels(labels: Iterable[str]) -> set[str]:
    """
    Normalize labels and split malformed comma-separated values.

    Example:
        "CWE-465,CWE-485" -> {"CWE-465", "CWE-485"}
    """

    normalized: set[str] = set()

    for raw_label in labels:
        for value in raw_label.split(","):
            label = value.strip()

            if label and label not in PSEUDO_CWE_LABELS:
                normalized.add(label)

    return normalized


def cwe_sort_key(cwe_id: str) -> tuple[int, str]:
    """Sort standard CWE identifiers numerically when possible."""

    try:
        return int(cwe_id.split("-", maxsplit=1)[1]), cwe_id
    except (IndexError, ValueError):
        return 10**9, cwe_id


def load_cwe_hierarchy(path: Path) -> dict[str, CweNode]:
    """Load the complete CWE hierarchy produced by the updated parser."""

    if not path.exists():
        raise FileNotFoundError(f"CWE hierarchy file not found: {path}")

    nodes: dict[str, CweNode] = {}

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "cwe_id",
            "name",
            "abstraction",
            "status",
            "mapping_usage",
            "primary_parent",
            "other_parents",
            "children",
            "depth",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                "The CWE hierarchy CSV is missing columns: "
                f"{sorted(missing_columns)}. "
                "Run the updated CWE parser first."
            )

        for row in reader:
            cwe_id = row["cwe_id"].strip()

            if not cwe_id:
                continue

            primary_parent = row["primary_parent"].strip() or None

            nodes[cwe_id] = CweNode(
                cwe_id=cwe_id,
                name=row["name"].strip(),
                abstraction=row["abstraction"].strip(),
                status=row["status"].strip(),
                mapping_usage=row["mapping_usage"].strip(),
                primary_parent=primary_parent,
                other_parents=tuple(parse_json_list(row["other_parents"])),
                children=tuple(parse_json_list(row["children"])),
                depth=int(row["depth"]),
            )

    if not nodes:
        raise ValueError(f"No CWE nodes were loaded from {path}")

    return nodes


def load_cve_dataset(
    path: Path,
) -> tuple[list[dict[str, str]], list[set[str]], list[str], int]:
    """
    Load CVE rows and clean their multilabel ``cwe_ids`` field.

    Returns the number of removed pseudo-label occurrences as an additional
    statistic.
    """

    if not path.exists():
        raise FileNotFoundError(f"CVE dataset file not found: {path}")

    rows: list[dict[str, str]] = []
    labels_by_row: list[set[str]] = []
    removed_pseudo_label_occurrences = 0

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"The CVE dataset has no header: {path}")

        if "cwe_ids" not in reader.fieldnames:
            raise ValueError(
                f"The CVE dataset {path} does not contain 'cwe_ids'."
            )

        fieldnames = list(reader.fieldnames)

        for row in reader:
            raw_labels = parse_json_list(row["cwe_ids"])

            removed_pseudo_label_occurrences += sum(
                label in PSEUDO_CWE_LABELS
                for label in raw_labels
            )

            labels = normalize_cwe_labels(raw_labels)

            rows.append(dict(row))
            labels_by_row.append(labels)

    return (
        rows,
        labels_by_row,
        fieldnames,
        removed_pseudo_label_occurrences,
    )


def build_label_memberships(
    labels_by_row: list[set[str]],
) -> dict[str, set[int]]:
    """Build ``CWE -> set of CVE row indexes`` memberships."""

    label_to_rows: dict[str, set[int]] = defaultdict(set)

    for row_index, labels in enumerate(labels_by_row):
        for cwe_id in labels:
            label_to_rows[cwe_id].add(row_index)

    return dict(label_to_rows)


def ordered_direct_parents(
    node: CweNode,
) -> list[tuple[str, str]]:
    """Return direct parents in the required preference order."""

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    if node.primary_parent:
        ordered.append((node.primary_parent, "primary_parent"))
        seen.add(node.primary_parent)

    for parent_id in sorted(node.other_parents, key=cwe_sort_key):
        if parent_id not in seen:
            ordered.append((parent_id, "other_direct_parent"))
            seen.add(parent_id)

    return ordered


def find_merge_target(
    source_cwe: str,
    nodes: dict[str, CweNode],
) -> tuple[str, int, str] | None:
    """
    Find the preferred non-Prohibited ancestor.

    Search order:
    1. primary parent;
    2. another direct parent;
    3. grandparent;
    4. increasingly distant ancestors.
    """

    source_node = nodes.get(source_cwe)

    if source_node is None:
        return None

    queue: deque[tuple[str, int, str]] = deque()
    visited = {source_cwe}

    for parent_id, reason in ordered_direct_parents(source_node):
        queue.append((parent_id, 1, reason))

    while queue:
        candidate_id, distance, first_relation = queue.popleft()

        if candidate_id in visited:
            continue

        visited.add(candidate_id)
        candidate = nodes.get(candidate_id)

        if candidate is None:
            continue

        if candidate.can_be_final_class:
            if distance == 1:
                selection = first_relation
            elif distance == 2:
                selection = "grandparent"
            else:
                selection = f"ancestor_level_{distance}"

            return candidate_id, distance, selection

        # Prohibited nodes may be traversed but cannot be final classes.
        for parent_id, _ in ordered_direct_parents(candidate):
            if parent_id not in visited:
                queue.append(
                    (parent_id, distance + 1, first_relation)
                )

    return None


def active_labels_sorted_bottom_up(
    label_to_rows: dict[str, set[int]],
    nodes: dict[str, CweNode],
) -> list[str]:
    """Return active labels ordered by decreasing hierarchy depth."""

    return sorted(
        (
            cwe_id
            for cwe_id, row_indexes in label_to_rows.items()
            if row_indexes
        ),
        key=lambda cwe_id: (
            -nodes[cwe_id].depth if cwe_id in nodes else 1,
            cwe_sort_key(cwe_id),
        ),
    )


def merge_labels_bottom_up(
    labels_by_row: list[set[str]],
    label_to_rows: dict[str, set[int]],
    nodes: dict[str, CweNode],
    min_support: int,
) -> tuple[list[MergeStep], set[str]]:
    """
    Progressively merge rare labels into preferred ancestors.

    The function mutates ``labels_by_row`` and ``label_to_rows`` in place.
    """

    if min_support < 1:
        raise ValueError("min_support must be at least 1.")

    merge_steps: list[MergeStep] = []
    unmerged_rare_labels: set[str] = set()
    step_number = 0

    while True:
        merged_in_pass = False

        for source_cwe in active_labels_sorted_bottom_up(
            label_to_rows,
            nodes,
        ):
            source_rows = label_to_rows.get(source_cwe, set())
            source_count = len(source_rows)

            if source_count == 0 or source_count >= min_support:
                continue

            target_info = find_merge_target(source_cwe, nodes)

            if target_info is None:
                unmerged_rare_labels.add(source_cwe)
                continue

            target_cwe, distance, selection = target_info

            if target_cwe == source_cwe:
                unmerged_rare_labels.add(source_cwe)
                continue

            target_rows = label_to_rows.setdefault(target_cwe, set())

            source_count_before = len(source_rows)
            target_count_before = len(target_rows)

            target_rows.update(source_rows)

            for row_index in source_rows:
                labels_by_row[row_index].discard(source_cwe)
                labels_by_row[row_index].add(target_cwe)

            label_to_rows[source_cwe] = set()

            source_node = nodes.get(source_cwe)
            target_node = nodes.get(target_cwe)

            step_number += 1
            merge_steps.append(
                MergeStep(
                    step=step_number,
                    source_cwe=source_cwe,
                    target_cwe=target_cwe,
                    source_depth=source_node.depth if source_node else -1,
                    target_depth=target_node.depth if target_node else -1,
                    hierarchy_distance=distance,
                    parent_selection=selection,
                    source_count_before=source_count_before,
                    target_count_before=target_count_before,
                    target_count_after=len(target_rows),
                )
            )

            unmerged_rare_labels.discard(source_cwe)
            merged_in_pass = True

        if not merged_in_pass:
            break

    final_rare_labels = {
        cwe_id
        for cwe_id, row_indexes in label_to_rows.items()
        if 0 < len(row_indexes) < min_support
    }

    unmerged_rare_labels.update(final_rare_labels)

    return merge_steps, unmerged_rare_labels


def remove_unmergeable_rare_labels(
    labels_by_row: list[set[str]],
    label_to_rows: dict[str, set[int]],
    rare_labels: set[str],
    nodes: dict[str, CweNode],
) -> list[RemovedRareLabel]:
    """
    Remove rare labels that remain after bottom-up aggregation.

    A label is removed only after all possible merge attempts have failed.
    The corresponding CVE row is kept when at least one other CWE remains.
    """

    removed_labels: list[RemovedRareLabel] = []

    for cwe_id in sorted(rare_labels, key=cwe_sort_key):
        row_indexes = set(label_to_rows.get(cwe_id, set()))

        if not row_indexes:
            continue

        if cwe_id not in nodes:
            reason = "absent_from_hierarchy"
        elif find_merge_target(cwe_id, nodes) is None:
            reason = "no_eligible_ancestor"
        else:
            reason = "below_threshold_after_progressive_merge"

        removed_labels.append(
            RemovedRareLabel(
                cwe_id=cwe_id,
                cve_count=len(row_indexes),
                reason=reason,
            )
        )

        for row_index in row_indexes:
            labels_by_row[row_index].discard(cwe_id)

        label_to_rows[cwe_id] = set()

    return removed_labels


def filter_rows_without_labels(
    rows: list[dict[str, str]],
    labels_by_row: list[set[str]],
) -> tuple[
    list[dict[str, str]],
    list[set[str]],
    list[dict[str, str]],
]:
    """Remove only CVE rows that have no labels left."""

    kept_rows: list[dict[str, str]] = []
    kept_labels: list[set[str]] = []
    removed_rows: list[dict[str, str]] = []

    for row, labels in zip(rows, labels_by_row, strict=True):
        if labels:
            kept_rows.append(row)
            kept_labels.append(labels)
        else:
            removed_rows.append(row)

    return kept_rows, kept_labels, removed_rows


def write_cve_dataset(
    path: Path,
    rows: list[dict[str, str]],
    labels_by_row: list[set[str]],
    fieldnames: list[str],
) -> None:
    """Write a CVE dataset with normalized CWE labels."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row, labels in zip(rows, labels_by_row, strict=True):
            output_row = dict(row)
            output_row["cwe_ids"] = json.dumps(
                sorted(labels, key=cwe_sort_key)
            )
            writer.writerow(output_row)


def write_removed_cves(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write CVEs removed because no valid CWE label remained."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_merge_mapping(
    path: Path,
    merge_steps: Iterable[MergeStep],
) -> None:
    """Write the progressive merge audit file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "source_cwe",
        "target_cwe",
        "source_depth",
        "target_depth",
        "hierarchy_distance",
        "parent_selection",
        "source_count_before",
        "target_count_before",
        "target_count_after",
    ]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for merge_step in merge_steps:
            writer.writerow(
                {
                    field_name: getattr(merge_step, field_name)
                    for field_name in fieldnames
                }
            )


def write_removed_labels(
    path: Path,
    removed_labels: Iterable[RemovedRareLabel],
) -> None:
    """Write the audit file for rare labels removed after merging."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["cwe_id", "cve_count", "reason"]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in removed_labels:
            writer.writerow(
                {
                    "cwe_id": item.cwe_id,
                    "cve_count": item.cve_count,
                    "reason": item.reason,
                }
            )


def write_final_cwe_counts(
    path: Path,
    nodes: dict[str, CweNode],
    label_to_rows: dict[str, set[int]],
    min_support: int,
) -> None:
    """Write the surviving CWE labels and their recomputed supports."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cwe_id",
        "name",
        "abstraction",
        "status",
        "mapping_usage",
        "primary_parent",
        "other_parents",
        "children",
        "depth",
        "cve_count",
        "meets_min_support",
    ]

    active_cwes = [
        cwe_id
        for cwe_id, row_indexes in label_to_rows.items()
        if row_indexes
    ]

    active_cwes.sort(
        key=lambda cwe_id: (
            -len(label_to_rows[cwe_id]),
            cwe_sort_key(cwe_id),
        )
    )

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cwe_id in active_cwes:
            node = nodes.get(cwe_id)
            count = len(label_to_rows[cwe_id])

            writer.writerow(
                {
                    "cwe_id": cwe_id,
                    "name": node.name if node else "",
                    "abstraction": node.abstraction if node else "",
                    "status": node.status if node else "",
                    "mapping_usage": (
                        node.mapping_usage if node else ""
                    ),
                    "primary_parent": (
                        node.primary_parent if node else ""
                    ),
                    "other_parents": json.dumps(
                        list(node.other_parents) if node else []
                    ),
                    "children": json.dumps(
                        list(node.children) if node else []
                    ),
                    "depth": node.depth if node else "",
                    "cve_count": count,
                    "meets_min_support": count >= min_support,
                }
            )


def print_summary(
    original_cve_count: int,
    final_cve_count: int,
    original_label_count: int,
    final_label_to_rows: dict[str, set[int]],
    merge_steps: list[MergeStep],
    removed_labels: list[RemovedRareLabel],
    removed_pseudo_label_occurrences: int,
    removed_cve_count: int,
    min_support: int,
) -> None:
    """Print final aggregation statistics."""

    active_counts = {
        cwe_id: len(row_indexes)
        for cwe_id, row_indexes in final_label_to_rows.items()
        if row_indexes
    }

    labels_below_threshold = {
        cwe_id: count
        for cwe_id, count in active_counts.items()
        if count < min_support
    }

    print()
    print("Hierarchy-aware merge completed.")
    print(f"Minimum support: {min_support}")
    print(f"Original number of CVEs: {original_cve_count}")
    print(f"Final number of CVEs: {final_cve_count}")
    print(f"Removed CVEs without remaining CWE: {removed_cve_count}")
    print(
        "Removed pseudo-label occurrences: "
        f"{removed_pseudo_label_occurrences}"
    )
    print(f"Original number of active CWE labels: {original_label_count}")
    print(f"Number of progressive merge steps: {len(merge_steps)}")
    print(f"Removed unmergeable rare CWE labels: {len(removed_labels)}")
    print(f"Final number of active CWE labels: {len(active_counts)}")
    print(
        "Final labels below threshold: "
        f"{len(labels_below_threshold)}"
    )

    if labels_below_threshold:
        print(
            "Warning: unexpected labels below threshold remain: "
            + ", ".join(
                f"{cwe_id} ({count})"
                for cwe_id, count in sorted(
                    labels_below_threshold.items(),
                    key=lambda item: cwe_sort_key(item[0]),
                )
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Progressively merge rare CWE labels into hierarchy ancestors."
        )
    )

    parser.add_argument(
        "--cve-dataset",
        type=Path,
        default=DEFAULT_CVE_DATASET_FILE,
    )
    parser.add_argument(
        "--cwe-hierarchy",
        type=Path,
        default=DEFAULT_CWE_HIERARCHY_FILE,
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
    )
    parser.add_argument(
        "--output-cve",
        type=Path,
        default=DEFAULT_MERGED_CVE_FILE,
    )
    parser.add_argument(
        "--output-cwe",
        type=Path,
        default=DEFAULT_MERGED_CWE_FILE,
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=DEFAULT_MAPPING_FILE,
    )
    parser.add_argument(
        "--output-removed-cves",
        type=Path,
        default=DEFAULT_REMOVED_CVE_FILE,
    )
    parser.add_argument(
        "--output-removed-labels",
        type=Path,
        default=DEFAULT_REMOVED_LABELS_FILE,
    )

    return parser.parse_args()


def main() -> None:
    """Run progressive hierarchy-aware aggregation."""

    args = parse_args()

    nodes = load_cwe_hierarchy(args.cwe_hierarchy)

    (
        rows,
        labels_by_row,
        fieldnames,
        removed_pseudo_label_occurrences,
    ) = load_cve_dataset(args.cve_dataset)

    original_cve_count = len(rows)
    label_to_rows = build_label_memberships(labels_by_row)

    original_label_count = sum(
        bool(row_indexes)
        for row_indexes in label_to_rows.values()
    )

    unknown_labels = sorted(
        set(label_to_rows) - set(nodes),
        key=cwe_sort_key,
    )

    if unknown_labels:
        print(
            "Warning: labels absent from the CWE hierarchy will be removed "
            "only if they remain below the minimum support: "
            + ", ".join(unknown_labels)
        )

    merge_steps, unmerged_rare_labels = merge_labels_bottom_up(
        labels_by_row=labels_by_row,
        label_to_rows=label_to_rows,
        nodes=nodes,
        min_support=args.min_support,
    )

    removed_labels = remove_unmergeable_rare_labels(
        labels_by_row=labels_by_row,
        label_to_rows=label_to_rows,
        rare_labels=unmerged_rare_labels,
        nodes=nodes,
    )

    (
        final_rows,
        final_labels_by_row,
        removed_cve_rows,
    ) = filter_rows_without_labels(rows, labels_by_row)

    # Rebuild memberships because CVE row indexes changed after filtering.
    final_label_to_rows = build_label_memberships(final_labels_by_row)

    write_cve_dataset(
        path=args.output_cve,
        rows=final_rows,
        labels_by_row=final_labels_by_row,
        fieldnames=fieldnames,
    )
    write_removed_cves(
        path=args.output_removed_cves,
        rows=removed_cve_rows,
        fieldnames=fieldnames,
    )
    write_merge_mapping(
        path=args.output_mapping,
        merge_steps=merge_steps,
    )
    write_removed_labels(
        path=args.output_removed_labels,
        removed_labels=removed_labels,
    )
    write_final_cwe_counts(
        path=args.output_cwe,
        nodes=nodes,
        label_to_rows=final_label_to_rows,
        min_support=args.min_support,
    )

    print_summary(
        original_cve_count=original_cve_count,
        final_cve_count=len(final_rows),
        original_label_count=original_label_count,
        final_label_to_rows=final_label_to_rows,
        merge_steps=merge_steps,
        removed_labels=removed_labels,
        removed_pseudo_label_occurrences=removed_pseudo_label_occurrences,
        removed_cve_count=len(removed_cve_rows),
        min_support=args.min_support,
    )

    print()
    print(f"Merged CVE dataset: {args.output_cve}")
    print(f"Final CWE counts: {args.output_cwe}")
    print(f"Progressive merge mapping: {args.output_mapping}")
    print(f"Removed rare labels: {args.output_removed_labels}")
    print(f"Removed CVEs without labels: {args.output_removed_cves}")


if __name__ == "__main__":
    main()

