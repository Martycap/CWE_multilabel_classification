import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import requests


CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
HIERARCHY_VIEW_ID = "1000"

CWE_ZIP_FILE = Path("data/raw/cwe/cwe_catalog.xml.zip")
CWE_XML_FILE = Path("data/raw/cwe/cwe_catalog.xml")
CWE_CSV_FILE = Path("data/cwe_hierarchy.csv")
CVE_DATASET_FILE = Path("data/cve_dataset.csv")
CWE_COUNTS_CSV_FILE = Path("data/cwe_hierarchy_with_counts.csv")
CWE_FILTERED_CSV_FILE = Path(
    "data/cwe_without_prohibited_with_counts.csv"
)


def download_cwe_catalog() -> None:
    """Download the latest CWE catalog in XML format."""

    CWE_ZIP_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CWE_ZIP_FILE.exists():
        print(f"{CWE_ZIP_FILE.name} already exists.")
        return

    print("Downloading the CWE catalog...")

    response = requests.get(CWE_URL, timeout=120)
    response.raise_for_status()

    CWE_ZIP_FILE.write_bytes(response.content)


def unzip_cwe_catalog() -> None:
    """Extract the XML file contained in the downloaded ZIP archive."""

    if CWE_XML_FILE.exists():
        print(f"{CWE_XML_FILE.name} already exists.")
        return

    print("Extracting the CWE catalog...")

    with zipfile.ZipFile(CWE_ZIP_FILE, "r") as archive:
        xml_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".xml")
        )

        with archive.open(xml_name) as compressed_file:
            with open(CWE_XML_FILE, "wb") as output_file:
                shutil.copyfileobj(
                    compressed_file,
                    output_file,
                )


def calculate_cwe_depths(
    cwes: dict[str, dict[str, object]],
) -> dict[str, int]:
    """
    Calculate the depth of every CWE in the hierarchy.

    A CWE without parents has depth 0.

    For a CWE with one or more parents, its depth is the maximum
    parent depth plus one. Using the maximum depth places the CWE
    at its deepest position when multiple hierarchy paths exist.
    """

    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def calculate_depth(cwe_id: str) -> int:
        if cwe_id in depths:
            return depths[cwe_id]

        if cwe_id in visiting:
            raise ValueError(
                f"Cycle detected in the CWE hierarchy at {cwe_id}."
            )

        visiting.add(cwe_id)

        cwe_data = cwes[cwe_id]
        parents = cwe_data["parents"]

        valid_parents = [
            parent_id
            for parent_id in parents
            if parent_id in cwes
        ]

        if not valid_parents:
            depth = 0
        else:
            depth = 1 + max(
                calculate_depth(parent_id)
                for parent_id in valid_parents
            )

        visiting.remove(cwe_id)
        depths[cwe_id] = depth

        return depth

    for cwe_id in cwes:
        calculate_depth(cwe_id)

    return depths


def create_cwe_hierarchy_csv() -> None:
    """
    Create a CSV containing each CWE with:

    - abstraction level;
    - status;
    - mapping usage;
    - primary parent;
    - other direct parents;
    - children;
    - hierarchy depth.
    """

    print("Parsing the CWE hierarchy...")

    tree = ElementTree.parse(CWE_XML_FILE)
    root = tree.getroot()

    cwes: dict[str, dict[str, object]] = {}

    for weakness in root.findall(".//{*}Weakness"):
        weakness_id = weakness.get("ID")

        if weakness_id is None:
            continue

        cwe_id = f"CWE-{weakness_id}"

        name = weakness.get("Name", "")
        abstraction = weakness.get("Abstraction", "")
        status = weakness.get("Status", "")

        mapping_usage = weakness.findtext(
            "./{*}Mapping_Notes/{*}Usage",
            default="",
        ).strip()

        primary_parents: set[str] = set()
        other_parents: set[str] = set()

        for relation in weakness.findall(
            "./{*}Related_Weaknesses/{*}Related_Weakness"
        ):
            if (
                relation.get("Nature") != "ChildOf"
                or relation.get("View_ID") != HIERARCHY_VIEW_ID
            ):
                continue

            parent_cwe_id = relation.get("CWE_ID")

            if parent_cwe_id is None:
                continue

            parent_id = f"CWE-{parent_cwe_id}"
            ordinal = relation.get("Ordinal", "").strip().lower()

            if ordinal == "primary":
                primary_parents.add(parent_id)
            else:
                other_parents.add(parent_id)

        # Normally MITRE defines at most one primary parent.
        # Sorting makes the result deterministic in the unexpected
        # case in which multiple primary relationships are present.
        sorted_primary_parents = sorted(
            primary_parents,
            key=lambda value: int(value.split("-")[1]),
        )

        primary_parent = (
            sorted_primary_parents[0]
            if sorted_primary_parents
            else ""
        )

        # Any additional primary parents are retained among the
        # alternative direct parents.
        if len(sorted_primary_parents) > 1:
            other_parents.update(sorted_primary_parents[1:])

        # Avoid storing the primary parent twice.
        if primary_parent:
            other_parents.discard(primary_parent)

        parents = set(other_parents)

        if primary_parent:
            parents.add(primary_parent)

        cwes[cwe_id] = {
            "name": name,
            "abstraction": abstraction,
            "status": status,
            "mapping_usage": mapping_usage,
            "primary_parent": primary_parent,
            "other_parents": other_parents,
            "parents": parents,
            "children": set(),
        }

    # Build children by reversing all direct parent relationships.
    for cwe_id, cwe_data in cwes.items():
        parents = cwe_data["parents"]

        for parent_id in parents:
            if parent_id in cwes:
                cwes[parent_id]["children"].add(cwe_id)

    depths = calculate_cwe_depths(cwes)

    CWE_CSV_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CWE_CSV_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "cwe_id",
                "name",
                "abstraction",
                "status",
                "mapping_usage",
                "primary_parent",
                "other_parents",
                "children",
                "depth",
            ]
        )

        sorted_cwe_ids = sorted(
            cwes,
            key=lambda value: int(value.split("-")[1]),
        )

        for cwe_id in sorted_cwe_ids:
            cwe_data = cwes[cwe_id]

            sorted_other_parents = sorted(
                cwe_data["other_parents"],
                key=lambda value: int(value.split("-")[1]),
            )

            sorted_children = sorted(
                cwe_data["children"],
                key=lambda value: int(value.split("-")[1]),
            )

            writer.writerow(
                [
                    cwe_id,
                    cwe_data["name"],
                    cwe_data["abstraction"],
                    cwe_data["status"],
                    cwe_data["mapping_usage"],
                    cwe_data["primary_parent"],
                    json.dumps(sorted_other_parents),
                    json.dumps(sorted_children),
                    depths[cwe_id],
                ]
            )

    print(f"CSV file created: {CWE_CSV_FILE}")
    print(f"Number of saved CWEs: {len(cwes)}")
    print(
        "Maximum hierarchy depth: "
        f"{max(depths.values(), default=0)}"
    )


def add_cve_counts_to_cwe_csv() -> None:
    """
    Create a new CWE CSV containing the number of CVEs
    associated with each CWE.
    """

    print("Counting CVEs associated with each CWE...")

    cwe_counts: Counter[str] = Counter()

    with open(
        CVE_DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            cwe_ids = json.loads(row["cwe_ids"])

            # Count each CWE only once for the same CVE.
            for cwe_id in set(cwe_ids):
                cwe_counts[cwe_id] += 1

    with open(
        CWE_CSV_FILE,
        "r",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV file {CWE_CSV_FILE} has no header."
            )

        fieldnames = [*reader.fieldnames, "cve_count"]

        with open(
            CWE_COUNTS_CSV_FILE,
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in reader:
                cwe_id = row["cwe_id"]
                row["cve_count"] = cwe_counts.get(cwe_id, 0)

                writer.writerow(row)

    print(f"CSV file created: {CWE_COUNTS_CSV_FILE}")


def filter_allowed_cwes() -> None:
    """
    Create a CSV excluding Prohibited CWEs
    and CWEs with no associated CVEs.

    Allowed, Allowed-with-Review, and Discouraged CWEs are kept.
    Pillar, Class, Base, Variant, and Compound abstractions are
    all permitted.
    """

    print("Filtering CWEs...")

    saved_cwes = 0
    removed_prohibited_cwes = 0
    removed_empty_cwes = 0

    with open(
        CWE_COUNTS_CSV_FILE,
        "r",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV file {CWE_COUNTS_CSV_FILE} has no header."
            )

        with open(
            CWE_FILTERED_CSV_FILE,
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=reader.fieldnames,
            )

            writer.writeheader()

            for row in reader:
                mapping_usage = (
                    row["mapping_usage"]
                    .strip()
                    .lower()
                )
                cve_count = int(row["cve_count"])

                if mapping_usage == "prohibited":
                    removed_prohibited_cwes += 1
                    continue

                if cve_count == 0:
                    removed_empty_cwes += 1
                    continue

                writer.writerow(row)
                saved_cwes += 1

    print(
        f"Filtered CSV file created: "
        f"{CWE_FILTERED_CSV_FILE}"
    )
    print(f"Number of saved CWEs: {saved_cwes}")
    print(
        "Number of removed Prohibited CWEs: "
        f"{removed_prohibited_cwes}"
    )
    print(
        "Number of removed CWEs with zero CVEs: "
        f"{removed_empty_cwes}"
    )


def main() -> None:
    download_cwe_catalog()
    unzip_cwe_catalog()
    create_cwe_hierarchy_csv()
    add_cve_counts_to_cwe_csv()
    filter_allowed_cwes()


if __name__ == "__main__":
    main()
