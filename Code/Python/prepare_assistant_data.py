"""Prepare new, explicitly partitioned assistant corpora without changing sources.

Usage: python Code/Python/prepare_assistant_data.py --spec sources.json --output NEW_DIR
The spec is schema_version 1 with a release name and sources list. Each source
requires id, path (relative to the spec), format (text/jsonl/chat), group,
partition (train/validation/test), language, provenance, permitted_use, extraction.
Group partitions are reviewed input, never randomly inferred. JSONL base records
have a text string. Chat records have schema_version 1 and messages of role/content.
Example source: {"id":"book-a", "path":"raw/book.txt", "format":"text",
"group":"book-family", "partition":"train", "language":"en",
"provenance":"Author/source reference", "permitted_use":"Reviewed permission",
"extraction":"UTF-8 source, manually reviewed"}.

The manifest records exact-duplicate removals, all members, cross-stage/message
overlap checks, and bounded near-duplicate candidates. COMPLETE means publication
finished, NOT that human corpus review or assistant-quality acceptance passed.
Base records with fewer than two normalized characters are reported and skipped;
empty chat messages are rejected. Group membership is per source file: separate
files explicitly when records have different source families or partitions.
No tokenizer/context-length, factual quality, license, or privacy certification
is performed. Near detection compares normalized word shingles and can miss
paraphrases or OCR changes. Raw content is preserved, not automatically cleaned.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata


PARTITIONS = ("train", "validation", "test")
METADATA = ("id", "path", "format", "group", "partition", "language",
            "provenance", "permitted_use", "extraction")


@dataclass(frozen=True)
class Limits:
    max_source_bytes: int = 16 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024
    max_record_bytes: int = 1024 * 1024
    max_records: int = 10_000
    max_audit_units: int = 20_000
    max_pairs: int = 1_000_000

    def validate(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("All preparation limits must be positive integers")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def parse_json(data):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(data, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def reject_links(path):
    """Reject symlinks and Windows reparse points in every existing component."""
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symlink/reparse path is not allowed: {part}")


def input_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Source path must use a relative path with '/' separators")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Source path must not contain traversal: {value}")
    result = root.joinpath(*relative.parts)
    reject_links(result)
    if not result.is_file():
        raise ValueError(f"Source is not a regular file: {value}")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source escapes specification directory: {value}")
    return result


def bounded_read(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Source exceeds {limit} byte limit: {path}")
    return data


def chat_messages(record, location):
    if (not isinstance(record, dict) or set(record) != {"schema_version", "messages"}
            or type(record["schema_version"]) is not int or record["schema_version"] != 1):
        raise ValueError(f"{location}: chat requires schema_version 1 and messages only")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location}: messages must be a nonempty list")
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{location}: message {index + 1} requires role/content only")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{location}: message {index + 1} content must be nonblank")
        if index == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError(f"{location}: message {index + 1} must have role {expected}")
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{location}: training conversation must end with assistant")
    return messages


def read_records(source, raw, limits):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source['id']}: source must be UTF-8 ({error})") from error
    if source["format"] == "text":
        yield source["id"], text, None
        return
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        location = f"{source['id']}/@record-{line_number}"
        if len(line.encode("utf-8")) > limits.max_record_bytes:
            raise ValueError(f"{location}: record exceeds {limits.max_record_bytes} bytes")
        try:
            record = parse_json(line)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{location}: invalid JSON: {error}") from error
        if source["format"] == "chat":
            messages = chat_messages(record, location)
            yield location, None, messages
        else:
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{location}: base JSONL requires a text string")
            yield location, record["text"], None


def shingles(text):
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if len(words) < 5:
        return {tuple(words)} if words else set()
    return {tuple(words[index:index + 5]) for index in range(len(words) - 4)}


def audit_units(units, limits):
    pair_count = len(units) * (len(units) - 1) // 2
    if pair_count > limits.max_pairs:
        raise ValueError(f"Near-duplicate audit requires {pair_count} pairs; limit is "
                         f"{limits.max_pairs}. Review smaller releases or explicitly raise the cap")
    exact, near = [], []
    features = [shingles(unit["text"]) for unit in units]
    for left_index, left in enumerate(units):
        for right_index in range(left_index + 1, len(units)):
            right = units[right_index]
            if left["record"] == right["record"]:
                continue
            pair = {"left": left["id"], "right": right["id"],
                    "cross_partition": left["partition"] != right["partition"],
                    "cross_stage": left["stage"] != right["stage"]}
            if left["text"] == right["text"]:
                if pair["cross_partition"]:
                    raise ValueError(f"Exact content leakage across partitions: {left['id']} "
                                     f"({left['partition']}) and {right['id']} ({right['partition']})")
                exact.append(pair)
            elif features[left_index] and features[right_index]:
                smaller = min(len(features[left_index]), len(features[right_index]))
                score = len(features[left_index] & features[right_index]) / smaller
                if score >= 0.8:
                    near.append({**pair, "shingle_containment": round(score, 6),
                                 "review": "unresolved"})
    return exact, near, pair_count


def prepare(spec_path, output, limits=None):
    """Validate/audit everything before reserving a new directory; return manifest."""
    limits = limits or Limits()
    limits.validate()
    spec_path, output = Path(spec_path).absolute(), Path(output).absolute()
    reject_links(spec_path)
    reject_links(output)
    if output.exists():
        raise ValueError(f"Output already exists; use a new release directory: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent does not exist: {output.parent}")
    raw_spec = bounded_read(spec_path, limits.max_source_bytes)
    try:
        spec = parse_json(raw_spec)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"Invalid specification JSON: {error}") from error
    if (not isinstance(spec, dict) or type(spec.get("schema_version")) is not int
            or spec["schema_version"] != 1):
        raise ValueError("Specification requires schema_version 1")
    if set(spec) != {"schema_version", "release", "sources"}:
        raise ValueError("Specification requires exactly schema_version, release, sources")
    if not isinstance(spec["release"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", spec["release"]):
        raise ValueError("Release must be a nonempty ASCII identifier")
    if not isinstance(spec["sources"], list) or not spec["sources"]:
        raise ValueError("Specification requires a nonempty sources list")
    sources, records, units, rejected = [], [], [], []
    seen_ids, seen_paths, groups = set(), set(), {}
    total_bytes = 0
    for source in sorted(spec["sources"], key=lambda item: str(item.get("id", ""))
                         if isinstance(item, dict) else ""):
        if not isinstance(source, dict) or set(source) != set(METADATA):
            raise ValueError(f"Each source requires exactly: {', '.join(METADATA)}")
        if any(not isinstance(source[key], str) or not source[key].strip() for key in METADATA):
            raise ValueError("Source metadata fields must be nonempty strings, including permitted_use")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source["id"]) or source["id"] in seen_ids:
            raise ValueError(f"Invalid or duplicate source id: {source['id']}")
        seen_ids.add(source["id"])
        if source["format"] not in ("text", "jsonl", "chat") or source["partition"] not in PARTITIONS:
            raise ValueError(f"{source['id']}: unsupported format or partition")
        group = source["group"]
        if group in groups and groups[group] != source["partition"]:
            raise ValueError(f"Source group {group!r} crosses partitions")
        groups[group] = source["partition"]
        path = input_path(spec_path.parent, source["path"])
        info = path.stat()
        physical = (info.st_dev, info.st_ino) if info.st_ino else str(path.resolve()).casefold()
        if physical in seen_paths:
            raise ValueError(f"Source alias or repeated physical input: {source['path']}")
        seen_paths.add(physical)
        raw = bounded_read(path, limits.max_source_bytes)
        total_bytes += len(raw)
        if total_bytes > limits.max_total_bytes:
            raise ValueError(f"Total input exceeds {limits.max_total_bytes} bytes")
        sources.append({**source, "bytes": len(raw), "sha256": digest(raw)})
        for record_id, text, messages in read_records(source, raw, limits):
            if len(records) + len(rejected) >= limits.max_records:
                raise ValueError(f"Record count exceeds {limits.max_records}")
            stage = "chat" if messages is not None else "base"
            payload = encoded({"schema_version": 1, "messages": messages}) if messages is not None else text.encode("utf-8")
            if len(payload) > limits.max_record_bytes:
                raise ValueError(f"{record_id}: record exceeds {limits.max_record_bytes} bytes")
            if text is not None and len(normalized(text)) < 2:
                rejected.append({"id": record_id, "reason": "blank" if not text.strip() else "too_short"})
                continue
            audit_texts = [message["content"] for message in messages] if messages is not None else [text]
            for index, content in enumerate(audit_texts):
                units.append({"id": f"{record_id}/@message-{index + 1}" if messages is not None else record_id,
                              "record": record_id, "text": normalized(content),
                              "partition": source["partition"], "stage": stage})
                if len(units) > limits.max_audit_units:
                    raise ValueError(f"Audit unit count exceeds {limits.max_audit_units}")
            canonical = (encoded([{**message, "content": normalized(message["content"])} for message in messages])
                         if messages is not None else normalized(text).encode("utf-8"))
            records.append({"id": record_id, "source": source["id"], "group": group,
                            "partition": source["partition"], "stage": stage,
                            "bytes": len(payload), "content_sha256": digest(payload),
                            "normalized_sha256": digest(canonical), "payload": payload,
                            "messages": messages})
    if not records:
        raise ValueError("No usable records after blank/too-short rejection")
    exact, near, pair_count = audit_units(units, limits)
    representatives, outputs, membership = {}, [], []
    counts = {stage: {part: 0 for part in PARTITIONS} for stage in ("base", "chat")}
    for record in records:
        key = (record["stage"], record["normalized_sha256"])
        member = {key: value for key, value in record.items() if key not in ("payload", "messages")}
        if key in representatives:
            member["duplicate_of"] = representatives[key]
        else:
            representatives[key] = record["id"]
            suffix = "jsonl" if record["stage"] == "chat" else "txt"
            name = digest(record["id"].encode("utf-8")) + "." + suffix
            relative = f"{record['stage']}/{record['partition']}/{name}"
            payload = record["payload"]
            if record["stage"] == "chat":
                payload = (json.dumps({"schema_version": 1, "messages": record["messages"]},
                                     ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            outputs.append((relative, payload))
            member["output"] = relative
            member["output_sha256"] = digest(payload)
            counts[record["stage"]][record["partition"]] += 1
        membership.append(member)
    manifest = {"schema_version": 1, "release": spec["release"],
                "spec_sha256": digest(raw_spec), "limits": vars(limits),
                "preparation": {"min_base_normalized_characters": 2,
                                "blank_chat_messages": "reject",
                                "base_output": "unchanged UTF-8 text per retained document",
                                "chat_output": "schema-1 JSONL per retained conversation",
                                "source_bytes": total_bytes,
                                "input_records": len(records) + len(rejected)},
                "sources": sources, "groups": groups, "records": membership,
                "rejected": rejected, "counts": counts,
                "exact_overlaps": exact, "near_candidates": near,
                "audit": {"normalization": "Unicode NFKC + whitespace collapse; case retained for exact",
                          "near_method": "casefold word 5-shingle containment >= 0.8; short units use whole word tuple",
                          "units": len(units), "pairs_upper_bound": pair_count,
                          "human_review": "required", "unresolved_near_candidates": len(near),
                          "all_stage_partitions_nonempty": all(count for stage in counts.values() for count in stage.values()),
                          "limitations": "May miss paraphrases, short embedded excerpts and OCR differences. Permissions, privacy, quality and tokenizer lengths require separate review."}}
    manifest_bytes = encoded(manifest)
    output.mkdir(exist_ok=False)
    # Any write failure deliberately leaves an incomplete new directory without
    # COMPLETE for inspection. No existing path is removed or overwritten.
    for stage in ("base", "chat"):
        for partition in PARTITIONS:
            (output / stage / partition).mkdir(parents=True)
    for relative, payload in outputs:
        with (output / relative).open("xb") as stream:
            stream.write(payload)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    with (output / "COMPLETE").open("xb") as stream:
        stream.write(encoded({"schema_version": 1, "manifest_sha256": digest(manifest_bytes)}))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for name, default in vars(Limits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args(argv)
    try:
        limits = Limits(**{name: getattr(args, name) for name in vars(Limits())})
        manifest = prepare(args.spec, args.output, limits)
    except (OSError, ValueError, RecursionError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"release": manifest["release"], "counts": manifest["counts"],
                      "human_review": "required", "near_candidates": len(manifest["near_candidates"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
