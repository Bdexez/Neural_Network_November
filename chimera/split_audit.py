"""Détecter les exemples identiques et les fenêtres temporelles qui se recouvrent."""

from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal, DecimalException
import hashlib
import json
import os
from pathlib import Path

from .data import file_identity


MODEL_FIELDS = ("text", "image", "image_path", "series", "class_label", "future")


def canonical(value):
    """Encodage typé ; les nombres JSON égaux restent égaux sans perte décimale."""
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        if not any(digits):
            return ["number", 0, "0", 0]
        digits = list(digits)
        while digits[-1] == 0:
            digits.pop()
            exponent += 1
        return ["number", sign, "".join(map(str, digits)), exponent]
    if isinstance(value, float):
        return ["nonfinite", str(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, list):
        return ["array", [canonical(item) for item in value]]
    if isinstance(value, dict):
        return ["object", [[key, canonical(value[key])] for key in sorted(value)]]
    raise ValueError("Type JSON non pris en charge")


def digest_value(value):
    encoded = json.dumps(canonical(value), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).digest()


def _same_file(first, second):
    if Path(first).resolve() == Path(second).resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


def _scan(path, *, fields, interval_fields, output):
    path = Path(path).resolve()
    identity = file_identity(path.stat())
    file_hash = hashlib.sha256()
    counts, first_lines = Counter(), {}
    intervals = defaultdict(list)
    records = 0
    with path.open("rb") as stream:
        if file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError(f"Le fichier a changé pendant l'audit : {path}")
        for line_number, line in enumerate(stream, 1):
            file_hash.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_int=Decimal, parse_float=Decimal, parse_constant=float)
                if not isinstance(record, dict):
                    raise ValueError("un objet JSON est attendu")
                if not any(key in record for key in ("text", "image", "image_path", "series")):
                    raise ValueError("une modalité d'entrée est requise")
                if "image_path" in record and "image" not in record:
                    image_path = record["image_path"]
                    if not isinstance(image_path, str) or not image_path:
                        raise ValueError("image_path doit être une chaîne non vide")
                    if output is not None and _same_file(output, path.parent / image_path):
                        raise ValueError("la sortie ne peut pas écraser une image d'entrée")
                if fields is None:
                    selected = {key: record[key] for key in MODEL_FIELDS if key in record}
                    if "image" in selected:
                        selected.pop("image_path", None)
                else:
                    absent = [key for key in fields if key not in record]
                    if absent:
                        raise ValueError(f"champs absents : {', '.join(absent)}")
                    selected = {key: record[key] for key in fields}
                digest = digest_value(selected)
                counts[digest] += 1
                first_lines.setdefault(digest, line_number)
                records += 1
                if interval_fields:
                    group_key, start_key, end_key = interval_fields
                    if any(key not in record for key in interval_fields):
                        raise ValueError("les trois champs d'intervalle sont requis")
                    group, start, end = record[group_key], record[start_key], record[end_key]
                    if not (isinstance(group, (str, Decimal)) and not isinstance(group, bool)
                            and (not isinstance(group, str) or group)):
                        raise ValueError("identifiant de série numérique ou chaîne non vide requis")
                    if not isinstance(start, Decimal) or not isinstance(end, Decimal) or start >= end:
                        raise ValueError("bornes numériques finies requises avec début < fin")
                    intervals[digest_value(group)].append((start, end, line_number))
            except (ValueError, TypeError, DecimalException, UnicodeDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if (file_identity(os.fstat(stream.fileno())) != identity
                or file_identity(path.stat()) != identity):
            raise ValueError(f"Le fichier a changé pendant l'audit : {path}")
    if not records:
        raise ValueError(f"Le fichier ne contient aucun exemple : {path}")
    return dict(path=str(path), sha256=file_hash.hexdigest(), records=records, counts=counts,
                first_lines=first_lines, intervals=intervals)


def _index_intervals(values):
    ordered = sorted(values, key=lambda row: row[0])
    starts, maxima, lines = [], [], []
    for start, end, line in ordered:
        starts.append(start)
        if not maxima or end > maxima[-1]:
            maxima.append(end)
            lines.append(line)
        else:
            maxima.append(maxima[-1])
            lines.append(lines[-1])
    return starts, maxima, lines


def _overlapping_rows(values, index, limit):
    starts, maxima, lines = index
    count, examples = 0, []
    for start, end, line in values:
        position = bisect_left(starts, end) - 1
        if position >= 0 and maxima[position] > start:
            count += 1
            if len(examples) < limit:
                examples.append([line, lines[position]])
    return count, examples


def audit_splits(splits, *, fields=None, interval_fields=None, output=None, max_examples=5):
    """Audit exact, sans conserver ni exporter les valeurs des enregistrements."""
    if len(splits) < 2:
        raise ValueError("Au moins deux jeux sont requis")
    if fields is not None and (not fields or len(set(fields)) != len(fields) or any(not field for field in fields)):
        raise ValueError("Les champs de comparaison doivent être non vides et distincts")
    if interval_fields is not None and (len(interval_fields) != 3 or len(set(interval_fields)) != 3
                                        or any(not field for field in interval_fields)):
        raise ValueError("Trois noms de champs d'intervalle distincts sont requis")
    if type(max_examples) is not int or not 0 <= max_examples <= 100:
        raise ValueError("max_examples doit être compris entre 0 et 100")
    if output is not None and any(_same_file(output, path) for path in splits.values()):
        raise ValueError("La sortie doit être distincte des jeux d'entrée")
    scanned = {name: _scan(path, fields=fields, interval_fields=interval_fields, output=output)
               for name, path in splits.items()}
    summaries = {name: dict(path=data["path"], sha256=data["sha256"], records=data["records"],
                            unique_records=len(data["counts"]),
                            duplicate_rows=data["records"] - len(data["counts"])) for name, data in scanned.items()}
    pairs = []
    names = list(scanned)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left, right = scanned[left_name], scanned[right_name]
            shared = left["counts"].keys() & right["counts"].keys()
            pair = dict(left=left_name, right=right_name, same_jsonl=left["sha256"] == right["sha256"],
                        shared_unique_records=len(shared),
                        shared_rows_left=sum(left["counts"][key] for key in shared),
                        shared_rows_right=sum(right["counts"][key] for key in shared),
                        examples=[dict(left_line=left["first_lines"][key], right_line=right["first_lines"][key])
                                  for key in sorted(shared)[:max_examples]])
            if interval_fields:
                left_rows = right_rows = groups = 0
                interval_examples = []
                for group in sorted(left["intervals"].keys() & right["intervals"].keys()):
                    a, b = left["intervals"][group], right["intervals"][group]
                    n, examples = _overlapping_rows(a, _index_intervals(b), max_examples - len(interval_examples))
                    m, _ = _overlapping_rows(b, _index_intervals(a), 0)
                    left_rows += n
                    right_rows += m
                    groups += int(n > 0)
                    interval_examples.extend(dict(left_line=x, right_line=y) for x, y in examples)
                pair["intervals"] = dict(groups_with_overlap=groups, overlapping_rows_left=left_rows,
                                         overlapping_rows_right=right_rows, examples=interval_examples)
            pairs.append(pair)
    return dict(fields=list(fields) if fields is not None else list(MODEL_FIELDS),
                interval_fields=list(interval_fields) if interval_fields else None, splits=summaries, pairs=pairs,
                scope="Comparaison exacte des champs choisis ; l'absence de correspondance ne prouve pas l'indépendance.")
