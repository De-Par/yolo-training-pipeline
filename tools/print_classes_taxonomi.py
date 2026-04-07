from __future__ import annotations

from tools._runtime import bootstrap_project_root
bootstrap_project_root(__file__, levels=1)

import argparse
import ast
import yaml
import json

from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml_names(path: Path) -> Dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = data.get('names', {})

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        result: Dict[int, str] = {}
        for key, value in names.items():
            result[int(key)] = str(value)
        return dict(sorted(result.items()))

    return {}


def load_pt_names(path: Path) -> Dict[int, str]:
    from ultralytics import YOLO

    model = YOLO(str(path))
    names = getattr(model, 'names', None)

    if names is None:
        return {}

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        result: Dict[int, str] = {}
        for key, value in names.items():
            result[int(key)] = str(value)
        return dict(sorted(result.items()))

    return {}


def try_parse_names_blob(value: Any) -> Optional[Dict[int, str]]:
    if value is None:
        return None

    if isinstance(value, dict):
        return {int(k): str(v) for k, v in sorted(value.items(), key=lambda item: int(item[0]))}

    if isinstance(value, list):
        return {i: str(name) for i, name in enumerate(value)}

    text = str(value).strip()
    if not text:
        return None

    parsers = [
        json.loads,
        ast.literal_eval,
    ]

    for parser in parsers:
        try:
            parsed = parser(text)
        except Exception:
            continue

        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in sorted(parsed.items(), key=lambda item: int(item[0]))}
        if isinstance(parsed, list):
            return {i: str(name) for i, name in enumerate(parsed)}

    return None


def load_onnx_names(path: Path) -> Dict[int, str]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta()
    custom = dict(meta.custom_metadata_map or {})

    candidate_keys = [
        'names',
        'classes',
        'class_names',
    ]

    for key in candidate_keys:
        if key in custom:
            parsed = try_parse_names_blob(custom[key])
            if parsed:
                return parsed

    return {}


def print_mapping(title: str, mapping: Dict[int, str]) -> None:
    print(f'\n{title}')
    if not mapping:
        print('  <empty>')
        return

    for class_id, class_name in mapping.items():
        print(f'  {class_id:>3} -> {class_name}')


def compare_by_id(left_name: str, left: Dict[int, str], right_name: str, right: Dict[int, str]) -> None:
    print(f'\nCompare by id: {left_name} vs {right_name}')

    all_ids = sorted(set(left) | set(right))
    any_diff = False

    for class_id in all_ids:
        a = left.get(class_id)
        b = right.get(class_id)
        if a != b:
            any_diff = True
            print(f'  id={class_id}: {left_name}={a!r} | {right_name}={b!r}')

    if not any_diff:
        print('  mappings are identical by id')


def compare_by_name(left_name: str, left: Dict[int, str], right_name: str, right: Dict[int, str]) -> None:
    print(f'\nCompare by name: {left_name} vs {right_name}')

    left_rev = {name: idx for idx, name in left.items()}
    right_rev = {name: idx for idx, name in right.items()}

    all_names = sorted(set(left_rev) | set(right_rev))
    any_diff = False

    for class_name in all_names:
        a = left_rev.get(class_name)
        b = right_rev.get(class_name)
        if a != b:
            any_diff = True
            print(f'  class={class_name!r}: {left_name}_id={a} | {right_name}_id={b}')

    if not any_diff:
        print('  mappings are identical by name')


def main() -> None:
    parser = argparse.ArgumentParser(description="Script for classes comparison between .pt / .yaml / .onnx")
    parser.add_argument('--pt', type=Path, required=True, help='Path to best.pt')
    parser.add_argument('--yaml', type=Path, required=True, help='Path to dataset yaml')
    parser.add_argument('--onnx', type=Path, default=None, help='Optional path to best.onnx')
    args = parser.parse_args()

    pt_path = args.pt.expanduser().resolve()
    yaml_path = args.yaml.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve() if args.onnx else None

    if not pt_path.exists():
        raise FileNotFoundError(f'PT file not found: {pt_path}')
    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')
    if onnx_path is not None and not onnx_path.exists():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')

    pt_names = load_pt_names(pt_path)
    yaml_names = load_yaml_names(yaml_path)
    onnx_names = load_onnx_names(onnx_path) if onnx_path is not None else {}

    print(f'PT   : {pt_path}')
    print(f'YAML : {yaml_path}')
    if onnx_path is not None:
        print(f'ONNX : {onnx_path}')

    print_mapping('PT names', pt_names)
    print_mapping('YAML names', yaml_names)
    if onnx_path is not None:
        print_mapping('ONNX names', onnx_names)

    compare_by_id('PT', pt_names, 'YAML', yaml_names)
    compare_by_name('PT', pt_names, 'YAML', yaml_names)

    if onnx_path is not None:
        compare_by_id('PT', pt_names, 'ONNX', onnx_names)
        compare_by_name('PT', pt_names, 'ONNX', onnx_names)
        compare_by_id('YAML', yaml_names, 'ONNX', onnx_names)
        compare_by_name('YAML', yaml_names, 'ONNX', onnx_names)

    print('\nSummary')
    print(f'  PT   == YAML by id   : {pt_names == yaml_names}')
    if onnx_path is not None:
        print(f'  PT   == ONNX by id   : {pt_names == onnx_names}')
        print(f'  YAML == ONNX by id   : {yaml_names == onnx_names}')

    if onnx_path is not None and not onnx_names:
        print('\nNote')
        print('  ONNX class names were not found in metadata.')
        print('  This does not always mean the ONNX model is wrong.')
        print('  Some exports simply do not embed names into model metadata.')


if __name__ == '__main__':
    main()
    