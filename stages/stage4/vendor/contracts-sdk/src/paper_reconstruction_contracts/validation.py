"""Offline JSON Schema validation; no network references or Stage 3 imports."""
from functools import lru_cache
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from .common import ContractError


@lru_cache(maxsize=3)
def validator(name):
    schema = json.loads((Path(__file__).parent/'schemas'/(name+'.schema.json')).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_schema(value, name):
    reject_closed_truth(value)
    errors = sorted(validator(name).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        first = errors[0]
        raise ContractError('Invalid '+name+' contract at '+str(list(first.path))+': '+first.message)


def reject_closed_truth(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'ground_truth', 'ground_truth_transform', 'canonical_u', 'seam_target', 'labels',
                       'reference_anchors', 'reference_transform', 'teacher_transform', 'closed_evaluation', 'evaluation'}:
                raise ContractError('Closed evaluation/training content is forbidden: '+key)
            if key in {'validation_truth_opened', 'protected_truth_opened', 'raw_truth_arrays_loaded', 'reference_anchors_used',
                       'teacher_pose_fallback_used'} and child is not False:
                raise ContractError('Closed truth policy flag is not false: '+key)
            reject_closed_truth(child)
    elif isinstance(value, list):
        for child in value: reject_closed_truth(child)
