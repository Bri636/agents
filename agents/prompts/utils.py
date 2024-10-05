from __future__ import annotations
import json

""" Parsing util functions for prompts """

def strip_json(string: str) -> str:
    return string.strip('```json').strip('```').strip()

def load_json(string: str) -> str:
    return json.loads(string)

PARSING_STRATEGIES = {
    'dict': load_json,
    'json': strip_json
}