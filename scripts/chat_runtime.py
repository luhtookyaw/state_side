"""Shared runtime helpers for interactive role chats."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "Patient_Psi_CM_Dataset.json"


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing dataset file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Dataset is not valid JSON: {path}") from exc

    if not isinstance(data, list) or not data:
        raise SystemExit(f"Dataset must contain a non-empty JSON list: {path}")
    return data


def choose_patient(
    dataset: list[dict[str, Any]], patient_id: str | None, rng: random.Random
) -> dict[str, Any]:
    if patient_id:
        for patient in dataset:
            if str(patient.get("id")) == patient_id:
                return patient
        raise SystemExit(f"No patient scenario found with id {patient_id!r}.")
    return rng.choice(dataset)


def uses_max_completion_tokens(model: str) -> bool:
    normalized_model = model.lower()
    return normalized_model.startswith(("gpt-5", "o1", "o3", "o4"))


def unsupported_parameter_error(exc: Exception, parameter: str) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("param") == parameter:
            return True

    message = str(exc).lower()
    return "unsupported parameter" in message and parameter.lower() in message


def call_model(client: Any, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    token_parameter = (
        "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"
    )
    request[token_parameter] = max_tokens

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        if token_parameter != "max_tokens" or not unsupported_parameter_error(
            exc, "max_tokens"
        ):
            raise
        request.pop("max_tokens")
        request["max_completion_tokens"] = max_tokens
        response = client.chat.completions.create(**request)

    text = response.choices[0].message.content
    return (text or "").strip()


def openai_model(default: str = "gpt-4o-mini") -> str:
    return os.getenv("OPENAI_MODEL", default)


def create_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install dependencies with "
            "`pip install -r requirements.txt` or activate the project virtualenv."
        ) from exc
    return OpenAI()
