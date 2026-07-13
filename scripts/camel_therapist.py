"""CAMEL vLLM-backed therapist helper."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from chat_runtime import (  # noqa: E402
    DEFAULT_DATASET,
    ROOT_DIR,
    Turn,
    choose_patient,
    load_dataset,
    load_environment,
)
from therapist import join_value  # noqa: E402


SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CAMEL_VLLM_SERVER = "http://127.0.0.1:8000/v1"
DEFAULT_CAMEL_MODEL_ID = "LangAGI-Lab/camel"
DEFAULT_CACTUS_CASES = ROOT_DIR / "data" / "cactus_all_cases.json"
DEFAULT_OPENING_MESSAGE = "Hi, it's nice to meet you. What brings you to therapy today?"


def load_cactus_cases(path: Path = DEFAULT_CACTUS_CASES) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing CACTUS cases file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"CACTUS cases file is not valid JSON: {path}") from exc

    if not isinstance(data, dict) or not data:
        raise SystemExit(f"CACTUS cases must be a non-empty JSON object: {path}")

    cases: dict[str, dict[str, Any]] = {}
    for case_id, case in data.items():
        if isinstance(case, dict):
            cases[str(case_id)] = case
    if not cases:
        raise SystemExit(f"CACTUS cases file contains no case objects: {path}")
    return cases


def case_for_patient(
    patient: dict[str, Any],
    cases: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    patient_id = patient.get("id")
    if patient_id is not None and str(patient_id) in cases:
        case_id = str(patient_id)
        return case_id, cases[case_id]

    patient_name = str(patient.get("name", "")).strip().lower()
    for case_id, case in cases.items():
        client_info = case.get("intake_form", {}).get("client_info", {})
        if not isinstance(client_info, dict):
            continue
        case_name = str(client_info.get("name", "")).strip().lower()
        if patient_name and case_name == patient_name:
            return case_id, case

    raise SystemExit(
        "Could not map patient to CACTUS case. "
        f"patient_id={patient_id!r}, patient_name={patient.get('name')!r}"
    )


def cactus_to_intake_reason(session: Any, case: dict[str, Any]) -> tuple[str, str]:
    """Convert CACTUS intake_form to the same CAMEL intake/reason format as test-camel."""
    intake_form = case.get("intake_form", {}) or {}
    client_info = intake_form.get("client_info", {}) if isinstance(intake_form, dict) else {}
    if not isinstance(client_info, dict):
        client_info = {}

    intake = session.build_intake_form(
        name=str(client_info.get("name", "")),
        age=str(client_info.get("age", "")),
        gender=str(client_info.get("gender", "")),
        occupation=str(client_info.get("occupation", "")),
        education=str(client_info.get("education", "")),
        marital_status=str(client_info.get("marital_status", "")),
        family_details=str(client_info.get("family_details", "")),
    )
    reason = ""
    if isinstance(intake_form, dict):
        reason = str(intake_form.get("reason_for_seeking_counseling", "") or "")
    if not reason:
        reason = "The client seeks counseling support."
    return intake, reason


def import_camel_agents() -> tuple[type[Any], type[Any]]:
    try:
        from camel_agent import CamelCounselingSession, CounselorAgent
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "")
        if missing in {"camel_agent", "langchain_openai", "langchain_core"}:
            raise SystemExit(
                "Missing CAMEL dependency. Ensure src/camel_agent.py is present and "
                "install its dependencies, e.g. `pip install langchain-openai langchain`."
            ) from exc
        raise
    return CamelCounselingSession, CounselorAgent


class CamelTherapist:
    """Therapist role backed by the CAMEL counseling model served through vLLM."""

    def __init__(
        self,
        vllm_server: str = DEFAULT_CAMEL_VLLM_SERVER,
        model_id: str = DEFAULT_CAMEL_MODEL_ID,
        temperature: float = 0.0,
        max_tokens: int = 512,
        cactus_cases_path: Path = DEFAULT_CACTUS_CASES,
    ) -> None:
        self.vllm_server = vllm_server
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cases = load_cactus_cases(cactus_cases_path)
        self.session: Any | None = None
        self.case_id: str | None = None
        self.case: dict[str, Any] | None = None
        self.intake_form: str | None = None
        self.reason: str | None = None
        self.started = False
        self.last_response_json: dict[str, Any] | None = None

    def opening(self, patient: dict[str, Any]) -> str:
        CamelCounselingSession, _ = import_camel_agents()
        case_id, case = case_for_patient(patient, self.cases)

        session = CamelCounselingSession(
            vllm_server=self.vllm_server,
            model_id=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        intake_form, reason = cactus_to_intake_reason(session, case)
        session.intake_form = intake_form
        session.reason = reason
        session.history = [{"role": "Counselor", "message": DEFAULT_OPENING_MESSAGE}]

        self.session = session
        self.case_id = case_id
        self.case = case
        self.intake_form = intake_form
        self.reason = reason
        self.started = False
        self.last_response_json = {
            "opening": True,
            "patient_id": patient.get("id"),
            "cactus_case_id": case_id,
            "therapist_message": DEFAULT_OPENING_MESSAGE,
            "intake_form": session.intake_form,
            "reason": session.reason,
            "cbt_technique": session.cbt_technique,
            "cbt_plan": session.cbt_plan,
            "cbt_plan_source": "pending_first_client_message",
            "history_length": len(session.history),
        }
        return DEFAULT_OPENING_MESSAGE

    def reply(self, patient: dict[str, Any], conversation: list[Any]) -> str:
        if self.session is None:
            self.opening(patient)
        if not conversation or conversation[-1].speaker != "Client":
            raise SystemExit("CAMEL therapist reply requires the latest client turn.")

        assert self.session is not None
        client_text = conversation[-1].text
        if not self.started:
            if not self.intake_form or not self.reason:
                raise SystemExit("CAMEL therapist session is missing intake or reason.")
            _, CounselorAgent = import_camel_agents()
            self.session.start(
                intake_form=self.intake_form,
                reason=self.reason,
                first_client_message=client_text,
            )
            counselor = CounselorAgent(
                vllm_server=self.session.vllm_server,
                model_id=self.session.model_id,
                cbt_plan=self.session.cbt_plan or "",
                temperature=self.session.temperature,
                max_tokens=self.session.max_tokens,
            )
            response = counselor.next_utterance(
                client_information=self.session.intake_form or "",
                reason=self.session.reason or "",
                history=self.session.history,
            )
            self.session.history.append({"role": "Counselor", "message": response})
            self.started = True
            cbt_plan_source = "generated_by_camel_start_after_first_client"
        else:
            response = self.session.step(client_text)
            cbt_plan_source = "existing_generated_plan"
        if not isinstance(response, str) or not response.strip():
            raise SystemExit("CAMEL therapist returned an empty counselor response.")
        text = response.strip()
        self.last_response_json = {
            "opening": False,
            "patient_id": patient.get("id"),
            "cactus_case_id": self.case_id,
            "client_message": client_text,
            "therapist_message": text,
            "reason": self.session.reason,
            "cbt_technique": self.session.cbt_technique,
            "cbt_plan": self.session.cbt_plan,
            "cbt_plan_source": cbt_plan_source,
            "history_length": len(self.session.history),
        }
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat directly with the CAMEL therapist.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--patient-id", help="Patient scenario id, e.g. 1-1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--camel-vllm-server",
        default=os.getenv("CAMEL_VLLM_SERVER", DEFAULT_CAMEL_VLLM_SERVER),
        help="OpenAI-compatible vLLM base URL for CAMEL.",
    )
    parser.add_argument(
        "--camel-model-id",
        default=os.getenv("CAMEL_MODEL_ID", DEFAULT_CAMEL_MODEL_ID),
        help="CAMEL model id served by vLLM.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cactus-cases", type=Path, default=DEFAULT_CACTUS_CASES)
    parser.add_argument("--no-opening", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()
    rng = random.Random(args.seed)
    patient = choose_patient(load_dataset(args.dataset), args.patient_id, rng)
    therapist = CamelTherapist(
        vllm_server=args.camel_vllm_server,
        model_id=args.camel_model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cactus_cases_path=args.cactus_cases,
    )
    conversation: list[Turn] = []

    print(f"Talking to CAMEL therapist for client {join_value(patient.get('name'))}.")
    print("Type client messages. Type 'quit' or 'exit' to stop.")
    if not args.no_opening:
        opening = therapist.opening(patient)
        conversation.append(Turn("Therapist", opening))
        print(f"Therapist> {opening}")

    while True:
        user_text = input("Client> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        conversation.append(Turn("Client", user_text))
        text = therapist.reply(patient, conversation)
        conversation.append(Turn("Therapist", text))
        print(f"Therapist> {text}")


if __name__ == "__main__":
    main()
