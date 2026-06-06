"""SMAT therapist that ranks anonymous MI/CBT candidate responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chat_runtime import (  # noqa: E402
    ROOT_DIR,
    Turn,
    call_model,
    create_openai_client,
    load_environment,
    load_text,
    openai_model,
)
from therapist import (  # noqa: E402
    format_history,
    format_therapist_prompt,
    join_value,
    opening_therapist_message,
)


SMAT_PROMPT_DIR = ROOT_DIR / "prompts" / "smat_therapist"
DEFAULT_RANKING_PROMPT = SMAT_PROMPT_DIR / "ranking_agent.txt"
DEFAULT_MI_AGENT_PROMPTS = {
    "reflection_agent": SMAT_PROMPT_DIR / "mi_agents" / "reflection_agent.txt",
    "affirmation_agent": SMAT_PROMPT_DIR / "mi_agents" / "affirmation_agent.txt",
    "questioning_agent": SMAT_PROMPT_DIR / "mi_agents" / "questioning_agent.txt",
    "summarization_agent": SMAT_PROMPT_DIR / "mi_agents" / "summarization_agent.txt",
}
DEFAULT_CBT_AGENT_PROMPT = SMAT_PROMPT_DIR / "cbt_agents" / "cbt_agent.txt"
DEFAULT_CBT_SELECTOR_PROMPT = (
    SMAT_PROMPT_DIR / "cbt_agents" / "cbt_technique_selector.txt"
)
CBT_TECHNIQUES = (
    "Efficiency Evaluation",
    "Pie Chart Technique",
    "Alternative Perspective",
    "Decatastrophizing",
    "Pros and Cons Analysis",
    "Evidence-Based Questioning",
    "Reality Testing",
    "Continuum Technique",
    "Changing Rules to Wishes",
    "Behavior Experiment",
    "Problem-Solving Skills Training",
    "Systematic Exposure",
)


def render_prompt(template: str, **values: str) -> str:
    """Replace named placeholders without treating example JSON braces as format fields."""
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered


def parse_json_response(raw: str, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise SystemExit(f"{context} did not return JSON: {raw}") from None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{context} returned invalid JSON: {raw}") from exc

    if not isinstance(parsed, dict):
        raise SystemExit(f"{context} returned JSON that is not an object: {raw}")
    return parsed


class SMATTherapist:
    """Generate MI/CBT candidates, rank them anonymously, and return the best one."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        temperature: float,
        ranking_prompt_path: Path = DEFAULT_RANKING_PROMPT,
        mi_agent_prompt_paths: dict[str, Path] | None = None,
        cbt_selector_prompt_path: Path = DEFAULT_CBT_SELECTOR_PROMPT,
        cbt_agent_prompt_path: Path = DEFAULT_CBT_AGENT_PROMPT,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.temperature = temperature
        self.ranking_template = load_text(ranking_prompt_path)
        self.mi_agent_templates = {
            agent_name: load_text(prompt_path)
            for agent_name, prompt_path in (
                mi_agent_prompt_paths or DEFAULT_MI_AGENT_PROMPTS
            ).items()
        }
        self.cbt_selector_template = load_text(cbt_selector_prompt_path)
        self.cbt_agent_template = load_text(cbt_agent_prompt_path)
        self.previous_cbt_techniques: list[str] = []
        self.last_response_json: dict[str, Any] | None = None

    def opening(self, patient: dict[str, Any]) -> str:
        template = self.mi_agent_templates["questioning_agent"]
        response = self.generate_mi_response(template, patient, [])
        self.last_response_json = {
            "opening": True,
            "openness_level": None,
            "candidate_responses": [
                {
                    "id": "candidate_1",
                    "response": response,
                }
            ],
            "candidate_metadata": {
                "candidate_1": {
                    "family": "mi",
                    "agent": "questioning_agent",
                }
            },
            "ranking": ["candidate_1"],
            "ranking_response": {"ranking": ["candidate_1"]},
            "selected_response_id": "candidate_1",
            "selected_metadata": {
                "family": "mi",
                "agent": "questioning_agent",
            },
        }
        return response

    def reply(
        self,
        patient: dict[str, Any],
        conversation: list[Any],
        openness_level: int,
    ) -> str:
        candidates, metadata = self.generate_candidates(
            patient,
            conversation,
            openness_level,
        )
        ranking_response = self.rank_candidates(conversation, candidates)
        ranking = self.normalize_ranking(ranking_response, candidates)
        selected_id = ranking[0]
        selected_candidate = next(
            candidate for candidate in candidates if candidate["id"] == selected_id
        )

        self.last_response_json = {
            "openness_level": openness_level,
            "candidate_responses": candidates,
            "candidate_metadata": metadata,
            "ranking": ranking,
            "ranking_response": ranking_response,
            "selected_response_id": selected_id,
            "selected_metadata": metadata.get(selected_id),
        }
        return str(selected_candidate["response"])

    def generate_candidates(
        self,
        patient: dict[str, Any],
        conversation: list[Any],
        openness_level: int,
    ) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
        raw_candidates: list[dict[str, Any]] = []

        for agent_name, template in self.mi_agent_templates.items():
            response = self.generate_mi_response(template, patient, conversation)
            raw_candidates.append(
                {
                    "family": "mi",
                    "agent": agent_name,
                    "response": response,
                }
            )

        if openness_level > 3:
            cbt_recommendation = self.select_cbt_technique(conversation)
            response = self.generate_cbt_response(conversation, cbt_recommendation)
            raw_candidates.append(
                {
                    "family": "cbt",
                    "agent": "cbt_agent",
                    "response": response,
                    "cbt_recommendation": cbt_recommendation,
                }
            )

        candidates: list[dict[str, str]] = []
        metadata: dict[str, dict[str, Any]] = {}
        for index, raw_candidate in enumerate(raw_candidates, start=1):
            candidate_id = f"candidate_{index}"
            candidates.append(
                {
                    "id": candidate_id,
                    "response": str(raw_candidate["response"]),
                }
            )
            metadata[candidate_id] = {
                key: value
                for key, value in raw_candidate.items()
                if key != "response"
            }

        return candidates, metadata

    def generate_mi_response(
        self,
        template: str,
        patient: dict[str, Any],
        conversation: list[Any],
    ) -> str:
        prompt = format_therapist_prompt(template, patient, conversation)
        return call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=220,
        )

    def select_cbt_technique(self, conversation: list[Any]) -> dict[str, Any]:
        prompt = render_prompt(
            self.cbt_selector_template,
            conversation_history=format_history(conversation),
            allowed_cbt_techniques="\n".join(f"- {name}" for name in CBT_TECHNIQUES),
            previous_cbt_techniques=(
                "\n".join(f"- {name}" for name in self.previous_cbt_techniques)
                if self.previous_cbt_techniques
                else "None."
            ),
        )
        raw = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=640,
        )
        recommendation = parse_json_response(raw, "CBT technique selector")
        technique = recommendation.get("recommended_cbt_technique")
        if isinstance(technique, str) and technique in CBT_TECHNIQUES:
            self.previous_cbt_techniques.append(technique)
        return recommendation

    def generate_cbt_response(
        self,
        conversation: list[Any],
        cbt_recommendation: dict[str, Any],
    ) -> str:
        prompt = render_prompt(
            self.cbt_agent_template,
            cbt_recommendation=json.dumps(
                cbt_recommendation,
                indent=2,
                ensure_ascii=False,
            ),
            conversation_history=format_history(conversation),
        )
        return call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=260,
        )

    def rank_candidates(
        self,
        conversation: list[Any],
        candidates: list[dict[str, str]],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            self.ranking_template,
            conversation_history=format_history(conversation, max_turns=6),
            candidate_responses=json.dumps(
                candidates,
                indent=2,
                ensure_ascii=False,
            ),
        )
        raw = call_model(
            self.openai_client,
            self.model,
            prompt,
            self.temperature,
            max_tokens=300,
        )
        return parse_json_response(raw, "Ranking agent")

    def normalize_ranking(
        self,
        ranking_response: dict[str, Any],
        candidates: list[dict[str, str]],
    ) -> list[str]:
        valid_ids = [candidate["id"] for candidate in candidates]
        valid_id_set = set(valid_ids)
        ranking_value = ranking_response.get("ranking")

        ranking: list[str] = []
        if isinstance(ranking_value, list):
            for response_id in ranking_value:
                if response_id in valid_id_set and response_id not in ranking:
                    ranking.append(str(response_id))

        for response_id in valid_ids:
            if response_id not in ranking:
                ranking.append(response_id)
        return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat directly with the SMAT therapist.")
    parser.add_argument(
        "--patient-name",
        default="Client",
        help="Client name used by the therapist. Defaults to Client.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Defaults to 0.8.",
    )
    parser.add_argument(
        "--openness-level",
        type=int,
        default=3,
        help="Openness level used for candidate filtering. Defaults to 3.",
    )
    parser.add_argument(
        "--no-opening",
        action="store_true",
        help="Do not print the therapist's default opening message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment()

    if args.openness_level < 1 or args.openness_level > 5:
        raise SystemExit("--openness-level must be between 1 and 5.")

    patient = {"name": args.patient_name}
    model = args.model or openai_model()
    openai_client = create_openai_client()
    therapist = SMATTherapist(openai_client, model, args.temperature)
    conversation: list[Turn] = []

    print(f"Talking to SMAT therapist for client {join_value(patient.get('name'))}.")
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
        text = therapist.reply(patient, conversation, args.openness_level)
        conversation.append(Turn("Therapist", text))
        print(f"Therapist> {text}")


if __name__ == "__main__":
    main()
