"""
camel_client.py

Reusable Python client for LangAGI-Lab/camel served via a vLLM OpenAI-compatible server.

Prereqs:
  pip install langchain-openai langchain
  # and run vLLM separately, e.g.:
  # vllm serve LangAGI-Lab/camel --host 0.0.0.0 --port 8000

Usage from another file:
  from camel_client import CamelCounselingSession

  sess = CamelCounselingSession(
      vllm_server="http://127.0.0.1:8000/v1",
      model_id="LangAGI-Lab/camel",
  )

  intake = sess.build_intake_form(
      name="Laura",
      age="45",
      gender="female",
      occupation="Office Job",
      education="College Graduate",
      marital_status="Single",
      family_details="Lives alone",
  )
  reason = "I feel anxious and overwhelmed at work and can't sleep."

  sess.start(intake_form=intake, reason=reason)  # creates CBT plan using initial client message
  print("CBT plan:\n", sess.cbt_plan)

  reply = sess.step(client_message="I keep thinking I'll mess everything up.")
  print("Counselor:", reply)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


RESPONSE_PROMPT = """<|start_header_id|>system<|end_header_id|>

You are playing the role of a counselor in a psychological counseling session. Your task is to use the provided client information and counseling planning to generate the next counselor utterance in the dialogue. The goal is to create a natural and engaging response that builds on the previous conversation and aligns with the counseling plan.<|eot_id|><|start_header_id|>user<|end_header_id|>

Client Information:
{client_information}

Reason for seeking counseling:
{reason_counseling}

Counseling planning:
{cbt_plan}

Counseling Dialogue:
{history}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

CBT_PLAN_PROMPT = """<|start_header_id|>system<|end_header_id|>

You are a counselor specializing in CBT techniques. Your task is to use the provided client information, and dialogue to generate an appropriate CBT technique and a detailed counseling plan.<|eot_id|><|start_header_id|>user<|end_header_id|>

Types of CBT Techniques:
Efficiency Evaluation, Pie Chart Technique, Alternative Perspective, Decatastrophizing, Pros and Cons Analysis, Evidence-Based Questioning, Reality Testing, Continuum Technique, Changing Rules to Wishes, Behavior Experiment, Problem-Solving Skills Training, Systematic Exposure

Client Information:
{client_information}

Reason for seeking counseling:
{reason_counseling}

Counseling Dialogue:
{history}

Choose an appropriate CBT technique and create a counseling plan based on that technique.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


# ---------- Core agent wrappers ----------

class _BaseAgent:
    def __init__(
        self,
        vllm_server: str,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        # vLLM OpenAI-compatible server typically ignores the key, but the SDK expects one.
        self.llm = OpenAI(
            temperature=temperature,
            openai_api_key="EMPTY",
            openai_api_base=vllm_server,
            max_tokens=max_tokens,
            model=model_id,
        )

    @staticmethod
    def _history_to_text(history: List[Dict[str, str]]) -> str:
        # expects [{"role": "Counselor"/"Client", "message": "..."}]
        return "\n".join(f"{m['role'].capitalize()}: {m['message']}" for m in history)


class CBTPlannerAgent(_BaseAgent):
    def __init__(self, vllm_server: str, model_id: str, prompt: str = CBT_PLAN_PROMPT):
        super().__init__(vllm_server=vllm_server, model_id=model_id)
        self.prompt_template = PromptTemplate(
            input_variables=["client_information", "reason_counseling", "history"],
            template=prompt,
        )

    def generate_plan(
        self,
        client_information: str,
        reason: str,
        history: List[Dict[str, str]],
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Returns: (cbt_technique, cbt_plan, raw_response)
        """
        history_text = self._history_to_text(history)
        prompt = self.prompt_template.format(
            client_information=client_information,
            reason_counseling=reason,
            history=history_text,
        )
        raw = self.llm.invoke(prompt)

        # Parsing is fragile because model outputs vary; we keep it permissive + return raw too.
        cbt_technique: Optional[str] = None
        cbt_plan: Optional[str] = None

        # Try common patterns used in their demo parsing
        try:
            cbt_technique = raw.split("Counseling")[0].replace("\n", "").strip() or None
        except Exception:
            cbt_technique = None

        try:
            # Their example: "Counseling planning:\n ... \nCBT"
            cbt_plan = raw.split("Counseling planning:\n", 1)[1].split("\nCBT", 1)[0].strip()
        except Exception:
            # Fallback: keep most of the response if we can't find delimiters
            cbt_plan = raw.strip() if raw else None

        return cbt_technique, cbt_plan, raw


class CounselorAgent(_BaseAgent):
    def __init__(
        self,
        vllm_server: str,
        model_id: str,
        cbt_plan: str,
        prompt: str = RESPONSE_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        super().__init__(
            vllm_server=vllm_server,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.cbt_plan = cbt_plan
        self.prompt_template = PromptTemplate(
            input_variables=["client_information", "reason_counseling", "cbt_plan", "history"],
            template=prompt,
        )

    def next_utterance(
        self,
        client_information: str,
        reason: str,
        history: List[Dict[str, str]],
    ) -> str:
        history_text = self._history_to_text(history)
        prompt = self.prompt_template.format(
            client_information=client_information,
            reason_counseling=reason,
            cbt_plan=self.cbt_plan,
            history=history_text,
        )

        raw = self.llm.invoke(prompt)

        # Clean up common artifacts in some servers/responses
        text = raw
        if "'message':" in text:
            text = text.split("'message':", 1)[1].split(", {", 1)[0]
        text = (
            text.split("Counselor:")[-1]
            .replace("\n", " ")
            .replace("\\", "")
            .replace('"', "")
            .strip()
        )
        return text


# ---------- High-level reusable session ----------

@dataclass
class CamelCounselingSession:
    """
    Reusable session object:
      - stores intake form, reason, CBT plan, and running history
      - provides .start() then .step(client_message) -> counselor_message
    """
    vllm_server: str
    model_id: str = "LangAGI-Lab/camel"
    max_tokens: int = 512
    temperature: float = 0.0

    intake_form: Optional[str] = None
    reason: Optional[str] = None
    cbt_technique: Optional[str] = None
    cbt_plan: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def build_intake_form(
        self,
        name: str,
        age: str,
        gender: str,
        occupation: str,
        education: str,
        marital_status: str,
        family_details: str,
    ) -> str:
        return (
            f"Name: {name}\n"
            f"Age: {age}\n"
            f"Gender: {gender}\n"
            f"Occupation: {occupation}\n"
            f"Education: {education}\n"
            f"Marital Status: {marital_status}\n"
            f"Family Details: {family_details}"
        )

    def start(self, intake_form: str, reason: str, first_client_message: str = "") -> None:
        """
        Initializes the session and generates the CBT plan after the first client message.

        If you already have an existing dialogue history, skip this and set:
          self.intake_form, self.reason, self.history then call self.ensure_plan()
        """
        self.intake_form = intake_form
        self.reason = reason

        # Initial greeting (you can change this)
        greeting = "Hi, it's nice to meet you. What brings you to therapy today?"
        self.history = [
            {"role": "Counselor", "message": greeting},
        ]

        if first_client_message:
            self.history.append({"role": "Client", "message": first_client_message})

        self.ensure_plan()

    def ensure_plan(self) -> None:
        """Create CBT plan if missing."""
        if not self.intake_form or not self.reason:
            raise ValueError("intake_form and reason must be set before planning.")

        if self.cbt_plan:
            return

        planner = CBTPlannerAgent(
            vllm_server=self.vllm_server,
            model_id=self.model_id,
        )
        technique, plan, _raw = planner.generate_plan(
            client_information=self.intake_form,
            reason=self.reason,
            history=self.history,
        )
        self.cbt_technique = technique
        self.cbt_plan = plan or ""

    def step(self, client_message: str) -> str:
        """
        Add a client message and get next counselor reply.
        Returns counselor reply (string).
        """
        if not self.intake_form or not self.reason:
            raise ValueError("Call start() or set intake_form/reason first.")

        if not self.cbt_plan:
            self.ensure_plan()

        self.history.append({"role": "Client", "message": client_message})

        counselor = CounselorAgent(
            vllm_server=self.vllm_server,
            model_id=self.model_id,
            cbt_plan=self.cbt_plan or "",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        reply = counselor.next_utterance(
            client_information=self.intake_form,
            reason=self.reason,
            history=self.history,
        )
        self.history.append({"role": "Counselor", "message": reply})
        return reply
