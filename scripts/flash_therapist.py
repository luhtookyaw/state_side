"""Flash API-backed therapist helper."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


DEFAULT_FLASH_API_URL = "http://localhost:8000"


class FlashTherapist:
    """Therapist role backed by the local flash memory API."""

    def __init__(self, api_url: str = DEFAULT_FLASH_API_URL) -> None:
        self.api_url = api_url.rstrip("/")
        self.conversation_id: str | None = None
        self.last_response_json: dict[str, Any] | None = None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(
                f"Flash therapist API request failed: {exc.code} {url}\n{error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Could not reach flash therapist API at {url}: {exc.reason}"
            ) from exc

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Flash therapist API returned non-JSON response from {url}."
            ) from exc
        if not isinstance(data, dict):
            raise SystemExit(f"Flash therapist API returned unexpected response: {data}")
        return data

    def _therapist_message(self, response: dict[str, Any]) -> str:
        message = response.get("therapist_message")
        if not isinstance(message, str) or not message.strip():
            raise SystemExit("Flash therapist API response is missing therapist_message.")
        return message.strip()

    def opening(self, patient: dict[str, Any]) -> str:
        response = self._post_json(
            "/sessions",
            {"therapy_approach": "hybrid", "memory": "flash"},
        )
        conversation_id = response.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise SystemExit("Flash therapist API response is missing conversation_id.")
        self.conversation_id = conversation_id.strip()
        self.last_response_json = response
        return self._therapist_message(response)

    def reply(self, patient: dict[str, Any], conversation: list[Any]) -> str:
        if self.conversation_id is None:
            raise SystemExit("Flash therapist session has not been created.")
        if not conversation or conversation[-1].speaker != "Client":
            raise SystemExit("Flash therapist reply requires the latest client turn.")

        response = self._post_json(
            f"/sessions/{self.conversation_id}/messages",
            {"message": conversation[-1].text},
        )
        self.last_response_json = response
        return self._therapist_message(response)
