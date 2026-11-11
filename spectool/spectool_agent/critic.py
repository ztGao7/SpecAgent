import math
from typing import Optional

from PIL import Image
from openai import OpenAI

from .model import pil_to_data_url
from .prompts import CRITIC_PROMPT
from .types import ToolCall


class SemanticCritic:

    def __init__(self, api_base: str, model_name: str, api_key: str = "EMPTY"):
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model_name = model_name

    def verify(
        self,
        draft_call: Optional[ToolCall],
        main_call: Optional[ToolCall],
        image: Image.Image,
        threshold: float = 0.95,
    ) -> tuple[bool, float]:
        if draft_call is None and main_call is None:
            return True, 1.0
        if draft_call is None or main_call is None:
            return False, 0.0

        if (draft_call.tool_name == main_call.tool_name
                and draft_call.params == main_call.params):
            return True, 1.0

        prompt = CRITIC_PROMPT.format(
            tool_a=f"{draft_call.tool_name} with params: {draft_call.params}",
            tool_b=f"{main_call.tool_name} with params: {main_call.params}",
        )

        img_url = pil_to_data_url(image)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": prompt},
        ]}]

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
            )

            top = response.choices[0].logprobs.content[0].top_logprobs
            lp_dict = {item.token: item.logprob for item in top}

            lp_pos = lp_dict.get("positive", -100.0)
            lp_neg = lp_dict.get("negative", -100.0)

            score = math.exp(lp_pos) / (math.exp(lp_pos) + math.exp(lp_neg))
            return score >= threshold, score
        except Exception:
            return False, 0.0
