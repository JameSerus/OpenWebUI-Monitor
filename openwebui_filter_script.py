"""
title: Usage Monitor
author: VariantConst & OVINC CN
git_url: https://github.com/VariantConst/OpenWebUI-Monitor.git
version: 0.3.6
requirements: httpx
license: MIT
"""

import logging
import time
from typing import Dict, Optional
from httpx import AsyncClient
from pydantic import BaseModel, Field
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRANSLATIONS = {
    "en": {
        "request_failed": "Request failed: {error_msg}",
        "insufficient_balance": "Insufficient balance: Current balance `{balance:.4f}`",
        "cost": "Cost: €{cost:.4f}",
        "balance": "Balance: €{balance:.4f}",
        "tokens": "Tokens(Context + Input + Output): {context}+{input}+{output}",
        "time_spent": "Time: {time:.2f}s",
        "tokens_per_sec": "{tokens_per_sec:.2f} T/s",
    },
    "de": {
        "request_failed": "Anfrage fehlgeschlagen: {error_msg}",
        "insufficient_balance": "Zu wenig Guthaben: Aktuelles Guthaben `{balance:.4f}`",
        "cost": "Kosten: €{cost:.4f}",
        "balance": "Guthaben: €{balance:.4f}",
        "tokens": "Tokens(Context + Input + Output): {context}+{input}+{output}",
        "time_spent": "Zeit: {time:.2f}s",
        "tokens_per_sec": "{tokens_per_sec:.2f} T/s",
    },
}


class CustomException(Exception):
    pass


class Filter:
    class Valves(BaseModel):
        api_endpoint: str = Field(
            default="", description="openwebui-monitor's base url"
        )
        api_key: str = Field(default="", description="openwebui-monitor's api key")
        owui_endpoint: str = Field(
            default="", description="openwebui's base url"
        )
        owui_api_key: str = Field(default="", description="openwebui's api key")
        chunk_size: int = Field(
            default=700,
            description="Amount of tokens per knowledge base subsection.",
        )
        top_k: int = Field(
            default=3,
            description="Amount of hits for knowledge base chunsk which are given to the model.",
        )
        priority: int = Field(default=5, description="filter priority")
        language: str = Field(default="zh", description="language (en/zh)")
        show_time_spent: bool = Field(default=True, description="show time spent")
        show_tokens_per_sec: bool = Field(
            default=True, description="show tokens per second"
        )
        show_cost: bool = Field(default=True, description="show cost")
        show_balance: bool = Field(default=True, description="show balance")
        show_tokens: bool = Field(default=True, description="show tokens")

    def __init__(self):
        self.type = "filter"
        self.name = "OpenWebUI Monitor"
        self.valves = self.Valves()
        self.outage_map: Dict[str, bool] = {}
        self.start_time: Optional[float] = None

    def get_text(self, key: str, **kwargs) -> str:
        lang = self.valves.language if self.valves.language in TRANSLATIONS else "en"
        text = TRANSLATIONS[lang].get(key, TRANSLATIONS["en"][key])
        return text.format(**kwargs) if kwargs else text

    async def request(
        self, client: AsyncClient, url: str, headers: dict, json_data: dict
    ):
        json_data = json.loads(
            json.dumps(
                json_data, default=lambda o: o.dict() if hasattr(o, "dict") else str(o)
            )
        )

        response = await client.post(url=url, headers=headers, json=json_data)
        response.raise_for_status()
        response_data = response.json()
        if not response_data.get("success"):
            logger.error(self.get_text("request_failed", error_msg=response_data))
            raise CustomException(
                self.get_text("request_failed", error_msg=response_data)
            )
        return response_data

    async def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        __user__ = __user__ or {}
        __metadata__ = __metadata__ or {}
        self.start_time = time.time()
        user_id = __user__.get("id", "default")

        client = AsyncClient()

        try:
            response_data = await self.request(
                client=client,
                url=f"{self.valves.api_endpoint}/api/v1/inlet",
                headers={"Authorization": f"Bearer {self.valves.api_key}"},
                json_data={"user": __user__, "body": body},
            )
            self.outage_map[user_id] = response_data.get("balance", 0) <= 0
            if self.outage_map[user_id]:
                logger.info(
                    self.get_text(
                        "insufficient_balance", balance=response_data.get("balance", 0)
                    )
                )
                raise CustomException(
                    self.get_text(
                        "insufficient_balance", balance=response_data.get("balance", 0)
                    )
                )
            return body

        except Exception as err:
            logger.exception(self.get_text("request_failed", error_msg=err))
            if isinstance(err, CustomException):
                raise err
            raise Exception(f"error calculating usage, {err}") from err

        finally:
            await client.aclose()

    async def outlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
    ) -> dict:
        __user__ = __user__ or {}
        __metadata__ = __metadata__ or {}
        user_id = __user__.get("id", "default")

        if self.outage_map.get(user_id, False):
            return body

        client = AsyncClient()

        try:
            model_response = await client.get(url=f"{self.valves.owui_endpoint}/api/models", headers={"Authorization": f"Bearer {self.valves.owui_api_key}"})
            model_data = model_response.json()
            target_model_id = body.get("model")
            models = model_data.get("data", [])
            model = next((m for m in models if m["id"] == target_model_id), None)

            local_chunk_size = 0
            local_top_k = 0
            local_system_prompt = ""

            if model:
                knowledge = model.get("info", {}).get("meta", {}).get("knowledge", [])
                if knowledge:
                    local_chunk_size = self.valves.chunk_size
                    local_top_k = self.valves.top_k
                system_prompt = model.get("info", {}).get("params", {}).get("system", {})
                if system_prompt:
                    local_system_prompt = system_prompt

            

            response_data = await self.request(
                client=client,
                url=f"{self.valves.api_endpoint}/api/v1/outlet",
                headers={"Authorization": f"Bearer {self.valves.api_key}"},
                json_data={
                    "user": __user__,
                    "chunk_size": local_chunk_size,
                    "top_k": local_top_k,
                    "system_prompt": local_system_prompt,
                    "body": body
                },
            )

            stats_list = []
            if self.valves.show_tokens:
                stats_list.append(
                    self.get_text(
                        "tokens",
                        context=local_chunk_size*local_top_k+response_data["systemPromptTokens"],
                        input=response_data["inputTokens"],
                        output=response_data["outputTokens"],
                    )
                )
            if self.valves.show_cost:
                stats_list.append(
                    self.get_text("cost", cost=response_data["totalCost"])
                )
            if self.valves.show_balance:
                stats_list.append(
                    self.get_text("balance", balance=response_data["newBalance"])
                )
            if self.start_time and self.valves.show_time_spent:
                elapsed = time.time() - self.start_time
                stats_list.append(self.get_text("time_spent", time=elapsed))
                if self.valves.show_tokens_per_sec:
                    tokens_per_sec = (
                        response_data["outputTokens"] / elapsed if elapsed > 0 else 0
                    )
                    stats_list.append(
                        self.get_text("tokens_per_sec", tokens_per_sec=tokens_per_sec)
                    )

            stats = " | ".join(stats_list)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": stats, "done": True}}
                )

            logger.info("usage_monitor: %s %s", user_id, stats)
            return body

        except Exception as err:
            logger.exception(self.get_text("request_failed", error_msg=err))
            raise Exception(self.get_text("request_failed", error_msg=err))
        finally:
            await client.aclose()
