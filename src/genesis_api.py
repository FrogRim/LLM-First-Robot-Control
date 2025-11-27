import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


@dataclass
class GenesisClientConfig:
    base_url: str
    api_key: str


class GenesisClient:
    """Genesis AI API 간단 클라이언트."""

    def __init__(self, config: Optional[GenesisClientConfig] = None) -> None:
        load_dotenv()
        base_url = config.base_url if config else os.getenv("GENESIS_API_BASE_URL", "")
        api_key = config.api_key if config else os.getenv("GENESIS_API_KEY", "")
        if not base_url or not api_key:
            raise ValueError("Genesis API base URL 또는 API key가 설정되어 있지 않습니다.")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def generate_text(self, model_id: str, prompt: str, **gen_params: Any) -> Dict[str, Any]:
        url = f"{self.base_url}/models/{model_id}:generate"
        payload = {"prompt": prompt}
        payload.update(gen_params)
        resp = self.session.post(url, data=json.dumps(payload), timeout=300)
        resp.raise_for_status()
        return resp.json()

    def upload_dataset(self, name: str, file_path: str) -> Dict[str, Any]:
        url = f"{self.base_url}/datasets"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/jsonl")}
            data = {"name": name}
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != "content-type"}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=600)
        resp.raise_for_status()
        return resp.json()

    def submit_train_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/jobs/train"
        resp = self.session.post(url, data=json.dumps(config), timeout=300)
        resp.raise_for_status()
        return resp.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/jobs/{job_id}"
        resp = self.session.get(url, timeout=120)
        resp.raise_for_status()
        return resp.json()
