import os
from typing import Any
from openai import OpenAI

from maseval import ModelAdapter
from maseval.benchmark.converse import DefaultAgentConverseBenchmark, ensure_data_exists, load_tasks
from maseval.interface.inference import OpenAIModelAdapter


class MyDefaultConverseBenchmark(DefaultAgentConverseBenchmark):
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        return OpenAIModelAdapter(client=client, model_id=model_id, seed=kwargs.get("seed"))


def main() -> None:
    ensure_data_exists(domain="travel_planning")
    tasks = load_tasks(domain="travel_planning", split="privacy", limit=3)

    benchmark = MyDefaultConverseBenchmark(progress_bar=True)
    results = benchmark.run(
        tasks=tasks,
        agent_data={
            "model_id": "deepseek-chat",
            "attacker_model_id": "deepseek-chat",
            "max_turns": 10,
        },
    )
    print(results)


if __name__ == "__main__":
    main()
