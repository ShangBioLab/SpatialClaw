# scripts/test_insight_extraction.py
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from spatialclaw.memory.layered_store import LayeredMemoryStore
from spatialclaw.memory.insight_extractor import InsightExtractor

def make_llm():
    from openai import OpenAI
    api_key = (
        os.getenv("DEEPSEEK_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("LLM_API_KEY") or ""
    )
    base_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
    model = os.getenv("SPATIALCLAW_MODEL") or "deepseek-chat"
    client = OpenAI(api_key=api_key, base_url=base_url)

    def _llm(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        result = resp.choices[0].message.content or ""
        print(f"  [LLM] {result[:120]}")
        return result

    return _llm

async def test():
    store = LayeredMemoryStore()
    await store.initialize()

    session_id = "cli:cli_user:__interactive__"

    analyses = await store.get_memories(session_id, "analysis", limit=500)
    print(f"当前 analysis 条数: {len(analyses)}")
    for a in analyses[:3]:
        print(f"  - {a.task_main}")
        print(f"    trajectory 前150字: {a.task_trajectory[:150]!r}")

    extractor = InsightExtractor(store=store, llm_callable=make_llm())

    print("\n触发 refine_insights (num_points=3)...")
    await extractor.refine_insights(session_id, num_points=3)

    insights = await store.get_memories(session_id, "insight", limit=50)
    print(f"\n生成的 insight ({len(insights)} 条):")
    for ins in sorted(insights, key=lambda x: x.score, reverse=True):
        if ins.rule:
            print(f"  [{ins.score:.1f}] {ins.rule}")
        else:
            print(f"  [{ins.score:.1f}] (rule空) {ins.biological_label[:60]}")

    await store.close()

asyncio.run(test())
