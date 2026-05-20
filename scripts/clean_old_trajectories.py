# scripts/clean_old_trajectories.py
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import aiosqlite

DB_PATH = PROJECT_ROOT / ".config" / "spatialclaw" / "memory.db"

async def clean():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        # 查所有 analysis 类型的 memory
        async with db.execute(
            "SELECT id, content FROM memories WHERE content LIKE '%\"memory_type\":\"analysis\"%' AND deprecated=0"
        ) as cursor:
            rows = await cursor.fetchall()

        print(f"找到 {len(rows)} 条 analysis（未废弃）")

        import json
        kept = 0
        removed = 0
        for row_id, content in rows:
            try:
                data = json.loads(content)
                traj = data.get("task_trajectory", "")
                task_main = data.get("task_main", "")
                if len(traj) < 500:
                    await db.execute(
                        "UPDATE memories SET deprecated=1 WHERE id=?",
                        (row_id,)
                    )
                    removed += 1
                    print(f"  删除 id={row_id}: {task_main} (trajectory长度={len(traj)})")
                else:
                    kept += 1
                    print(f"  保留 id={row_id}: {task_main} (trajectory长度={len(traj)})")
            except Exception as e:
                print(f"  解析失败 id={row_id}: {e}")

        await db.commit()
        print(f"\n保留 {kept} 条，删除 {removed} 条")

asyncio.run(clean())