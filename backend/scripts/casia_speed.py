#!/usr/bin/env python3
import asyncio, sys, time
from datetime import datetime

sys.path.insert(0, '/app')
from sqlalchemy import select, func
from app.domain.models import PersonPhoto
from app.infrastructure import db as db_module

async def main():
    db_module.configure_engine()
    start_active = None
    start_t = time.time()
    prev_active = None
    prev_t = None
    print(f"{'time':<10} {'active':>9} {'added':>9} {'rate/s':>8} {'ETA to +250k':>12}")
    for _ in range(1000):
        async with db_module.AsyncSessionLocal() as db:
            cnt = await db.execute(select(func.count(PersonPhoto.photo_id)).where(PersonPhoto.status == 'active'))
            active = int(cnt.scalar_one())
        now = time.time()
        if start_active is None:
            start_active = active
        added = active - start_active
        rate = 0.0
        eta = '—'
        if prev_active is not None and now > prev_t:
            rate = (active - prev_active) / (now - prev_t)
            remaining = max(0, 250000 - added)
            if rate > 0:
                secs = remaining / rate
                eta = f"{secs/60:.1f}m" if secs < 3600 else f"{secs/3600:.1f}h"
        print(f"{datetime.now().strftime('%H:%M:%S'):<10} {active:>9,} {added:>9,} {rate:>8.1f} {eta:>12}")
        prev_active, prev_t = active, now
        await asyncio.sleep(10)

if __name__ == '__main__':
    asyncio.run(main())
