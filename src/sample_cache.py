"""Đọc mẫu đã làm sạch từ cache của src.prepare, chỉ khi cache khớp seed và cỡ mẫu.

Cache hợp lệ khi cạnh sample.csv có quality_report.json với cùng `seed` và
`requested_sample`. Không khớp hoặc thiếu báo cáo thì quét lại CSV gốc và in lý do.
"""
import json
from pathlib import Path
import pandas as pd
from .data import load_sample


def cache_problem(cache, sample, seed):
    """Trả về lý do không dùng được cache, hoặc None nếu dùng được."""
    cache = Path(cache)
    if not cache.exists():
        return f"không có {cache}"
    report_path = cache.with_name("quality_report.json")
    if not report_path.exists():
        return f"không có {report_path} để kiểm tra seed và cỡ mẫu"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("seed") != seed or report.get("requested_sample") != sample:
        return (f"cache có seed={report.get('seed')}, sample={report.get('requested_sample')}, "
                f"khác yêu cầu seed={seed}, sample={sample}")
    return None


def load_frame(data, cache, sample, seed):
    problem = cache_problem(cache, sample, seed) if cache else "không dùng cache"
    if problem is None:
        frame = pd.read_csv(cache, dtype={"game_id": "string"})
        print(f"Đọc mẫu từ cache {cache}: {len(frame)} dòng (seed {seed}, sample {sample})", flush=True)
        return frame
    print(f"Không dùng cache ({problem}); quét CSV gốc {data}...", flush=True)
    return load_sample(data, sample, seed)
