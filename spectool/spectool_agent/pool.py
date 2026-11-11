"""
Draft 模型的共享池（线程安全）。

DraftPool 是 Draft 模型和 Main 模型之间的通信桥梁。
Draft 模型在后台线程中运行，将其每一步的中间产出（PoolEntry）
写入 DraftPool；Main 模型（主线程）从池中读取新条目并进行审阅。

所有读写操作都通过线程锁保护，确保并发安全。
"""
import threading

from .types import PoolEntry


class DraftPool:
    """线程安全的 Draft 产出共享池

    Draft 模型在后台不断生成中间结果并存入池中；
    Main 模型轮询检查新条目，一旦有答案或工具调用结果便进行审阅。
    """

    def __init__(self):
        self._lock = threading.Lock()       # 线程锁保护所有访问
        self._entries: list[PoolEntry] = [] # 所有池条目
        self._answer: str | None = None     # 已提交的第一个答案（用于快速检索）
        self._answer_event = threading.Event()

    def add(self, entry: PoolEntry) -> None:
        """添加一条 Draft 产出到池中。
        如果该条目包含答案且池中尚无答案，则记录为池答案。"""
        with self._lock:
            self._entries.append(entry)
            if entry.answer is not None and self._answer is None:
                self._answer = entry.answer
                self._answer_event.set()

    def get_all(self) -> list[PoolEntry]:
        """获取池中所有条目（返回副本以确保线程安全）"""
        with self._lock:
            return list(self._entries)

    def get_new_since(self, index: int) -> list[PoolEntry]:
        """获取指定索引之后的新条目"""
        with self._lock:
            return list(self._entries[index:])

    def snapshot_since(self, index: int):
        """Return entries and cursor atomically so concurrent adds are not lost."""
        with self._lock:
            return list(self._entries[index:]), len(self._entries)

    def position_after(self, entry: PoolEntry) -> int:
        with self._lock:
            return next(i + 1 for i, item in enumerate(self._entries) if item is entry)

    def commit_if_fresh(self, answer, seen_count, commit):
        """Serialize publication with Draft events; callback must not reenter pool."""
        with self._lock:
            if self._answer is not None and self._answer.strip() != answer.strip():
                return "draft_answer_conflict"
            if any(e.answer is None for e in self._entries[seen_count:]):
                return "stale_evidence"
            return "accepted" if commit() else "already_answered"

    def mark_quality(self, entry: PoolEntry, passed: bool) -> None:
        """Record the QJ verdict before an entry becomes fallback context."""
        with self._lock:
            entry.quality_passed = passed

    def has_answer(self) -> bool:
        """池中是否已有 draft 提交的答案"""
        with self._lock:
            return self._answer is not None

    def get_answer(self) -> str | None:
        """获取 draft 提交的第一个答案"""
        with self._lock:
            return self._answer

    def wait_for_answer(self, timeout: float | None = None) -> bool:
        """等待 Draft 提交最终答案；用于抢占仍在运行的 QJ。"""
        return self._answer_event.wait(timeout)

    def reset(self) -> None:
        """清空池（用于复用）"""
        with self._lock:
            self._entries.clear()
            self._answer = None
            self._answer_event.clear()

    @property
    def entry_count(self) -> int:
        """池中条目总数"""
        with self._lock:
            return len(self._entries)
