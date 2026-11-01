"""Small compatibility/performance patch for the pinned Toolkit loader.

Its module collections are membership sets in practice. Linear list membership
causes millions of repeated comparisons while attaching a 22B model's layers.
Keep iteration order while testing object identity in constant time.
"""
from pathlib import Path

MARKER = "# ADAM_IDENTITY_MEMBERSHIP_V1"
IDENTITY_LIST = '''# ADAM_IDENTITY_MEMBERSHIP_V1
class _IdentityList(list):
    def __init__(self, values=()):
        super().__init__()
        self._ids = set()
        self.extend(values)

    def __contains__(self, value):
        return id(value) in self._ids

    def append(self, value):
        if id(value) not in self._ids:
            self._ids.add(id(value))
            super().append(value)

    def extend(self, values):
        for value in values:
            self.append(value)


'''


def apply():
    path = Path(__file__).resolve().parents[2] / "runtimes/ltx25/upstream/toolkit/memory_management/manager.py"
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return
    changes = {
        "class MemoryManager:": IDENTITY_LIST + "class MemoryManager:",
        "self.unmanaged_modules: list[torch.nn.Module] = []": "self.unmanaged_modules: list[torch.nn.Module] = _IdentityList()",
        "modules_processed = [x for x in ignore_modules]": "modules_processed = _IdentityList(ignore_modules)",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError("Unexpected Toolkit memory manager source; refusing to apply an incompatible patch.")
        source = source.replace(before, after)
    temporary = path.with_suffix(".adam-tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(path)
    print("Applied ADAM identity-membership optimization to the pinned memory manager.")


def patch_lora():
    path = Path(__file__).resolve().parents[2] / "runtimes/ltx25/upstream/toolkit/lora_special.py"
    source = path.read_text(encoding="utf-8")
    marker = "# ADAM_SKIP_UNUSED_FULL_WEIGHT_PROBE_V1"
    if marker in source:
        return
    before = "is_leaf_with_weight = (\n                            len(list(child_module.children())) == 0"
    after = "is_leaf_with_weight = (\n                            (all_layers or bool(self.full_if_contains))  # ADAM_SKIP_UNUSED_FULL_WEIGHT_PROBE_V1\n                            and len(list(child_module.children())) == 0"
    if source.count(before) != 1:
        raise RuntimeError("Unexpected LoRA source; refusing to patch incompatible full-weight detection.")
    temporary = path.with_suffix(".adam-tmp")
    temporary.write_text(source.replace(before, after), encoding="utf-8")
    temporary.replace(path)
    print("Patched unused full-weight inspection to avoid dequantizing the frozen base during LoRA setup.")


if __name__ == "__main__":
    apply()
    patch_lora()
