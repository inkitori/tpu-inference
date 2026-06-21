# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field


@dataclass
class Int4Config:
    group_size: int = 64
    bits: int = 4
    overrides: dict = field(default_factory=dict)  # module-name -> {"bits", "group_size"}

    @classmethod
    def from_hf_quant_config(cls, q: dict) -> "Int4Config":
        gs = int(q.get("group_size", 64))
        bits = int(q.get("bits", 4))
        overrides = {k: v for k, v in q.items()
                     if isinstance(v, dict) and "bits" in v}
        return cls(group_size=gs, bits=bits, overrides=overrides)

    def bits_for(self, module_name: str) -> tuple[int, int]:
        for k, v in self.overrides.items():
            if module_name.endswith(k) or k in module_name:
                return int(v["bits"]), int(v.get("group_size", self.group_size))
        return self.bits, self.group_size
