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

from tpu_inference.layers.jax.quantization.int4 import Int4Config


def test_parses_base_repo_block():
    cfg = Int4Config.from_hf_quant_config({"group_size": 64, "bits": 4})
    assert cfg.group_size == 64 and cfg.bits == 4
    assert cfg.bits_for("model.layers.0.mlp.switch_mlp.gate_proj") == (4, 64)


def test_per_module_override_8bit_router():
    raw = {"group_size": 64, "bits": 4,
           "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}}
    cfg = Int4Config.from_hf_quant_config(raw)
    assert cfg.bits_for("model.layers.0.mlp.gate") == (8, 64)
    assert cfg.bits_for("model.layers.0.mlp.switch_mlp.gate_proj") == (4, 64)


def test_dispatch_returns_int4config():
    from tpu_inference.layers.jax.quantization import get_tpu_quantization_config

    class HF:
        quantization = {"group_size": 64, "bits": 4}
        quantization_config = {}

    class MC:
        hf_config = HF()
        quantization = None

    class VC:
        model_config = MC()

    out = get_tpu_quantization_config(VC())
    assert out.__class__.__name__ == "Int4Config"
