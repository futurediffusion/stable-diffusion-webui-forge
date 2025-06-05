import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

# Create stub modules required by face_restoration_utils
for name in ["devices", "errors", "face_restoration", "shared"]:
    mod = types.ModuleType(name)
    if name == "face_restoration":
        class FaceRestoration:
            pass
        mod.FaceRestoration = FaceRestoration
    sys.modules.setdefault(f"modules.{name}", mod)

# Stub modules_forge.utils with a dummy prepare_free_memory
mf_utils = types.ModuleType("modules_forge.utils")
mf_utils.prepare_free_memory = lambda *_, **__: None
sys.modules.setdefault("modules_forge.utils", mf_utils)

from modules.face_restoration_utils import bgr_image_to_rgb_tensor, rgb_tensor_to_bgr_image


def test_bgr_to_tensor_roundtrip():
    np.random.seed(0)
    img = np.random.rand(4, 5, 3).astype(np.float32)
    tensor = bgr_image_to_rgb_tensor(img)
    assert isinstance(tensor, torch.Tensor)
    out = rgb_tensor_to_bgr_image(tensor)
    assert np.allclose(out, img, atol=1e-6)
