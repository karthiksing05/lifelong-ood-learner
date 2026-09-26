"""Baselines that need no training: served through the same websocket protocol as GuidedPolicy.

  PromptHooks     the frozen prior under ANOTHER instruction (instruction retrieval: the training task nearest the demo)
  PlaybackPolicy  the demo's own actions, open loop, indexed by the episode step (no policy at all)
  VinnPolicy      non-parametric imitation from ONE demo (VINN, Pari et al. 2021): embed the camera frames with an ImageNet
                  ResNet-18, find the demo frame nearest the current one, execute the demo's action chunk from there
"""
import numpy as np


class PromptHooks:
    """Replace the episode's instruction; the prior acts unguided."""

    def __init__(self, text):
        self.text = str(text)

    def rewrite(self, obs, meta):
        obs = dict(obs)
        obs["prompt"] = self.text
        return obs

    def __call__(self, obs, pre, meta=None):
        return None, None


def _chunk(actions, f, H=10):
    idx = np.minimum(np.arange(int(f), int(f) + H), len(actions) - 1)
    return np.asarray(actions, np.float32)[idx]


class PlaybackPolicy:
    """Open-loop replay of the demo's actions: the chunk that starts at the current episode step."""

    metadata = {}

    def __init__(self, actions, H=10):
        self.actions = np.asarray(actions, np.float32)
        self.H = H

    def infer(self, obs, *, noise=None):
        t = int(obs.get("meta/t", 0) or 0)
        return {"actions": _chunk(self.actions, t, self.H), "state": np.asarray(obs["observation/state"], np.float32)}


class VinnPolicy:
    """Nearest demo frame in ResNet-18 feature space (both cameras, cosine), then that frame's demo chunk."""

    metadata = {}

    def __init__(self, actions, images, wrist, device="cuda", H=10):
        import torch
        import torchvision
        self.torch = torch
        self.device = device
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.actions = np.asarray(actions, np.float32)
        self.H = H
        self.bank = self._embed(np.asarray(images), np.asarray(wrist))                # [T, 1024]

    def _feat(self, imgs):
        torch = self.torch
        x = torch.as_tensor(np.asarray(imgs), device=self.device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            f = self.net((x - self.mean) / self.std)
        return torch.nn.functional.normalize(f, dim=1)

    def _embed(self, images, wrist, block=64):
        parts = []
        for i in range(0, len(images), block):
            parts.append(self.torch.cat([self._feat(images[i:i + block]), self._feat(wrist[i:i + block])], 1))
        return self.torch.cat(parts)

    def infer(self, obs, *, noise=None):
        q = self._embed(np.asarray(obs["observation/image"])[None], np.asarray(obs["observation/wrist_image"])[None])
        f = int((self.bank @ q[0]).argmax())
        return {"actions": _chunk(self.actions, f, self.H), "state": np.asarray(obs["observation/state"], np.float32)}
