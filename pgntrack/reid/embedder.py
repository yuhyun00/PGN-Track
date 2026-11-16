# Adapted from McByte++ (Apache-2.0): OSNet re-ID features with McByte++'s crop preprocessing (GTA-Link
# transforms).  The OSNet network is torchreid's osnet_x1_0 (deep-person-reid, MIT), an external dependency.
"""OSNet appearance embeddings of boxes, L2-normalised, with McByte++'s exact crop preprocessing."""
import numpy as np

INPUT_SIZE = (256, 128)                       # height, width of the network input
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def crop_box(frame_bgr: np.ndarray, tlwh):
    """Integer-clipped RGB crop of a (x, y, w, h) box as McByte++ cuts it; None if the crop is empty."""
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = tlwh
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(w_img, x + w)), int(min(h_img, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2, :]
    if crop.size == 0:
        return None
    return crop[:, :, ::-1]                   # BGR -> RGB (as cv2.cvtColor)


def load_osnet(weights: str, device: str):
    """osnet_x1_0 in eval mode with the matching tensors of a torchreid checkpoint (classifier discarded)."""
    import torch
    from torchreid.models.osnet import osnet_x1_0
    model = osnet_x1_0(num_classes=1, pretrained=False)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)     # trusted torchreid checkpoint
    state = ckpt.get('state_dict', ckpt)
    own = model.state_dict()
    for k, v in state.items():
        k = k[7:] if k.startswith('module.') else k
        if k in own and own[k].size() == v.size():
            own[k] = v
    model.load_state_dict(own)
    return model.eval().to(device)


class OSNetEmbedder:
    """embed(frame_bgr, boxes_tlwh) -> (N, D) float32 L2-normalised rows; a box with an empty crop gets a zero row.

    frame_bgr is the image as decoded by cv2.imread.  Crops follow McByte++: integer clip to the image, RGB, PIL
    bilinear resize to 256 x 128, ImageNet normalisation, osnet_x1_0 (eval), L2 normalisation of each row.
    batch_size 1 runs one crop per forward pass, as McByte++; larger batches are faster but may change the last
    float bits.  device 'cuda' falls back to the CPU when CUDA is unavailable (as McByte++).
    """

    dim = 512
    _shared = {}                                  # one model per process and (weights, device, batch_size)

    @classmethod
    def from_config(cls, cfg: dict, device: str = None, batch_size: int = 1) -> 'OSNetEmbedder':
        """Shared embedder of the `reid` config section (or of a full config): weights under $WEIGHTS_ROOT."""
        from ..data import resolve
        sec = cfg.get('reid', cfg) if isinstance(cfg.get('reid'), dict) else cfg
        key = (resolve(sec['weights'], 'WEIGHTS_ROOT'), device or sec.get('device', 'cuda'), int(batch_size))
        if key not in cls._shared:
            cls._shared[key] = cls(*key)
        return cls._shared[key]

    def __init__(self, weights: str, device: str = 'cuda', batch_size: int = 1):
        import torch
        import torchvision.transforms as T
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.model = load_osnet(weights, device)
        self.transform = T.Compose([T.Resize(list(INPUT_SIZE)), T.ToTensor(),
                                    T.Normalize(mean=list(PIXEL_MEAN), std=list(PIXEL_STD))])

    def embed(self, frame_bgr: np.ndarray, boxes_tlwh) -> np.ndarray:
        """(N, 4) boxes (x, y, w, h) in pixels -> (N, 512) float32 L2-normalised embeddings (zero row: no crop).

        boxes_tlwh may also be a list of per-box arrays; each keeps its dtype for the crop arithmetic (McByte++
        crops boxes of tracks born in the first frame in float32).
        """
        import torch
        from PIL import Image
        boxes = list(boxes_tlwh) if isinstance(boxes_tlwh, (list, tuple)) else np.asarray(boxes_tlwh).reshape(-1, 4)
        out = np.zeros((len(boxes), self.dim), dtype=np.float32)
        idx, tensors = [], []
        for i, b in enumerate(boxes):
            crop = crop_box(frame_bgr, b)
            if crop is not None:
                idx.append(i)
                tensors.append(self.transform(Image.fromarray(np.ascontiguousarray(crop))))
        for s in range(0, len(idx), self.batch_size):
            with torch.no_grad():
                feats = self.model(torch.stack(tensors[s:s + self.batch_size]).to(self.device))
            for i, f in zip(idx[s:s + self.batch_size], feats.cpu().numpy()):
                n = np.linalg.norm(f)                 # per row, as McByte++ (float32)
                if n != 0:
                    out[i] = f / n
        return out

    __call__ = embed
