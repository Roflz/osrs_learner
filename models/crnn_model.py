# xp_labeler/models/crnn_model.py

import torch
from torchvision import transforms
from PIL import Image
from config import CRNN_MODEL_PATH

class CrnnModel:
    """
    Loads your standalone TorchScript CRNN and replicates the exact
    predict+CTC‑decode logic from the old xp_labeler script.
    """

    def __init__(self,
                 model_path: str = CRNN_MODEL_PATH,
                 device: str = None):
        # device
        self.device    = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # load your scripted CRNN
        self.ctc_model = torch.load(model_path, map_location=self.device)
        self.ctc_model.eval()

        # CTC blank index
        self.BLANK_IDX = 10

        # same preprocessing as before
        self._preprocess = transforms.Compose([
            transforms.Resize((64, 128)),            # height=64, width=128
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                   # → [0,1]
            transforms.Normalize((0.5,), (0.5,))     # → roughly [-1,+1]
        ])


    def predict_from_boxes(self,
                           pil_image: Image.Image,
                           boxes: list[tuple[float,float,float,float]]
                          ) -> list[tuple[str, float]]:
        """
        For each (x0,y0,x1,y1) in boxes:
          1) crop from pil_image
          2) run through the exact same transforms.Resize(64,128)+Grayscale+ToTensor+Normalize
          3) unsqueeze to (1,1,64,128)
          4) forward through self.ctc_model
          5) CTC collapse (remove repeats/blanks) + avg confidence*100
        Returns list of (decoded_string, avg_conf_percent).
        """
        results = []
        for (x0, y0, x1, y1) in boxes:
            # 1) crop & preprocess
            crop = pil_image.crop((x0, y0, x1, y1))
            t    = self._preprocess(crop).unsqueeze(0).to(self.device)  # (1,1,64,128)

            # 2) forward
            with torch.no_grad():
                logp = self.ctc_model(t)
                # normalize to (T, C)
                if logp.dim() == 3 and logp.shape[0] == 1:
                    probs = logp.squeeze(0).softmax(dim=1)  # (T, C)
                else:
                    probs = logp.permute(1, 0, 2)[0].softmax(dim=1)

                pred = probs.argmax(dim=1)  # (T,)

            # 3) collapse repeats + remove blanks
            seq   = []
            confs = []
            prev  = self.BLANK_IDX
            for timestep, p in enumerate(pred):
                pi = int(p.item())
                if pi != prev and pi != self.BLANK_IDX:
                    seq.append(pi)
                    confs.append(probs[timestep, pi].item())
                prev = pi

            # 4) build string & average confidence
            text     = ''.join(str(d) for d in seq)
            avg_conf = (sum(confs) / len(confs) * 100) if confs else 0.0

            results.append((text, avg_conf))

        return results


def get_crnn_predictor():
    return CrnnModel()
