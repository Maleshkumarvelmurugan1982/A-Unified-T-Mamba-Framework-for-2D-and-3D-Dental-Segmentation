"""Web demo (no extra installs): run on the college GPU VM, open the page from your laptop browser,
upload an X-ray, see input | mask | overlay.
  python demo_web.py --weight ./runs/<run>/checkpoints/best_TMamba2D.pth --port 8000
Laptop (same college network):  http://172.16.17.119:8000      (or: ssh -L 8000:localhost:8000 user@172.16.17.119 then http://localhost:8000)"""
import os, sys, io, base64, argparse, cgi, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P = {"seed": 1777777, "cuda": True, "benchmark": False, "deterministic": True, "resize_shape": (640, 1280),
     "normalize_means": (0.50297405, 0.54711632, 0.71049083), "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
     "model_name": "TMamba2D", "dataset_name": "Tooth2D-X-Ray-6k", "dimension": "2d", "in_channels": 3, "classes": 2,
     "high_frequency": 0.9, "low_frequency": 0.1, "scaling_version": "SMALL", "multi_gpu": False, "local_rank": 0,
     "pretrain": None, "save_dir": None}
M = {}

def b64(im):
    b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()

def segment(img):
    w, h = img.size
    x = M["tf"](img).unsqueeze(0).to(M["dev"])
    t0 = time.time()
    with torch.no_grad():
        pred = torch.argmax(M["model"](x), dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    dt = time.time() - t0
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
    m = np.array(mask) > 0
    ov = np.array(img).astype(np.float32); ov[m] = 0.55 * ov[m] + 0.45 * np.array([255, 0, 0], np.float32)
    return mask, Image.fromarray(ov.astype(np.uint8)), dt

PAGE = """<html><body style="font-family:sans-serif;background:#111;color:#eee;text-align:center">
<h2>T-Mamba 2D Dental Segmentation - Live Demo</h2>
<form method=post enctype=multipart/form-data><input type=file name=f accept=image/* required>
<button type=submit>Segment</button></form>%s</body></html>"""

class H(BaseHTTPRequestHandler):
    def _send(self, body):
        d = (PAGE % body).encode(); self.send_response(200); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self): self._send("")
    def do_POST(self):
        try:
            fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})
            img = Image.open(io.BytesIO(fs["f"].file.read())).convert("RGB")
            mask, ov, dt = segment(img)
            cell = lambda t, im: '<td><div>%s</div><img style="max-width:32vw" src="data:image/png;base64,%s"></td>' % (t, b64(im))
            self._send("<p>time: %.2fs</p><table align=center><tr>%s%s%s</tr></table>" % (dt, cell("Input", img), cell("Predicted mask", mask.convert("RGB")), cell("Overlay", ov)))
        except Exception as e:
            self._send("<p style='color:red'>error: %s</p>" % e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weight", required=True); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--gpu", default="0")
    a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from lib import utils, models
    from lib.utils.checkpoint import load_pretrained_flexible
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); P["device"] = dev
    utils.reproducibility(P["seed"], P["deterministic"], P["benchmark"])
    model = models.get_model(P); load_pretrained_flexible(model, a.weight); model = model.to(dev).eval()
    M.update(model=model, dev=dev, tf=transforms.Compose([transforms.Resize(P["resize_shape"]), transforms.ToTensor(),
             transforms.Normalize(mean=P["normalize_means"], std=P["normalize_stds"])]))
    print("ready: http://0.0.0.0:%d" % a.port); HTTPServer(("0.0.0.0", a.port), H).serve_forever()
