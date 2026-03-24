import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class SuperPointONNX:
    def __init__(self, model_path="superpoint.onnx", img_size=(640, 480)):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size

    def preprocess(self, img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = img[None, None, :, :]  # (1,1,H,W)
        return img

    def extract(self, img, max_kpts=1024):
        inp = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: inp})

        kpts = outputs[0][0]        # (N, 2)
        scores = outputs[1][0]      # (N,)
        desc = outputs[2][0]        # (N, D)

        if len(scores) > max_kpts:
            idx = np.argsort(scores)[-max_kpts:]
            kpts = kpts[idx]
            desc = desc[idx]

        return kpts, desc


def match_descriptors(desc0, desc1, threshold=0.7):
    sim = cosine_similarity(desc0, desc1)  # (N0, N1)

    matches = np.argmax(sim, axis=1)
    scores = np.max(sim, axis=1)

    valid = scores > threshold

    matched_pairs = [
        (i, matches[i], scores[i])
        for i in range(len(matches))
        if valid[i]
    ]

    return matched_pairs, scores


def compute_similarity(img1, img2, model_path="superpoint.onnx"):
    sp = SuperPointONNX(model_path)

    if isinstance(img1, str):
        img1 = cv2.imread(img1)
    if isinstance(img2, str):
        img2 = cv2.imread(img2)

    if img1 is None or img2 is None:
        return {"error": "Image load failed"}

    kpts0, desc0 = sp.extract(img1)
    kpts1, desc1 = sp.extract(img2)

    matches, scores = match_descriptors(desc0, desc1)

    num_good = len(matches)
    avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0

    final_score = min(1.0, (num_good / 200.0) * (avg_conf + 0.1))

    return {
        "similarity_score": round(final_score, 3),
        "num_matches": num_good,
        "kpts_img1": len(kpts0),
        "kpts_img2": len(kpts1),
    }


def visualize_matches(img1, img2, kpts0, kpts1, matches):
    img_out = cv2.hconcat([img1, img2])
    h1, w1 = img1.shape[:2]

    for i, j, score in matches:
        pt1 = tuple(map(int, kpts0[i]))
        pt2 = tuple(map(int, kpts1[j]))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))

        cv2.line(img_out, pt1, pt2_shifted, (0, 255, 0), 1)

    return img_out


# ==========================
# TEST
# ==========================
if __name__ == "__main__":
    ref = "/home/daniel/Desktop/drone/seed/seed.png"
    output_dir = Path("output_screenshots")

    if not output_dir.exists():
        print("Folder not found")
        exit()

    images = list(output_dir.glob("*.png"))

    if len(images) == 0:
        print("No images found")
        exit()

    results = []

    print("\n[INFO] Comparing images...\n")

    for img_path in images:
        result = compute_similarity(ref, str(img_path))

        if "error" in result:
            continue

        print(f"{img_path.name} → {result['similarity_score']}")

        results.append({
            "path": img_path,
            "score": result["similarity_score"]
        })

    # 🔥 sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n[TOP MATCHES]")
    for r in results[:1]:
        print(f"{r['path'].name} → {r['score']}")

    # 🔥 keep only top 3, delete rest
    keep = set([r["path"] for r in results[:1]])

    for r in results:
        if r["path"] not in keep:
            print(f"[DELETE] {r['path'].name}")
            r["path"].unlink()

    print("\n✅ Done. Only top 1 images kept.")