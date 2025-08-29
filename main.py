import os
import cv2
import torch
import argparse
import shutil
import tempfile
import numpy as np
from PIL import Image
from u2net import U2NET
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# U^2-Net loader
# ----------------------------
def load_u2net(model_path="u2net.pth"):
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net


# ----------------------------
# Preprocess
# ----------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0), image


# ----------------------------
# Segment with U^2-Net
# ----------------------------
def segment_leaf_u2net(img_path, model):
    tensor_img, orig_image = preprocess_image(img_path)
    if torch.cuda.is_available():
        tensor_img = tensor_img.cuda()

    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(tensor_img)
        pred = d1[:, 0, :, :]
        # robust normalization
        pred = pred - pred.min()
        den = pred.max()
        if den > 0:
            pred = pred / den
        else:
            pred = torch.zeros_like(pred)

    mask = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask = cv2.resize(mask, orig_image.size)  # (w,h)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    image = np.array(orig_image)  # RGB
    segmented = cv2.bitwise_and(image, image, mask=mask)  # RGB
    return segmented, mask


# ----------------------------
# Green refinement (RGB->HSV)
# ----------------------------
def refine_mask_with_color(image_rgb, mask_bin):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([22, 30, 30])
    upper_green = np.array([88, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    refined_mask = cv2.bitwise_and(mask_bin, mask_bin, mask=green_mask)
    return refined_mask


# ----------------------------
# Metrics (expects BGR input)
# ----------------------------
def compute_leaf_metrics(segmented_bgr, mask_bin):
    mask = (mask_bin > 0).astype(np.uint8)
    total_leaf = int(np.count_nonzero(mask))
    if total_leaf == 0:
        # Avoid divide-by-zero; return "worst" neutral-ish defaults
        return {"HGC%": 0.0, "YB%": 0.0, "DMG%": 0.0, "HCI": 0.0, "FM%": 100.0}

    leaf_pixels = cv2.bitwise_and(segmented_bgr, segmented_bgr, mask=mask)
    hsv = cv2.cvtColor(leaf_pixels, cv2.COLOR_BGR2HSV)

    # Healthy Green Coverage
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    hgc = np.count_nonzero(green_mask) / total_leaf * 100.0

    # Yellow + Brown
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([30, 255, 100])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    yb_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    yb = np.count_nonzero(yb_mask) / total_leaf * 100.0

    # Damage % via convex hull deficit
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = sum([cv2.contourArea(c) for c in contours])
    hull = cv2.convexHull(np.vstack(contours))
    hull_area = cv2.contourArea(hull)
    dmg = (hull_area - leaf_area) / hull_area * 100
    hci = leaf_area / hull_area

    # Foreign matter %
    foreign_mask = mask.copy()
    foreign_mask[green_mask > 0] = 0
    foreign_mask[yb_mask > 0] = 0
    fm = np.count_nonzero(foreign_mask) / total_leaf * 100.0

    return {
        "HGC%": round(hgc, 2),
        "YB%": round(yb, 2),
        "DMG%": round(dmg, 2),
        "HCI": round(hci, 2),
        "FM%": round(fm, 2),
    }


# ----------------------------
# Grade (keep your current logic)
# ----------------------------
def metric_to_grade(metrics):
    HGC = metrics["HGC%"]
    YB  = metrics["YB%"]
    DMG = metrics["DMG%"]
    HCI = metrics["HCI"]
    FM  = metrics["FM%"]

    # Healthy Green Coverage (HGC%)
    if HGC >= 95: g_HGC = 0
    elif HGC >= 90: g_HGC = 1
    elif HGC >= 80: g_HGC = 2
    elif HGC >= 60: g_HGC = 3
    else: g_HGC = 4

    # Yellow/Brown Coverage (YB%)
    if YB <= 1: g_YB = 0
    elif YB <= 3: g_YB = 1
    elif YB <= 7: g_YB = 2
    elif YB <= 15: g_YB = 3
    else: g_YB = 4

    # Damage %
    if DMG <= 0.5: g_DMG = 0
    elif DMG <= 2: g_DMG = 1
    elif DMG <= 5: g_DMG = 2
    elif DMG <= 10: g_DMG = 3
    else: g_DMG = 4

    # Hull Convexity Index
    if HCI >= 0.97: g_HCI = 0
    elif HCI >= 0.95: g_HCI = 1
    elif HCI >= 0.92: g_HCI = 2
    elif HCI >= 0.88: g_HCI = 3
    else: g_HCI = 4

    # Foreign Matter %
    if FM <= 0.2: g_FM = 0
    elif FM <= 0.5: g_FM = 1
    elif FM <= 1.0: g_FM = 2
    elif FM <= 2.0: g_FM = 3
    else: g_FM = 4

    # NOTE: Keeping your current rule (min). If "worst case" is desired, use max().
    final_grade = min(g_HGC, g_YB, g_DMG, g_HCI, g_FM)
    return final_grade


# ----------------------------
# Stage A: segment everything to a temp dir (or provided)
# ----------------------------
def segment_to_dir(model, image_path=None, folder_path=None, tmp_out_dir=None):
    if tmp_out_dir is None:
        tmp_out_dir = tempfile.mkdtemp(prefix="segmented_")

    paths = []
    if image_path:
        fnames = [os.path.basename(image_path)]
        src_dir = os.path.dirname(image_path) or "."
    else:
        fnames = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        src_dir = folder_path

    os.makedirs(tmp_out_dir, exist_ok=True)

    for fname in fnames:
        img_path = os.path.join(src_dir, fname)
        segmented, mask = segment_leaf_u2net(img_path, model)

        # Only green filter
        refined_mask = refine_mask_with_color(segmented, mask)
        final_segmented = cv2.bitwise_and(segmented, segmented, mask=refined_mask)  # RGB

        save_path = os.path.join(tmp_out_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(final_segmented, cv2.COLOR_RGB2BGR))  # write BGR
        # print(f"Processed: {fname}")
        paths.append((img_path, save_path))  # (original, segmented)

    return tmp_out_dir, paths


# ----------------------------
# Stage B: load segmented from disk & compute metrics
# ----------------------------
def grade_from_segmented(segmented_path):
    seg_img = cv2.imread(segmented_path)  # BGR
    if seg_img is None:
        raise RuntimeError(f"Failed to read segmented image: {segmented_path}")
    gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    metrics = compute_leaf_metrics(seg_img, mask)
    grade = metric_to_grade(metrics)
    return grade, metrics


def main():
    parser = argparse.ArgumentParser(description="Two-stage leaf grading (segment -> load segmented -> metrics)")
    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument("--image", type=str, help="Path to a single image")
    me_group.add_argument("--folder", type=str, help="Path to a folder of images")

    parser.add_argument("--model", type=str, default="u2net.pth", help="Path to U^2-Net weights")
    parser.add_argument("--save", type=str, default=None,
                        help="If provided, save segmented outputs named with grade into this folder")

    args = parser.parse_args()

    model = load_u2net(args.model)

    # Stage A: segment to tmp_dir (auto-created)
    if args.image:
        tmp_dir, pairs = segment_to_dir(model, image_path=args.image)
    else:
        tmp_dir, pairs = segment_to_dir(model, folder_path=args.folder)

    # Stage B: compute grades reading from segmented files
    results = []  # (orig_path, segmented_path, grade, metrics)
    for orig_p, seg_p in pairs:
        grade, metrics = grade_from_segmented(seg_p)
        results.append((orig_p, seg_p, grade, metrics))
        print(f"{os.path.basename(orig_p)} => Grade: {grade}, Metrics: {metrics}")

    # Optional export with grade in filename
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for orig_p, seg_p, grade, _ in results:
            base = os.path.basename(orig_p)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(args.save, f"{name}_grade{grade}{ext}")
            seg_img = cv2.imread(seg_p)  # already BGR on disk
            if seg_img is None:
                print(f"Warning: could not read {seg_p} for saving.")
                continue
            cv2.imwrite(out_path, seg_img)

    # Always clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
