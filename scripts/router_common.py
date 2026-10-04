"""
router_common.py  --  shared code for the CliniqAI router retrains WITHOUT bone_fracture
(gatekeeper v6 = Stage 1, router stage 2 v3).

Put this file in  E:\\programming\\CliniqAI\\scripts\\  (the notebooks add that folder to sys.path).
Living in a real module (not inside a notebook cell) keeps DataLoader workers picklable on Windows.

Deliberate choices (all carried over from the lessons in earlier versions):
  * 6 modalities only: brain_mri, breast_histology, kidney_ct, pneumonia, retinal, skin_lesion
  * plain, UNWEIGHTED CrossEntropyLoss (auto class-weighting caused the gatekeeper v4 regression)
  * group-aware splits (BreakHis slide id incl. lettered suffixes, pneumonia person/patient id,
    augmentation-suffix stripping) with a hard assert that no group crosses train/val/test
  * every sample carries a `source` tag so accuracy is reported per source, never as one number
  * Windows-safe file listing (os.walk + lower-cased suffix: no *.png / *.PNG double counting)
"""
import os
import re
import json
import math
import time
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
ARCH = "efficientnet_b0"

# modality (specialist name) -> folder under data/raw   (same mapping as gatekeeper v4/v5 + stage 2 v2)
MODALITY_DIRS = {
    "brain_mri": "brain_tumor",
    "breast_histology": "breast_histology_v2",   # BreakHis
    "kidney_ct": "kidney_stone",
    "pneumonia": "pneumonia",
    "retinal": "diabetic_retinopathy",
    "skin_lesion": "skin_lesion",
}
MODALITIES = sorted(MODALITY_DIRS)              # 6 classes, alphabetical
PER_MODALITY_CAP = {m: 12000 for m in MODALITIES}
PER_MODALITY_CAP["breast_histology"] = None     # keep all 7,909 BreakHis images

# folder-name tokens that must never end up in the NEGATIVE pool by accident
_MEDICAL_TOKENS = {
    "brain", "mri", "tumor", "tumour", "breast", "histology", "histopathology", "kidney", "stone",
    "ct", "pneumonia", "xray", "retinal", "retinopathy", "diabetic", "skin", "lesion", "melanoma",
    "fracture", "bone", "chest", "medical", "isic", "ham10000", "idc", "breakhis",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# file listing + group keys
# --------------------------------------------------------------------------------------
def list_images(folder):
    """Recursive, Windows-safe image listing (each file counted once)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    seen, out = set(), []
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                p = os.path.join(dp, fn)
                key = os.path.normcase(os.path.abspath(p))
                if key not in seen:
                    seen.add(key)
                    out.append(p)
    out.sort()
    return out


# BreakHis: SOB_B_TA-14-4659-40-001.png  /  SOB_M_PC-15-190EF-400-023.png  -> slide id "14-4659" / "15-190EF"
# (alphanumeric slide ids on purpose: a digits-only regex dropped >50% of images to "unknown" before)
_BREAKHIS = re.compile(r"SOB_[BM]_[A-Za-z]+-(\d+-[0-9A-Za-z]+)-\d+-\d+", re.I)
_PNEU_PERSON = re.compile(r"(person\d+)", re.I)
_PNEU_IM = re.compile(r"(?:^|[-_])IM-(\d{4})", re.I)
# offline-augmentation suffixes baked into filenames (the bone_fracture v1 leak pattern)
_AUG_SUFFIX = re.compile(r"([-_ ]?rotated\d*|[-_ ]?aug(mented)?\d*|[-_ ]?flip(ped)?\w*|[-_ ]?copy\d*)+", re.I)


def group_key(source, path):
    """Samples sharing a key are never allowed to be in different splits."""
    stem = Path(path).stem
    if source == "breast_histology":
        m = _BREAKHIS.search(stem)
        if m:
            return "bh:" + m.group(1).upper()
    if source == "pneumonia":
        m = _PNEU_PERSON.search(stem)
        if m:
            return "pn:" + m.group(1).lower()
        m = _PNEU_IM.search(stem)
        if m:
            return "pn:im" + m.group(1)
    base = _AUG_SUFFIX.sub("", stem)
    return f"{source}:{base}"


def group_split(items, key_fn, fracs=(0.8, 0.1, 0.1), seed=42):
    """items -> (train, val, test); whole groups are assigned together, deterministic."""
    groups = defaultdict(list)
    for it in items:
        groups[key_fn(it)].append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = len(items)
    t_end, v_end = fracs[0] * total, (fracs[0] + fracs[1]) * total
    out, cum = ([], [], []), 0
    for k in keys:
        idx = 0 if cum < t_end else (1 if cum < v_end else 2)
        out[idx].extend(groups[k])
        cum += len(groups[k])
    return out


def _capped(paths, cap, seed):
    if cap is None or len(paths) <= cap:
        return list(paths)
    return random.Random(seed).sample(list(paths), cap)


# --------------------------------------------------------------------------------------
# sample-list builders.  A sample = (path, label_idx, source)
# --------------------------------------------------------------------------------------
def build_modality_pools(data_raw, seed=42, verbose=True):
    pools = {}
    for m in MODALITIES:
        folder = Path(data_raw) / MODALITY_DIRS[m]
        paths = list_images(folder)
        if not paths:
            raise FileNotFoundError(f"No images for '{m}' in {folder} - fix MODALITY_DIRS in router_common.py")
        full = len(paths)
        pools[m] = _capped(paths, PER_MODALITY_CAP[m], seed)
        if verbose:
            print(f"  {m:18s} {MODALITY_DIRS[m]:22s} found {full:6d}  using {len(pools[m]):6d}")
    return pools


def build_stage2_samples(data_raw, class_names=None, seed=42):
    class_names = list(class_names or MODALITIES)
    pools = build_modality_pools(data_raw, seed)
    splits = {"train": [], "val": [], "test": []}
    for m, paths in pools.items():
        items = [(p, class_names.index(m), m) for p in paths]
        tr, va, te = group_split(items, lambda s: group_key(s[2], s[0]), seed=seed)
        splits["train"] += tr
        splits["val"] += va
        splits["test"] += te
    return splits


def discover_negative_sources(data_raw, extra_exclude=(), holdout=("fruits",)):
    data_raw = Path(data_raw)
    skip_exact = set(MODALITY_DIRS.values()) | {"bone_fracture"} | set(extra_exclude)
    found, skipped = [], []
    for d in sorted(os.listdir(data_raw)):
        p = data_raw / d
        low = d.lower()
        if not p.is_dir():
            continue
        if d in skip_exact or low.startswith(("manual_sanity", ".", "_")) or any(h in low for h in holdout):
            skipped.append(d)
            continue
        tokens = set(t for t in re.split(r"[^a-z0-9]+", low) if t)
        bad = tokens & _MEDICAL_TOKENS
        if bad:
            raise RuntimeError(
                f"data/raw/{d} looks medical (tokens {sorted(bad)}) but would be used as a NEGATIVE source.\n"
                f"Add '{d}' to EXTRA_EXCLUDE_DIRS in the notebook (or set NEGATIVE_SOURCES explicitly)."
            )
        found.append(d)
    return found, skipped


def build_gatekeeper_samples(data_raw, class_names=("non_medical", "medical"), neg_sources=None,
                             extra_exclude=(), holdout=("fruits",), max_neg_per_pos=None, seed=42):
    class_names = list(class_names)
    pos_idx, neg_idx = class_names.index("medical"), class_names.index("non_medical")
    print("Positives (6 modalities, bone_fracture NOT included):")
    pools = build_modality_pools(data_raw, seed)

    splits = {"train": [], "val": [], "test": []}
    for m, paths in pools.items():
        items = [(p, pos_idx, m) for p in paths]
        tr, va, te = group_split(items, lambda s: group_key(s[2], s[0]), seed=seed)
        splits["train"] += tr
        splits["val"] += va
        splits["test"] += te

    if neg_sources is None:
        neg_sources, skipped = discover_negative_sources(data_raw, extra_exclude, holdout)
        print(f"\nAuto-discovered negative sources ({len(neg_sources)}); skipped dirs: {skipped}")
    print("Negatives:")
    n_neg_total = 0
    for s in neg_sources:
        paths = list_images(Path(data_raw) / s)
        if not paths:
            print(f"  !! {s}: 0 images (missing / empty) - skipping")
            continue
        n_neg_total += len(paths)
        print(f"  {s:28s} {len(paths):7d}")
        items = [(p, neg_idx, s) for p in paths]
        tr, va, te = group_split(items, lambda x: group_key(x[2], x[0]), seed=seed)
        splits["train"] += tr
        splits["val"] += va
        splits["test"] += te

    if max_neg_per_pos:   # optional: thin the TRAIN negatives only (val/test keep the true distribution)
        tr = splits["train"]
        pos = [x for x in tr if x[1] == pos_idx]
        neg = [x for x in tr if x[1] == neg_idx]
        keep = int(max_neg_per_pos * len(pos))
        if len(neg) > keep:
            by_src = defaultdict(list)
            for x in neg:
                by_src[x[2]].append(x)
            rng, thinned = random.Random(seed), []
            for s, lst in by_src.items():
                k = max(1, round(keep * len(lst) / len(neg)))
                thinned += rng.sample(lst, min(k, len(lst)))
            print(f"\nMAX_NEG_PER_POS={max_neg_per_pos}: train negatives {len(neg)} -> {len(thinned)}")
            splits["train"] = pos + thinned

    for k in splits:
        random.Random(seed).shuffle(splits[k])
    info = {"neg_sources": neg_sources, "n_neg_total": n_neg_total,
            "n_pos_total": sum(len(v) for v in pools.values())}
    return splits, info


def print_split_table(splits, class_names):
    srcs = sorted({s for v in splits.values() for (_, _, s) in v})
    print(f"\n{'source':28s} {'class':12s} {'train':>8s} {'val':>8s} {'test':>8s}")
    for s in srcs:
        lab = next(l for v in splits.values() for (_, l, ss) in v if ss == s)
        c = [sum(1 for x in splits[k] if x[2] == s) for k in ("train", "val", "test")]
        print(f"{s:28s} {class_names[lab]:12s} {c[0]:8d} {c[1]:8d} {c[2]:8d}")
    tot = [len(splits[k]) for k in ("train", "val", "test")]
    print(f"{'TOTAL':28s} {'':12s} {tot[0]:8d} {tot[1]:8d} {tot[2]:8d}")
    for k in ("train", "val", "test"):
        cnt = Counter(class_names[l] for _, l, _ in splits[k])
        print(f"  {k:5s} class counts: {dict(cnt)}")


def assert_no_group_overlap(splits):
    gs = {k: {group_key(s, p) for p, _, s in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = gs[a] & gs[b]
        assert not inter, f"LEAKAGE: {len(inter)} groups shared between {a} and {b}, e.g. {sorted(inter)[:5]}"
    print("Group-overlap check passed (0 shared groups between train/val/test). Unique groups:",
          {k: len(v) for k, v in gs.items()})
    bh = {k: sum(1 for g in v if g.startswith("bh:")) for k, v in gs.items()}
    if sum(bh.values()):
        print("  BreakHis slide-ids per split:", bh)


# --------------------------------------------------------------------------------------
# transforms / dataset / loaders
# --------------------------------------------------------------------------------------
class RandomResolutionDegrade:
    """Down-scale then up-scale so the model sees low-res medical images too (50x50 IDC-patch lesson)."""
    def __init__(self, p=0.3, min_side=32, max_side=128):
        self.p, self.min_side, self.max_side = p, min_side, max_side

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        w, h = img.size
        target = random.randint(self.min_side, self.max_side)
        sc = target / max(w, h)
        if sc >= 1:
            return img
        small = img.resize((max(8, int(w * sc)), max(8, int(h * sc))), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)


def train_transform(degrade_p=0.0):
    t = []
    if degrade_p > 0:
        t.append(RandomResolutionDegrade(p=degrade_p))
    t += [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(t)


# !! must match the preprocessing in backend/app/pipeline.py (I have not seen that file) !!
def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_rgb(path):
    img = Image.open(path)
    img.draft("RGB", (448, 448))          # JPEG-only speed-up for huge fundus/skin photos; no-op otherwise
    return img.convert("RGB")


class ListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = list(samples), transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        for k in range(5):                                  # survive the odd corrupt file
            path, label, _ = self.samples[(i + k) % len(self.samples)]
            try:
                return self.transform(load_rgb(path)), label
            except Exception:
                continue
        raise RuntimeError(f"5 consecutive unreadable images starting at index {i}")


def make_loader(ds, batch_size, shuffle, num_workers=4, persistent=None):
    persistent = shuffle if persistent is None else persistent
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=shuffle)
    if num_workers > 0:
        kw.update(persistent_workers=persistent, prefetch_factor=4)
    return DataLoader(ds, **kw)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
def build_model(num_classes, pretrained=True):
    w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=w)
    m.classifier = nn.Sequential(          # wide head: keys classifier.1 / classifier.4 (matches model_loader)
        nn.Dropout(0.3), nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    return m


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_model(model, train_loader, val_loader, class_names, ckpt_dir, tag, device,
                epochs=30, lr=1e-3, weight_decay=1e-4, patience=7, resume=True):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    inprog = ckpt_dir / f"{tag}.checkpoint_inprogress.pt"
    model.to(device)
    criterion = nn.CrossEntropyLoss()        # deliberately UNWEIGHTED
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    use_amp = device.type == "cuda"
    scaler = _grad_scaler(use_amp)
    start, best, bad, hist = 0, math.inf, 0, []

    if resume and inprog.exists():
        ck = torch.load(inprog, map_location="cpu", weights_only=False)
        assert list(ck["class_names"]) == list(class_names), "in-progress checkpoint has different classes - delete it"
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, hist = ck["epoch"] + 1, ck["best_val_loss"], ck["history"]
        print(f"Resumed from {inprog.name}: continuing at epoch {start + 1}, best val_loss {best:.4f}")

    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        tl = tc = tn = 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tn += y.size(0)

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                vl += loss.item() * y.size(0)
                vc += (out.argmax(1) == y).sum().item()
                vn += y.size(0)
        tl, ta, vl, va = tl / tn, tc / tn, vl / vn, vc / vn
        sched.step(vl)
        rec = dict(epoch=epoch + 1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va,
                   lr=opt.param_groups[0]["lr"], minutes=(time.time() - t0) / 60)
        hist.append(rec)
        improved = vl < best
        print(f"epoch {epoch + 1:2d}/{epochs} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f} | "
              f"lr {rec['lr']:.1e} | {rec['minutes']:.1f} min" + ("  * best" if improved else ""))
        if improved:
            best, bad = vl, 0
            _atomic_save(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(),
                              scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                              best_val_loss=best, history=hist, class_names=list(class_names)), inprog)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping: no val_loss improvement for {patience} epochs.")
                break

    ck = torch.load(inprog, map_location="cpu", weights_only=False)   # always finish with the BEST weights
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded best weights (val_loss {ck['best_val_loss']:.4f}, epoch {ck['epoch'] + 1})")
    return hist


def save_final(model, class_names, history, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck = dict(model_state_dict=state, class_names=list(class_names), classes=list(class_names),
              num_classes=len(class_names), architecture=ARCH, input_size=IMG_SIZE,
              mean=MEAN, std=STD)
    if extra:
        ck.update(extra)
    torch.save(ck, path)
    hp = path.with_name(path.stem + ".history.json")
    hp.write_text(json.dumps(history, indent=2))
    # round-trip check: head shapes + keys must load into the standard head
    probe = build_model(len(class_names), pretrained=False)
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    print(f"Saved {path}  +  {hp.name}   (round-trip load OK, classes={list(class_names)})")


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, ds, device, batch_size=64, num_workers=4):
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=num_workers, persistent=False)
    model.to(device).eval()
    out = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        out.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def report(probs, samples, class_names, title="TEST"):
    y = np.array([s[1] for s in samples])
    pred = probs.argmax(1)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print(f"\n===== {title}: overall acc {(pred == y).mean():.4f}  (n={len(y)}) =====")
    print("Per-class recall:")
    for i, c in enumerate(class_names):
        print(f"  {c:18s} {cm[i, i] / max(1, cm[i].sum()):.4f}   (n={cm[i].sum()})")
    print("Confusion matrix (rows = true, cols = predicted):")
    w = max(len(c) for c in class_names) + 1
    print(" " * (w + 1) + " ".join(f"{c[:9]:>9s}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c:>{w}s} " + " ".join(f"{v:9d}" for v in cm[i]))
    print("Per-source accuracy (worst first):")
    per = defaultdict(lambda: [0, 0])
    for s, p in zip(samples, pred):
        per[s[2]][0] += int(p == s[1])
        per[s[2]][1] += 1
    for src, (c, t) in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {src:28s} {c / t:.4f}   (n={t})")
    return cm


def pair_confusions(cm, class_names, pairs):
    print("\nKnown confusion pairs:")
    for a, b in pairs:
        i, j = class_names.index(a), class_names.index(b)
        print(f"  {a} -> {b}: {cm[i, j]} / {cm[i].sum()}    |    {b} -> {a}: {cm[j, i]} / {cm[j].sum()}")


@torch.no_grad()
def _predict_files(model, files, device):
    tf = eval_transform()
    model.to(device).eval()
    res = []
    for f in files:
        x = tf(load_rgb(f)).unsqueeze(0).to(device)
        res.append(torch.softmax(model(x).float(), 1)[0].cpu().numpy())
    return res


def sanity_folder(model, folder, expected, class_names, device, label=None):
    """Real-world image check with an EXPECTED label (PASS/FAIL, not just printed probabilities)."""
    files = list_images(folder)
    label = label or Path(folder).name
    if not files:
        print(f"[{label}] WARNING: {folder} is missing/empty -> this check proves nothing yet")
        return None
    ok = 0
    for f, pr in zip(files, _predict_files(model, files, device)):
        pred = class_names[int(pr.argmax())]
        good = pred == expected
        ok += good
        probs = "  ".join(f"{c}={p:.3f}" for c, p in zip(class_names, pr))
        print(f"  [{'PASS' if good else 'FAIL'}] {Path(f).name[:45]:45s} -> {pred:16s} {probs}")
    print(f"[{label}] expected '{expected}': {ok}/{len(files)} passed")
    return ok, len(files)


def probe_folder(model, folder, class_names, device, n=200, seed=0, title=None):
    """Information only (no expected label): what does the model do with images it was NOT trained on?"""
    files = list_images(folder)
    title = title or Path(folder).name
    if not files:
        print(f"[{title}] folder missing/empty - skipped")
        return
    files = random.Random(seed).sample(files, min(n, len(files)))
    prs = np.stack(_predict_files(model, files, device))
    pred = prs.argmax(1)
    print(f"[{title}] {len(files)} images ->")
    for i, c in enumerate(class_names):
        m = pred == i
        if m.any():
            print(f"    {c:18s} {m.sum():4d} ({m.mean():5.1%})  mean conf {prs[m, i].mean():.3f}")
