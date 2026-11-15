"""SEED DE-LDS loading, normalization, window sampling and evaluation."""

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import os, random, json, hashlib
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class MatNormalizer:
    method = "mat_subject_session_minmax_v1"

    def __init__(self):
        self.groups = {}

    def fit_group(self, subject, session, batches):
        key = f"S{int(subject):02}_E{int(session)}"
        if key in self.groups:
            raise ValueError("Normalizer group already fitted")
        lower = upper = None
        count = 0
        for batch in batches:
            x = np.asarray(batch, dtype=np.float64)
            if x.ndim != 2 or not len(x) or (not np.isfinite(x).all()):
                raise ValueError("Expected finite N×feature input")
            lo, hi = (x.min(0), x.max(0))
            if lower is not None and lo.shape != lower.shape:
                raise ValueError("Feature width changed")
            lower = lo if lower is None else np.minimum(lower, lo)
            upper = hi if upper is None else np.maximum(upper, hi)
            count += len(x)
        if lower is None:
            raise ValueError("Empty normalization group")
        scaler = MinMaxScaler(feature_range=(-1, 1), clip=False).fit(
            np.stack([lower, upper])
        )
        self.groups[key] = dict(
            subject=int(subject),
            session=int(session),
            data_min=lower.tolist(),
            data_max=upper.tolist(),
            scale=scaler.scale_.tolist(),
            offset=scaler.min_.tolist(),
            rows=count,
        )
        return self

    def transform(self, subject, session, x):
        key = f"S{int(subject):02}_E{int(session)}"
        if key not in self.groups:
            raise ValueError(
                "Missing subject/session statistics; no fallback to source scaler"
            )
        g = self.groups[key]
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(g["scale"]) or not np.isfinite(x).all():
            raise ValueError("Invalid transform input")
        return x * np.asarray(g["scale"]) + np.asarray(g["offset"])

    def to_dict(self):
        return dict(
            method=self.method,
            feature_range=[-1, 1],
            groups=self.groups,
            target_statistics_policy="permitted; see run audit for actual access",
            uses_target_labels=False,
        )

    @classmethod
    def from_dict(cls, data):
        if data["method"] != cls.method:
            raise ValueError("Unknown normalizer protocol")
        obj = cls()
        obj.groups = data["groups"]
        return obj


class FeatureDirectory:
    """Read an existing Kaggle MAT dataset with the same virtual paths as SEED.zip."""

    prefix = "seed/ExtractedFeatures_1s/"

    def __init__(self, root):
        self.files = {}
        for p in Path(root).rglob("*.mat"):
            if p.name in self.files:
                raise ValueError("Ambiguous duplicate MAT basename")
            self.files[p.name] = p

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def namelist(self):
        return [self.prefix + n for n in self.files]

    def read(self, name):
        if not name.startswith(self.prefix):
            raise ValueError("Unexpected input path")
        return self.files[name[len(self.prefix) :]].read_bytes()


def load_de(archive, sessions=(1,), excluded_subjects=()):
    """Chronological session order, official DE-LDS, time×channel×band flattening."""
    if not sessions or not set(sessions) <= {1, 2, 3}:
        raise ValueError("Sessions must be a nonempty subset of 1,2,3")
    data, manifest = ({}, {})
    with FeatureDirectory(archive) if Path(archive).is_dir() else ZipFile(archive) as z:
        prefix = "seed/ExtractedFeatures_1s/"
        labels = (
            loadmat(BytesIO(z.read(prefix + "label.mat")))["label"].ravel().astype(int)
            + 1
        )
        if labels.tolist() != [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]:
            raise ValueError("Unexpected SEED trial labels")
        for subject in [s for s in range(1, 16) if s not in excluded_subjects]:
            members = sorted(
                (
                    n
                    for n in z.namelist()
                    if n.startswith(prefix + str(subject) + "_") and n.endswith(".mat")
                )
            )
            if len(members) != 3:
                raise ValueError("Expected three dated sessions")
            for session in sorted(set(sessions)):
                member = members[session - 1]
                raw = z.read(member)
                d = loadmat(BytesIO(raw))
                xs, ys, trials, seconds, ids = ([], [], [], [], [])
                for trial in range(1, 16):
                    a = d[f"de_LDS{trial}"]
                    if a.ndim != 3 or a.shape[0] != 62 or a.shape[2] != 5:
                        raise ValueError("Expected channel×time×5 bands")
                    x = np.ascontiguousarray(
                        a.transpose(1, 0, 2).reshape(-1, 310), dtype=np.float64
                    )
                    if not np.isfinite(x).all():
                        raise ValueError("Nonfinite DE-LDS")
                    xs.append(x)
                    ys.extend([labels[trial - 1]] * len(x))
                    trials.extend([trial] * len(x))
                    seconds.extend(range(len(x)))
                    ids.extend(
                        (
                            f"S{subject:02}_E{session}_T{trial:02}_W{w:04}"
                            for w in range(len(x))
                        )
                    )
                key = (subject, session)
                data[key] = dict(
                    x=np.concatenate(xs),
                    y=np.asarray(ys),
                    trial=np.asarray(trials),
                    second=np.asarray(seconds),
                    ids=np.asarray(ids),
                )
                if len(set(ids)) != len(ids) or set(ys) != {0, 1, 2}:
                    raise ValueError("Invalid samples")
                feature_hash = hashlib.sha256()
                for name, dtype in [
                    ("x", "<f8"),
                    ("y", "<i8"),
                    ("trial", "<i8"),
                    ("second", "<i8"),
                ]:
                    feature_hash.update(
                        np.ascontiguousarray(data[key][name], dtype=dtype).tobytes()
                    )
                feature_hash.update("\n".join(ids).encode())
                manifest[f"S{subject:02}_E{session}"] = dict(
                    member=member,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    samples=len(ids),
                    feature_sha256=feature_hash.hexdigest(),
                )
    return (data, manifest)


def normalize_features(features):
    """Only X is accepted, with one separate scaler per (subject, session)."""
    norm = MatNormalizer()
    result = {}
    for (s, e), x in sorted(features.items()):
        norm.fit_group(s, e, [x])
        result[s, e] = norm.transform(s, e, x).astype(np.float32)
    return (result, norm.to_dict())


def source_split(target):
    sources = [s for s in range(1, 16) if s != target]
    if len(sources) != 14:
        raise ValueError("Target must be 1..15")
    order = np.random.default_rng(2022 + target).permutation(sources).tolist()
    return dict(train=sorted(order[3:]), val=sorted(order[:3]), test=[target])


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(t, p)


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    return dict(
        accuracy=float(accuracy_score(y, p)),
        ba=float(balanced_accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, p, average="weighted", zero_division=0)),
        recall=(cm.diagonal() / cm.sum(1)).tolist(),
        confusion=cm.tolist(),
        samples=len(y),
    )


class Data:

    def __init__(self, raw, subjects):
        self.raw = {k: v for k, v in raw.items() if k[0] in subjects}
        self.rows = []
        self.index = {}
        self.pools = {}
        for (s, e), v in sorted(self.raw.items()):
            for t in sorted(set(v["trial"])):
                ids = np.flatnonzero(v["trial"] == t)
                for a in range(len(ids) - 35):
                    i = int(ids[a])
                    r = dict(
                        subject=s,
                        session=e,
                        trial=int(t),
                        start_sample=a,
                        y=int(v["y"][i]),
                        offset=i,
                    )
                    self.index[s, e, int(t), a] = len(self.rows)
                    self.rows.append(r)
                    self.pools.setdefault((s, e, int(r["y"])), set()).add((int(t), a))
        self.subjects = sorted(subjects)

    def batch(self, ids, device):
        rr = [self.rows[int(i)] for i in ids]
        x = np.stack(
            [
                self.raw[r["subject"], r["session"]]["x"][
                    r["offset"] : r["offset"] + 36
                ].reshape(36, 62, 5)
                for r in rr
            ]
        )
        meta = {
            k: [r[k] for r in rr]
            for k in ["subject", "session", "trial", "start_sample"]
        }
        return (
            torch.as_tensor(x, device=device),
            torch.tensor([r["y"] for r in rr], device=device),
            meta,
        )

    def sample(self, rng):
        ss = sorted(map(int, rng.choice(self.subjects, 4, replace=False)))
        keys = []
        for c in range(3):
            pool = sorted(
                set.intersection(*[self.pools[s, e, c] for s in ss for e in [1, 2, 3]])
            )
            keys.append(pool[int(rng.integers(len(pool)))])
        return [self.index[s, e, t, w] for s in ss for e in [1, 2, 3] for t, w in keys]


def evaluate(model, data, device):
    model.eval()
    ys = []
    pp = []
    ce = []
    with torch.no_grad():
        for a in range(0, len(data.rows), 128):
            x, y, _ = data.batch(range(a, min(a + 128, len(data.rows))), device)
            z = model(x)
            ys.extend(y.cpu().tolist())
            pp.extend(z.argmax(-1).cpu().tolist())
            ce.extend(
                F.cross_entropy(z, y, reduction="none", label_smoothing=0.1)
                .cpu()
                .tolist()
            )
    subjects = np.array([r["subject"] for r in data.rows])
    result = {}
    for s in data.subjects:
        ix = subjects == s
        result[str(s)] = dict(
            **metrics(np.array(ys)[ix], np.array(pp)[ix]),
            ce=float(np.mean(np.array(ce)[ix])),
        )
    return (
        dict(
            ba=float(np.mean([v["ba"] for v in result.values()])),
            ce=float(np.mean([v["ce"] for v in result.values()])),
            subjects=result,
        ),
        dict(
            y=np.array(ys),
            pred=np.array(pp),
            sample_id=np.array(
                [
                    f"S{r['subject']:02}_E{r['session']}_T{r['trial']:02}_W{r['start_sample']:04}_C36"
                    for r in data.rows
                ]
            ),
        ),
    )
