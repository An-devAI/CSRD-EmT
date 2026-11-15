"""Metric teacher and conditional adversarial distillation objectives."""

import torch
from torch.nn import functional as F
from .models import Reverse


def teacher_objective(model, out, y, meta, cfg, step):
    ce = F.cross_entropy(out["logits"], y, label_smoothing=0.1)
    loss = ce
    parts = {"ce": float(ce.detach()), "metric": 0.0, "domain": 0.0}
    warm = min(step / 500, 1.0)
    if cfg["lambda_s"]:
        sim = model.metric(out["pooled"], y)
        loss = loss + cfg["lambda_s"] * warm * sim
        parts["metric"] = float(sim.detach())
    if cfg["lambda_d"]:
        mapping = {s: i for i, s in enumerate(model.subjects)}
        labels = torch.tensor(
            [mapping[int(s)] for s in meta["subject"]], device=y.device
        )
        logits = model.discriminator(
            Reverse.apply(out["pooled"], cfg["lambda_d"] * warm)
        )
        domain = F.cross_entropy(logits, labels)
        loss = loss + domain
        parts["domain"] = float(domain.detach())
    return (loss, parts)


def student_objective(model, out, y, meta, cfg, step, teacher, x, bank):
    ce = F.cross_entropy(out["logits"], y, label_smoothing=0.1)
    loss = ce
    kd = ce * 0
    domain = ce * 0
    m = cfg["method"]
    warm = min(step / 500, 1.0)
    if m not in ["ce", "no_kd"]:
        with torch.no_grad():
            if m == "xded":
                q = (
                    torch.stack([out["logits"].detach()[y == c].mean(0) for c in y])
                    .div(2)
                    .softmax(-1)
                )
            else:
                t = teacher(x, return_features=True)
                q = (
                    t["logits"].div(2).softmax(-1)
                    if m == "kd_global"
                    else bank.target(t["normalized"], meta, m != "no_exclusion")
                )
        kd = (
            F.kl_div(F.log_softmax(out["logits"] / 2, dim=-1), q, reduction="batchmean")
            * 4
        )
        loss = loss + cfg["kd"] * warm * kd
    if m != "ce":
        h = model.trunk(Reverse.apply(out["pooled"], cfg["adv"] * warm))
        labels = torch.tensor(
            [model.subjects.index(s) for s in meta["subject"]], device=y.device
        )
        domain = torch.stack(
            [
                F.cross_entropy(
                    head(h if m in ["kd_global", "global", "xded"] else h[y == c]),
                    labels if m in ["kd_global", "global", "xded"] else labels[y == c],
                )
                for c, head in enumerate(model.heads)
            ]
        ).mean()
        loss = loss + domain
    return (
        loss,
        dict(
            ce=float(ce.detach()), kd=float(kd.detach()), domain=float(domain.detach())
        ),
    )
