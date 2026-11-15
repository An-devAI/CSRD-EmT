"""Subject-balanced prototype bank and excluded-reference soft targets."""

import torch
from torch.nn import functional as F


class Bank:

    def __init__(self, centers, subjects):
        self.centers = centers
        self.subjects = list(subjects)

    def prototypes(self, trial, exclude_trial=True):
        c = self.centers
        labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        return F.normalize(
            torch.stack(
                [
                    c[
                        :,
                        :,
                        [
                            t
                            for t, y in enumerate(labels)
                            if y == cl and (not exclude_trial or t + 1 != trial)
                        ],
                    ].mean((1, 2))
                    for cl in range(3)
                ],
                1,
            ),
            dim=-1,
        )

    def target(self, z, meta, exclude=True):
        probs = []
        for i, (s, t) in enumerate(zip(meta["subject"], meta["trial"])):
            p = self.prototypes(t, exclude).to(z.device)
            q = torch.softmax(torch.einsum("d,scd->sc", z[i], p) * 5, dim=-1)
            mask = [j for j, v in enumerate(self.subjects) if not exclude or v != s]
            assert mask
            probs.append(q[mask].mean(0))
        return torch.stack(probs).detach()


def make_bank(teacher, data, device):
    centers = torch.zeros(len(data.subjects), 3, 15, 32)
    counts = torch.zeros(len(data.subjects), 3, 15)
    ids = [i for i, r in enumerate(data.rows) if r["start_sample"] % 36 == 0]
    teacher.eval()
    with torch.no_grad():
        for a in range(0, len(ids), 128):
            ii = ids[a : a + 128]
            x, _, meta = data.batch(ii, device)
            z = teacher(x, return_features=True)["normalized"].cpu()
            for j, i in enumerate(ii):
                r = data.rows[i]
                index = (
                    data.subjects.index(r["subject"]),
                    r["session"] - 1,
                    r["trial"] - 1,
                )
                centers[index] += z[j]
                counts[index] += 1
    assert (counts > 0).all()
    centers /= counts[..., None]
    assert torch.isfinite(centers).all()
    return (
        Bank(centers, data.subjects),
        dict(counts=counts.tolist(), subjects=data.subjects),
    )
