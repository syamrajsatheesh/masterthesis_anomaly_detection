from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics


class RaPP:
    def __init__(
        self,
        model,
        rapp_start_index: int = 1,
        rapp_end_index: int = -1,
        loss_reduction: str = "sum",
    ):
        assert hasattr(model, "encoder")
        assert loss_reduction in ["sum", "mean"]
        super().__init__()
        self.model = model
        self.rapp_start_index = rapp_start_index
        self.rapp_end_index = (
            len(self.model.encoder) + rapp_end_index + 1
            if rapp_end_index < 0
            else rapp_end_index
        )
        assert self.rapp_start_index < self.rapp_end_index
        self.reduction_fn = torch.mean if loss_reduction == "mean" else torch.sum
        self.mu = None
        self.s = None
        self.v = None

    def get_pathaway_recon_diff(
        self, x: torch.Tensor, recon_x: torch.Tensor
    ) -> torch.Tensor:
        diffs = [recon_x - x]
        for layer in self.model.encoder[: self.rapp_end_index]:
            x = layer(x)
            recon_x = layer(recon_x)
            diff = recon_x - x
            diffs.append(diff)
        diffs = torch.cat(diffs[self.rapp_start_index :], dim=1)
        return diffs

    @torch.no_grad()
    def fit(self, train_loader: DataLoader) -> None:
        self.model.eval()
        outputs = []
        for batch in train_loader:
            outputs += [self.training_step(batch)]
        self.training_epoch_end(outputs)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        recon_x = self.model(x)
        diffs = self.get_pathaway_recon_diff(x, recon_x)
        return diffs

    def training_epoch_end(self, outputs: List[Any]) -> None:
        outputs = torch.cat(outputs, dim=0)
        self.mu = outputs.mean(dim=0, keepdim=True)
        _, self.s, self.v = (outputs - self.mu).svd()

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        outputs = []
        for batch in test_loader:
            outputs += [self.test_step(batch)]
        result = self.test_epoch_end(outputs)
        return result

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        x, y = batch
        recon_x = self.model(x)
        score = self.reduction_fn((recon_x - x) ** 2, dim=1)
        diffs = self.get_pathaway_recon_diff(x, recon_x)
        return {"score": score, "diffs": diffs, "label": y}

    def test_epoch_end(self, outputs: List[Any]) -> Dict[str, float]:
        rapp_score = []
        score = []
        label = []
        for output in outputs:
            rapp_score += [output["diffs"]]
            score += [output["score"]]
            label += [output["label"]]

        label = torch.cat(label).numpy()
        score = torch.cat(score).numpy()

        rapp_score = torch.cat(rapp_score, dim=0)
        sap_score = self.reduction_fn(rapp_score ** 2, dim=1).numpy()
        nap_score = self.reduction_fn(
            (torch.mm(rapp_score - self.mu, self.v) / self.s) ** 2, dim=1
        ).numpy()

        auroc = get_auroc(label, score)
        aupr = get_aupr(label, score)
        sap_auroc = get_auroc(label, sap_score)
        sap_aupr = get_aupr(label, sap_score)
        nap_auroc = get_auroc(label, nap_score)
        nap_aupr = get_aupr(label, nap_score)
        result = {
            "auroc": auroc,
            "aupr": aupr,
            "sap_auroc": sap_auroc,
            "sap_aupr": sap_aupr,
            "nap_auroc": nap_auroc,
            "nap_aupr": nap_aupr,
        }
        return result

def get_auroc(label: np.ndarray, score: np.ndarray) -> float:
    fprs, tprs, _ = metrics.roc_curve(label, score)
    return metrics.auc(fprs, tprs)


def get_aupr(label: np.ndarray, score: np.ndarray) -> float:
    precisions, recalls, _ = metrics.precision_recall_curve(label, score)
    return metrics.auc(recalls, precisions)
