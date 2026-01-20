import torch
import torch.nn as nn

L1 = nn.L1Loss()
L2 = nn.MSELoss()

class PSNRLoss(nn.Module):
    def __init__(self, reduce_coef=0.1) -> None:
        super().__init__()
        self.reduce_coef = reduce_coef

    def forward(self, pred, gt):
        delta = pred - gt
        delta = torch.where(delta < 0, delta * self.reduce_coef, delta)
        tmp_zero = torch.zeros_like(delta)

        l1 = L1(delta, tmp_zero)
        l2 = L2(delta, tmp_zero)

        return l1 + l2

class RMSELoss(nn.Module):
    def __init__(self, reduce_coef=0.1) -> None:
        super().__init__()
        self.reduce_coef = reduce_coef

    def forward(self, pred, gt):
        delta = pred - gt
        delta = torch.where(delta > 0, delta * self.reduce_coef, delta)
        tmp_zero = torch.zeros_like(delta)

        l1 = L1(delta, tmp_zero)
        l2 = L2(delta, tmp_zero)

        return l1 + l2


class RLoss(nn.Module):
    def __init__(self, eps=1e-8, reduce_coef=0.1):
        super().__init__()
        self.eps = eps
        self.psnr_loss_fn = PSNRLoss(reduce_coef)
        self.rmse_loss_fn = RMSELoss(reduce_coef)

    def forward(self, r, rmse, psnr, Imax, N):
        # Compute sqrt(N) * r
        rmse_pred = torch.sqrt(N) * r
        
        # Compute loss1: L1 between sqrt(N)*r and rmse
        rmse_loss = self.rmse_loss_fn(rmse_pred, rmse)

        # Compute loss2: L1 between PSNR formula and ground truth PSNR
        # PSNR = 20 * log10(Imax / (sqrt(N) * r))
        # To avoid numerical issues, compute log10 of a safe value
        rmse_ped_safe = torch.clamp(rmse_pred, min=self.eps)
        psnr_pred = 20 * torch.log10(Imax / rmse_ped_safe)
        
        # Clip PSNR predictions to reasonable range
        psnr_pred = torch.clamp(psnr_pred, min=0, max=100)

        psnr_loss = self.psnr_loss_fn(psnr_pred, psnr)

        # Combine losses
        total_loss = rmse_loss + psnr_loss
        
        # Debug logging (optional)
        if torch.isnan(total_loss).any():
            print(f"Warning: NaN detected! r min: {r.min().item():.6f}, r max: {r.max().item():.6f}")
            print(f"rmse_pred min: {rmse_pred.min().item():.6f}, max: {rmse_pred.max().item():.6f}")
            print(f"psnr_pred min: {psnr_pred.min().item():.6f}, max: {psnr_pred.max().item():.6f}")
        
        result = {
            'total_loss': total_loss,
            'rmse_loss': rmse_loss,
            'psnr_loss': psnr_loss
        }

        return result


