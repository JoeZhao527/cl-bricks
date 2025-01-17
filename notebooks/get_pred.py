import torch
pred_path = "/share/scratch/haokaizhao/cl-bricks/logs/train/runs/2025-01-17_17-02-20/prediction.pt"
out_path = "./spec_res.pt"
threshold = 0.5

pred = torch.load(pred_path)
res = torch.concat(pred,dim=0)

predicted = len(res[res > threshold])
print(f"threshold: {threshold}, predicted: {predicted}, predicted percentage: {predicted / 315720}")

torch.save(torch.where(res > threshold), out_path)