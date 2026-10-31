import torch
from decompose import load_sae

model = load_sae()
print(model)  # 打印模型结构，看 decoder 是否带 bias=True

z = torch.zeros(1280 * 128, device="cuda", dtype=torch.half)
print("||decoder(0)|| =", model.decoder(z).float().norm().item())
print("||decode(0)||  =", model.decode(z).float().norm().item())