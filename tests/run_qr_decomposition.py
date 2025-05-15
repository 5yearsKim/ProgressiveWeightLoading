import torch

# 1) Fix seed
torch.manual_seed(42)

# 2) Create raw w (8×4) and build a band-mask
w_raw = torch.randn(8, 4)
bandwidth = 0 

m, n = w_raw.size()

# make a mask so only |i-j| ≤ bandwidth stay non-zero
rows = torch.arange(w_raw.size(0)).unsqueeze(1)   # shape (8,1)
cols = torch.arange(w_raw.size(1)).unsqueeze(0)   # shape (1,4)

mask = ((rows - cols) <= abs(m - n) + bandwidth)  & ((rows - cols) >= - bandwidth)

# apply mask
w = w_raw * mask.to(torch.float32)

print('w:')
print(w)

x = torch.tensor([1., 2., 3., 4])

# 3) QR decomposition (reduced mode)
#    Now Q.shape = (8,4), R.shape = (4,4)
Q, R = torch.linalg.qr(w)

# 4) Multiply Q @ x to get an 8-dim vector
y = Q @ x

print("w.shape:", w.shape)   # → torch.Size([8, 4])
print("Q:", Q)   # → torch.Size([8, 4])
print("R:", R)   # → torch.Size([4, 4])
print("y.shape:", y.shape)   # → torch.Size([8])
print("y:", y)

x_orig = Q.T @ y
print("x_orig:", x_orig)   # → torch.Size([4])

# 5) Quick check
assert y.shape == (8,), f"Expected y.shape to be (8,), but got {y.shape}"

# #  check if Q is orthogonal
# print("Q.T @ Q:", Q.T @ Q)   # → torch.Size([4, 4])


