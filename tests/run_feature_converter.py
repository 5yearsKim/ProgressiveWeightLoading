from pwl_model.layers.feature_converter import FeatureConverter
import torch
import torch.nn as nn
import torch.nn.functional as F



def init_orthonormal_rows(weight: torch.Tensor):
    """
    Initialize 'weight' so that its rows are orthonormal: WW^T = I.
    Works for rectangular weight of shape (rows, cols):
      - If rows <= cols: generate a cols×cols orthogonal matrix and take first 'rows' rows.
      - If rows >  cols: directly apply orthogonal_ (requires rows >= cols).
    """
    rows, cols = weight.shape
    print('rows, cols: ', rows, cols)
    if rows <= cols:
        temp = torch.empty(cols, cols)
        nn.init.orthogonal_(temp)
        weight.data.copy_(temp[:rows, :])
    else:
        nn.init.orthogonal_(weight)
    
    # print('test orthogonality')
    # print(weight @ weight.t())


def test_converter(shape, dim_s, dim_t, tol=1e-6):
    print(f"\nTesting shape={shape}, dim_t={dim_t}, dim_s={dim_s}")
    conv = FeatureConverter(dim_s=dim_s, dim_t=dim_t)

    # Orthonormal-init the linear weight and zero-bias
    init_orthonormal_rows(conv.linear.weight)
    if conv.linear.bias is not None:
        conv.linear.bias.data.zero_()

    conv.eval()

    x = torch.randn(shape)
    y = conv(x, reverse=False)
    x_rec = conv(y, reverse=True)

    print('x_rec shape:', x_rec.shape)
    max_diff = (x - x_rec).abs().max().item()
    print(f"  max abs difference after round-trip: {max_diff:.2e}")
    print("  PASSED" if max_diff < tol else "  FAILED")


torch.manual_seed(0)
test_converter((1, 10), dim_s=5,  dim_t=10)
test_converter((1, 3, 10), dim_s=2,  dim_t=3)
test_converter((1, 5, 10, 10), dim_s=2, dim_t=5)

