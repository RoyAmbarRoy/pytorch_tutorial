import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.zeros(3, requires_grad=True) #torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(x)
print(y)
print(w)
print(b)
print(z)
print(loss)


# Backprobagation function: grad_fn
print(z.grad_fn, loss.grad_fn)

# Computing gradients
loss.backward(retain_graph=True)
loss.backward(retain_graph=True)
print(w.grad, b.grad)

# Disabling gradient tracking
z = torch.matmul(x, w) + b
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
# Another way - detach()
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z.requires_grad)
print(z_det.requires_grad)


# Jacobian product
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
print('inp: ', inp, 'out', out)
out.backward(torch.ones_like(inp), retain_graph=True)
print('First call\n', inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print('\nSecond call\n', inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print('\nCall after zeroing gradient\n', inp.grad)