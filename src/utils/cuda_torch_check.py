import torch
# pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

x = torch.randn(1, 3, 224, 224, device='cuda')
conv = torch.nn.Conv2d(3, 3, 3).cuda()

out = conv(x)
print(out.shape)