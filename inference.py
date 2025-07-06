import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.pwclite import PWCLite
from models.simple_unet import SimpleUNetMask
from utils.torch_utils import restore_model
from types import SimpleNamespace
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 加载图片
img1_path = '/workspace/UnSAMFlow_data/Sintel-raw/scene/98/09988.png'  # 第一帧
img2_path = '/workspace/UnSAMFlow_data/Sintel-raw/scene/98/09988.png'  # 第二帧
flow_model_path = '/workspace/UnSAMFlow/results/our_maskinput/20250702_062528_sintel_simple_unet/model_ep25_ckpt.pth.tar'
mask_model_path = '/workspace/UnSAMFlow/results/our_maskinput/20250702_062528_sintel_simple_unet/mask_model_ep25_ckpt.pth.tar'
img = Image.open(img1_path).convert('RGB')
img2 = Image.open(img2_path).convert('RGB')

# 2. 预处理
transform = transforms.Compose([
    transforms.Resize((448, 1024)),
    transforms.ToTensor(),  # [0,1]
])
img1 = transform(img).unsqueeze(0)  # [1, 3, H, W]
img2 = transform(img2).unsqueeze(0)

# 4. 加载模型和权重
# flow模型
class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    def __setattr__(self, key, value):
        self[key] = value
    def __contains__(self, item):
        return dict.__contains__(self, item)

cfg = AttrDict({
    "type": "pwclite",
    "input_adj_map": False,
    "input_boundary": False,
    "add_mask_corr": False,
    "reduce_dense": False,
    "learned_upsampler": True,
    "aggregation_type": "residual"
})
flow_model = PWCLite(cfg)
flow_model = restore_model(flow_model,flow_model_path)  # 替换为你的PWCLite权重路径

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img1 = img1.to(device)
img2 = img2.to(device)
flow_model = flow_model.to(device)
flow_model.eval()

# mask模型
mask_model = SimpleUNetMask(in_channels=3, out_channels=10, features=[64,128,256,512], bilinear=True)
mask_model = restore_model(mask_model, mask_model_path)  # 替换为你的mask权重路径
mask_model = mask_model.to(device)
mask_model.eval()

# 5. 推理
with torch.no_grad():
    # mask推理
    mask = mask_model(img1,return_features=False)  # 只用第一帧
    print('Mask shape:', mask.shape)
    mask2,x1,x2,x3,x4,x5 = mask_model(img2,return_features=True)  # 只用第一帧
    full_seg1 = mask.detach().argmax(dim=1,keepdim=True).float()
    full_seg2 = mask2.detach().argmax(dim=1,keepdim=True).float()
    
    # flow推理，PWCLite需要两帧输入
    flow_out = flow_model(img1, img2, full_seg1, full_seg2,with_bk=True)  # 返回dict
    flow = flow_out['flows_12'][0]  # 取正向flow，shape: [1, 2, H, W]
    print('Flow shape:', flow.shape)

    

# 6. 可视化/保存结果（可选）
# flow可视化（只显示u分量）
plt.imshow(flow[0,0].cpu(), cmap='jet')
plt.title('Flow U')
plt.savefig('flow_u.png')

# mask可视化（取最大概率类别）
mask_pred = mask.argmax(dim=1)[0].cpu().numpy()
plt.imshow(mask_pred, cmap='tab20')
plt.title('Mask Prediction')
plt.savefig('mask_pred.png')

# ========== PCA可视化所有特征 ===========
features = [x1, x2, x3, x4, x5]
for idx, feat in enumerate(features, 1):
    arr = feat[0].detach().cpu().numpy()  # [C, H, W]
    C, H, W = arr.shape
    arr_flat = arr.reshape(C, -1).T  # [H*W, C]
    pca = PCA(n_components=3)
    arr_pca = pca.fit_transform(arr_flat)  # [H*W, 3]
    arr_pca_img = arr_pca.reshape(H, W, 3)
    arr_pca_img -= arr_pca_img.min()
    arr_pca_img /= arr_pca_img.max()
    plt.imshow(arr_pca_img)
    plt.title(f'PCA of Feature {idx}')
    plt.axis('off')
    plt.savefig(f'pca_feature{idx}.png')
    print(f"shape of feature{idx}: {arr.shape}")