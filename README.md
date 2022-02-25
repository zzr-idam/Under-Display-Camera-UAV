# Topic : Under Display Camera

# Title : Zero-Reference Image Restoration for Under-Display Camera of UAV

# Abstract : 
The exposed cameras of UAV can shake, shift, or even malfunction under the influence of harsh weather, while the add-on devices (Dupont lines)
are very vulnerable to damage. We can place a low-cost T-OLED overlay around the camera to protect it, but this would also introduce image degradation issues. In particular, the temperature variations in the atmosphere can create mist that adsorbs to the T-OLED, which can cause secondary disasters (i.e., more severe image degradation) during the UAV’s filming process. To solve the image degradation problem caused by overlaying T-OLEDs, in this paper we propose a new method to enhance the visual experience by enhancing the texture and color of images. Specifically, our method trains a lightweight network to estimate a low-rank affine grid on the input image, and then utilizes the grid to enhance the input image at block granularity. The advantages of our method are that no reference image is required and the loss function is developed from visual experience. In addition, our model can perform high-quality recovery of images of arbitrary resolution in real time. In the end, the limitations of our model and the collected datasets (including the daytime and nighttime scenes) are discussed.

Cite: http://arxiv.org/abs/2202.06283

Video dataset :  Link：https://pan.baidu.com/s/1vMEZ3RmUOhiAKiTI_FyRjA 
Password： 3v5w 

Checkpoint: &hearts;

Link: https://pan.baidu.com/s/1TZevvpjGtOMHZwTg__MmtQ
Password: 1trl

## Test code

```python

import torch
import torch.optim
from torchvision import transforms
from PIL import Image
import torch 
from torchvision.utils import save_image
import net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = net.Zero_Net().to(device)
my_model.eval()
my_model.to(device)
my_model.load_state_dict(torch.load("train_model_udc.pth", map_location=torch.device('cpu'))) 
to_pil_image = transforms.ToPILImage()
tfs_full = transforms.Compose([
            transforms.ToTensor()
        ])
i = 0
for idx in range(1):
     image_in = Image.open('3.jpg').convert('RGB')
     full = tfs_full(image_in).unsqueeze(0).to(device)
     output = my_model(full)
     save_image(output[0], '{}.jpg'.format('3_test'))

 ```

## Static quantification approach

https://pytorch.org/docs/stable/quantization.html

## Contributions

Our model can completely replace bilateral grid learning in the field of image enhancement. &hearts; 
Our approach can process images of arbitrary resolution on a single GPU.                    &hearts;

+ https://github.com/google/hdrnet HDRNet
+ https://github.com/mousecpn/Joint-Bilateral-Learning JBL approach


# Thanks

https://github.com/bsun0802/Zero-DCE            Loss functions on our approach.

https://github.com/yunlongdong/semi-dehazing    Loss function on our approach.

https://github.com/rui1996/DeRaindrop           Comparison experiments.
