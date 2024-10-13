import torchvision.transforms as transforms

class Transform:
    def __init__(self, resize=224, mean=0.5, std=0.5):
        self.resize = resize
        self.mean = mean
        self.std = std

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(self.resize),  # 调整图像大小
            transforms.ToTensor(),           # 转换为Tensor
            transforms.Normalize((self.mean,), (self.std,))  # 标准化
        ])
