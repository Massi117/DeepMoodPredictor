import torch


class SafeAugment:
    def __init__(
        self,
        p_noise=0.5,
        p_scale=0.3,
        p_flip=0.0,   # set >0 only if justified
        noise_sigma=0.02
    ):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_flip = p_flip
        self.noise_sigma = noise_sigma


    def add_gaussian_noise(self, x, sigma=0.02):
        """
        x: torch.Tensor (C, D, H, W)
        sigma: std relative to normalized scale
        """
        noise = torch.randn_like(x) * sigma
        return x + noise
    

    def intensity_scaling(self, x, scale_range=(0.95, 1.05)):
        scale = torch.empty(1).uniform_(*scale_range).to(x.device)
        return x * scale
    

    def flip_left_right(self, x):
        # assumes last dimension is left-right (W)
        return torch.flip(x, dims=[-1])


    def __call__(self, x):
        if torch.rand(1) < self.p_noise:
            x = self.add_gaussian_noise(x, self.noise_sigma)

        if torch.rand(1) < self.p_scale:
            x = self.intensity_scaling(x)

        if self.p_flip > 0 and torch.rand(1) < self.p_flip:
            x = self.flip_left_right(x)

        return x