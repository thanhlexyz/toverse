import torch.nn as nn
import torch

class PerturbationLayer(nn.Module):
    
    def __init__(self, x: torch.Tensor, perturbation: str, kernel_size: int):
        super(PerturbationLayer, self).__init__()
        
        assert perturbation in ["motion_blur_0", "motion_blur_45", "motion_blur_90", "motion_blur_135", "box_blur", "sharpen"], f"Unknown perturbation: {perturbation}"
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        self.perturbation = perturbation
        self.kernel_size = kernel_size
        c = x.shape[1]
        
        A = self._create_kernels()
        
        conv_A = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=c,
            bias=False
        )
        
        with torch.no_grad():
            conv_A.weight.data = A.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            conv_result_A = conv_A(x)
        
        self.image_shape = (-1,) + x.shape[1:]
        
        output_size = conv_result_A.numel()
        self.linear_layer = nn.Linear(1, output_size, bias=True)
        
        with torch.no_grad():
            self.linear_layer.weight.data = conv_result_A.flatten().unsqueeze(1)
            self.linear_layer.bias.data = x.flatten()
    
    def _create_kernels(self) -> torch.Tensor:
        A = torch.zeros((self.kernel_size, self.kernel_size), dtype=torch.float32)
        
        if "motion_blur" in self.perturbation:
            if "45" in self.perturbation:
                A.fill_diagonal_(1 / self.kernel_size)
                A = torch.fliplr(A)
            elif "90" in self.perturbation:
                A[self.kernel_size // 2, :] = 1 / self.kernel_size
            elif "135" in self.perturbation:
                A.fill_diagonal_(1 / self.kernel_size)
            elif "0" in self.perturbation:
                A[:, self.kernel_size // 2] = 1 / self.kernel_size
            
            A[self.kernel_size // 2, self.kernel_size // 2] = 1 / self.kernel_size - 1
            
        elif self.perturbation == "box_blur":
            A.fill_(1 / self.kernel_size**2)
            A[self.kernel_size // 2, self.kernel_size // 2] = 1 / self.kernel_size**2 - 1
            
        elif self.perturbation == "sharpen":
            a_negative = -1 / (self.kernel_size**2 - (self.kernel_size - 1) * ((self.kernel_size - 1) / 2 + 1) - 1)
            A.fill_(a_negative)
            A[self.kernel_size // 2, self.kernel_size // 2] = 1
            
            for j in range(self.kernel_size):
                if j <= self.kernel_size // 2:
                    zero_end = (self.kernel_size - 1) // 2 - j
                else:
                    zero_end = j - (self.kernel_size - 1) // 2
                
                for i in range(zero_end):
                    A[j, i] = 0
                    A[j, self.kernel_size - 1 - i] = 0
            
        else:
            raise ValueError(f"Unknown perturbation: {self.perturbation}")
        
        return A

    
    def forward(self, z):
        output_flat = self.linear_layer(z)
        return output_flat.view(self.image_shape)

if __name__ == "__main__":

    x = torch.randn(1, 1, 28, 28)

    perturb_layer = PerturbationLayer(x, "motion_blur_45", kernel_size=3)

    z = torch.tensor([[0.5]])
    
    output = perturb_layer(z)
    print(output.shape)
    
