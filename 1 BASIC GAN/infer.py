import torch
from torchvision.utils import save_image, make_grid
from models import Generator

# Constants
z_dim = 64
image_dim = 28 * 28 * 1
n_images = 10
grid_size = 5  # Grid size for displaying images, adjust as needed

# Load the trained generator model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
gen_model = Generator(z_dim=z_dim, img_dim=image_dim).to(device)
gen_model.load_state_dict(torch.load("generator_model.pth"))
gen_model.eval()

# Generate random noise
noise = torch.randn(n_images, z_dim).to(device)

# Generate images from noise
with torch.no_grad():
    generated_images = gen_model(noise).reshape(n_images, 1, 28, 28)

# Make a grid of images
grid = make_grid(generated_images, nrow=grid_size, normalize=True)

# Save the grid image
save_image(grid, "generated_images_grid.png")
