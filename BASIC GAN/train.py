import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# local imports
from models import Generator, Discriminator


# HYPERPARAMETERS
device = "cuda:0" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

# model initialization
disc_model = Discriminator(img_dim=image_dim).to(device=device)
gen_model = Generator(z_dim=z_dim, img_dim=image_dim).to(device=device)


# data preparation
fixed_noise = torch.randn((batch_size, z_dim)).to(device=device)

# data transform
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# dataset
dataset = datasets.MNIST(root="./dataset", transform=data_transforms, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# training
optim_generator = torch.optim.Adam(gen_model.parameters(), lr=lr)
optim_discriminator = torch.optim.Adam(disc_model.parameters(), lr=lr)

criterion = nn.BCELoss()

# tensorboard setup
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# training loop
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (real_images, _) in enumerate(loader):
        real_images = real_images.view(-1, 784).to(device)
        noise = torch.randn((batch_size, z_dim)).to(device=device)
        # image generation
        fake_generated_images = gen_model(noise)
        # train discriminator maximize (log(D(real))+log(1 - D(G(noise))))
        disc_real = disc_model(real_images).view(-1)  # flatten the prediction
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # prediction on generated images
        disc_fake = disc_model(fake_generated_images).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc_model.zero_grad()
        lossD.backward(retain_graph=True)
        optim_discriminator.step()

        # train generator minimize (log(1 - D(G(noise)))) => maximize log(D(G(noise)))
        output = disc_model(fake_generated_images).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen_model.zero_grad()
        lossG.backward()
        optim_generator.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen_model(fixed_noise).reshape(-1, 1, 28, 28)
                data = real_images.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1


# save the generator model
torch.save(gen_model.state_dict(), "generator_model.pth")
