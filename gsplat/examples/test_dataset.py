
from read_write_model import read_cameras_binary, read_images_binary


images = read_images_binary("/disk/SYNTHETIC_NESPOF_DATA/hotdog_colmap/sparse/0/images.bin")
filtered = {img_id: img.name for img_id, img in images.items() if img.camera_id}

from datasets.colmap import HSIParser, HyperspectralDataset

parser = HSIParser(data_dir = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_npy", colmap_dir = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_colmap/sparse/0")
import torch
points = parser.points
print("points",points.shape)
print("1 point",points[0])
points_torch = torch.from_numpy(parser.points).float()
print(points_torch.shape)
print("2 point", points_torch[0])
print(parser.Ks_dict)

hsi_dataset = HyperspectralDataset(parser=parser,split="train")

image = hsi_dataset.__getitem__(2)["image"]
mask = hsi_dataset.__getitem__(2)["mask"]

print("mask shape", mask.shape)
print("shape", image.shape)
print(image[0,0,:])
print(image[510,510,:])

band_std = image.std(dim=-1)  # (511, 511) — std across 21 bands per pixel, get the standard variation in all the image

print(band_std.min())

# the background has std, fore ground has more variance
mask = band_std > 1e-4  # True = foreground, False = background


print("foreground pixels:", mask.sum().item())
print("background pixels:", (~mask).sum().item())

# masked_image = mask.numpy()
# print(masked_image.shape)
# import imageio
# masked_image = (masked_image * 255).astype(np.uint8)

# imageio.imwrite(
#                         f"1.png",
#                         masked_image,
#                     )


# trainloader = torch.utils.data.DataLoader(
#             hsi_dataset,
#             batch_size=5,
#             shuffle=True,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True,
#         )

# trainloader_iter = iter(trainloader)
# print(trainloader_iter)
# data = next(trainloader_iter)
# print(data["mask"].shape)
# print(data["image"].shape)

# masks = data["mask"].to("cuda") if "mask" in data else None
# print(masks)

# image = data["image"][0]
# mask_ = np.numpy(masks[0])
# print(masked_image.shape)
# import imageio
# masked_image = (masked_image * 255).astype(np.uint8)

# imageio.imwrite(
#                         f"1.png",
#                         masked_image,
#                     )


from datasets.colmap import Parser, Dataset
import numpy as np

parser = HSIParser(data_dir = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_rgb", colmap_dir = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_colmap/sparse/0", rgb_dir=True)

dataset_train = HyperspectralDataset(parser=parser,split="train")


trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

trainloader_iter = iter(trainloader)
print(trainloader_iter)
data = next(trainloader_iter)
masked_image = data["mask"][0].numpy()
image_data = data["image"][0].numpy()
print(data["mask"].shape)
print(data["image"].shape)

print(f"min {image_data.min()} and max {image_data.max()}")


import imageio
masked_image = (masked_image * 255).astype(np.uint8)

imageio.imwrite(
                        f"1.png",
                        masked_image,
                    )



image_data = (image_data * 255).astype(np.uint8)
print(f"min {image_data.min()} and max {image_data.max()}")


imageio.imwrite(
                        f"2.png",
                        image_data,
                    )
