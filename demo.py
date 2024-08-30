import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from unidepth.models import UniDepthV1

import pyvista as pv

import time

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model

#image_path = "./images/img.jpg"


# image et intrinsics sans resize 
image_path = "./images/pipe.jpg"

rgb_image_numpy = np.array(Image.open(image_path))

rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

intrinsics_ivm = np.array([
    [322.6092376708984, 0.0, 257.7363166809082],
    [0.0, 322.6092376708984, 186.6225147247314],
    [0.0, 0.0, 1.0]
])

intrinsics_ivm_tensor = torch.from_numpy(intrinsics_ivm).float().to(device)




# # image resize
#
# # Load and resize the image
# image_path = "./images/pipe.jpg"
# original_image = Image.open(image_path)
#
# # Desired new size (for example, 256x256)
# new_size = (256, 256)
# resized_image = original_image.resize(new_size)
#
# # Convert to numpy array and then to torch tensor
# rgb_image_numpy = np.array(resized_image)
# rgb = torch.from_numpy(rgb_image_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)  # C, H, W -> 1, C, H, W
#
# intrinsics_ivm = np.array([
#     [322.6092376708984, 0.0, 257.7363166809082],
#     [0.0, 322.6092376708984, 186.6225147247314],
#     [0.0, 0.0, 1.0]
# ])
#
# # Original image size
# original_size = original_image.size  # (width, height)
#
# # Scaling factors for width and height
# scale_x = new_size[0] / original_size[0]
# scale_y = new_size[1] / original_size[1]
#
# # Adjust the intrinsic parameters
# intrinsics_ivm[0, 0] *= scale_x  # fx
# intrinsics_ivm[1, 1] *= scale_y  # fy
# intrinsics_ivm[0, 2] *= scale_x  # cx
# intrinsics_ivm[1, 2] *= scale_y  # cy
#
# intrinsics_ivm_tensor = torch.from_numpy(intrinsics_ivm).float().to(device)









# Number of inferences
num_iterations = 100
total_inference_time = 0

# Warm-up iterations (to avoid any cold-start delays)
for _ in range(10):
    _ = model.infer(rgb)

# Loop over multiple inferences
for _ in range(num_iterations):
    start_time = time.time()
    predictions = model.infer(rgb)
    end_time = time.time()
    
    total_inference_time += (end_time - start_time)

# Calculate average inference time
average_inference_time = total_inference_time / num_iterations

print(f"Average Inference Time over {num_iterations} iterations: {average_inference_time:.4f} seconds")




# Measure inference time
start_time = time.time()

predictions = model.infer(rgb)

end_time = time.time()
inference_time = end_time - start_time

print(f"Inference Time: {inference_time:.4f} seconds")



# Metric Depth Estimation
depth = predictions["depth"]

print("depth : ", depth)


# Point Cloud in Camera Coordinate
xyz = predictions["points"]




# Intrinsics Prediction
intrinsics = predictions["intrinsics"]


print("intrinsics : ", intrinsics)


intrinsics_path = "assets/demo/intrinsics.npy"

# Load the intrinsics if available
intrinsics = torch.from_numpy(np.load(intrinsics_path)) # 3 x 3

print("intrinsics : ", intrinsics)


# Convertir le tensor en numpy array (déplacer sur CPU si nécessaire)
depth_np = depth.squeeze().cpu().numpy()



# Créer une figure avec 2 sous-graphiques
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Afficher l'image RGB de base dans le premier sous-graphe
ax[0].imshow(rgb_image_numpy)
ax[0].set_title("Image de Base")
ax[0].axis('off')  # Masquer les axes pour une meilleure visualisation

# Afficher la depth map dans le deuxième sous-graphe
im = ax[1].imshow(depth_np, cmap='plasma')
ax[1].set_title("Depth Map")
ax[1].axis('off')  # Masquer les axes pour une meilleure visualisation

# Ajouter une barre de couleur à la depth map
fig.colorbar(im, ax=ax[1])

# Afficher la figure
plt.show()

# 322.6092376708984, 322.6092376708984, 257.7363166809082, 186.6225147247314

# intrinsics_ivm = np.array([ [322.6092376708984, 0.0, 257.7363166809082],
#                             [0.0, 322.6092376708984, 186.6225147247314],
#                             [0.0, 0.0, 1.0]])


#intrinsics_ivm_tensor = torch.from_numpy(intrinsics_ivm).float()


# Measure inference time
start_time = time.time()

predictions = model.infer(rgb, intrinsics_ivm_tensor)

end_time = time.time()
inference_time = end_time - start_time

print(f"Inference Time: {inference_time:.4f} seconds")





depth = predictions["depth"]

# Convertir le tensor en numpy array (déplacer sur CPU si nécessaire)
depth_np = depth.squeeze().cpu().numpy()

# Créer une figure avec 2 sous-graphiques
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Afficher l'image RGB de base dans le premier sous-graphe
ax[0].imshow(rgb_image_numpy)
ax[0].set_title("Image de Base")
ax[0].axis('off')  # Masquer les axes pour une meilleure visualisation

# Afficher la depth map dans le deuxième sous-graphe
im = ax[1].imshow(depth_np, cmap='plasma')
ax[1].set_title("Depth Map")
ax[1].axis('off')  # Masquer les axes pour une meilleure visualisation

# Ajouter une barre de couleur à la depth map
fig.colorbar(im, ax=ax[1])

# Afficher la figure
plt.show()

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

print("intrinsics pred : ", intrinsics)

print("intrinsics ivm : ", intrinsics_ivm)


# Point Cloud in Camera Coordinate
xyz = predictions["points"]

print("xyz : ", xyz)
print("xyz.shape : ", xyz.shape)


# affichage pointcloud
# Convertir le tensor en NumPy array et déplacer sur CPU si nécessaire
xyz_np = xyz.squeeze().cpu().numpy()  # shape: (3, 376, 514)

# Réorganiser les dimensions pour obtenir un tableau de points (N, 3)
xyz_np = np.transpose(xyz_np, (1, 2, 0))  # shape: (376, 514, 3)
points = xyz_np.reshape(-1, 3)  # shape: (376*514, 3)

# # Créer un nuage de points PyVista
# point_cloud = pv.PolyData(points)
#
# # Visualiser avec PyVista
# plotter = pv.Plotter()
# plotter.add_mesh(point_cloud, point_size=1, render_points_as_spheres=True)
# plotter.show()

# Extraire la coordonnée Z pour le gradient de couleur
z_values = points[:, 2]  # Utiliser la profondeur (Z) comme valeur scalaire pour les couleurs

# Créer un nuage de points PyVista
point_cloud = pv.PolyData(points)

# Ajouter les valeurs Z comme scalaires pour le gradient de couleur
point_cloud['Z'] = z_values

# Créer un plotter PyVista
plotter = pv.Plotter()

# Ajouter le nuage de points avec un gradient de couleur basé sur Z
plotter.add_mesh(point_cloud, scalars='Z', cmap='viridis', point_size=3, render_points_as_spheres=True)

# Afficher le nuage de points avec le gradient de couleur
plotter.show()



# # Afficher la depth map avec matplotlib
# plt.imshow(depth_np, cmap='plasma')
# plt.colorbar()
# plt.title("Depth Map")
# plt.show()
