import os
import torch
import open3d as o3d
import numpy as np
import torch.nn as nn

from shap_e.util.notebooks import decode_latent_mesh
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents

class SobelEdgeDetector(nn.Module):
    def __init__(self, in_channels):
        super(SobelEdgeDetector, self).__init__()
        # Define Sobel filters
        sobel_x_weights = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float16).view(1, 1, 3, 3)
        sobel_y_weights = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float16).view(1, 1, 3, 3)

        # Initialize Conv2d layers with Sobel weights
        self.conv_x = nn.Conv2d(
            in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(
            in_channels, 1, kernel_size=3, padding=1, bias=False)

        # Set the weights to Sobel filters for each channel
        self.conv_x.weight.data = sobel_x_weights.repeat(1, in_channels, 1, 1)
        self.conv_y.weight.data = sobel_y_weights.repeat(1, in_channels, 1, 1)

    def forward(self, x):
        grad_x = self.conv_x(x) / 24
        grad_y = self.conv_y(x) / 24
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad

def exponential_decay_generator(initial_value, decay_rate):
    current_value = initial_value
    while True:
        yield current_value
        current_value *= decay_rate
        
def get_normalize_mesh_shapE(prompt,finetuned_model=False):
    ckpt = 'shapE_finetuned_with_330kdata.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading shapeE model...')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    # comment the below line to use the original model
    if finetuned_model:
        model.load_state_dict(torch.load(os.path.join('load/shapE-finetuned', ckpt), map_location=device
                                        )['model_state_dict'])
    diffusion = diffusion_from_config(load_config('diffusion'))
    print('ShapeE model loaded.')
    batch_size = 1
    guidance_scale = 15.0

    print('Generating coarse model...')
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    with torch.no_grad():
        gen_mesh = decode_latent_mesh(xm, latents).tri_mesh()

    vertices = np.asarray(gen_mesh.verts)
    faces = np.asarray(gen_mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    vertices = np.asarray(o3d_mesh.vertices)
    shift = np.mean(vertices, axis=0)
    scale = np.max(np.linalg.norm(vertices - shift, ord=2, axis=1))
    vertices = (vertices - shift) / scale
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return o3d_mesh


def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(pro_path)
    vertices = np.asarray(mesh.vertices)
    shift = np.mean(vertices, axis=0)
    scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1))
    vertices = (vertices-shift) / scale
    mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
    return mesh
