# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
from render import util

# the camera parameters of video.mp4


def get_camera_params(resolution=512, fov=45, elev_angle=-20, azim_angle=0):
    fovy = np.deg2rad(fov)
    elev = np.radians(elev_angle)
    azim = np.radians(azim_angle)
    proj_mtx = util.perspective(fovy, resolution / resolution, 1, 50)
    mv = util.translate(0, 0, -3) @ (util.rotate_x(elev) @ util.rotate_y(azim))
    normal_rotate = util.rotate_y_1(-azim) @ util.rotate_x_1(-elev)
    # nomral_rotate =  util.rotate_y_1(0) @ util.rotate_x_1(0)
    mvp = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]
    bkgs = torch.ones(1, resolution, resolution, 3,
                      dtype=torch.float32, device='cuda')
    return {
        'mvp': mvp[None, ...].cuda(),
        'mv': mv[None, ...].cuda(),
        'campos': campos[None, ...].cuda(),
        'resolution': [resolution, resolution],
        'spp': 1,
        'background': bkgs,
        'normal_rotate': normal_rotate[None, ...].cuda(),
    }


class DatasetMesh(torch.utils.data.Dataset):

    def __init__(self, glctx, FLAGS, validate=False, gif=False):
        # Init
        self.glctx = glctx
        self.FLAGS = FLAGS
        self.validate = validate
        self.gif = gif
        # self.custom             = costum
        self.aspect = FLAGS.train_res[1] / FLAGS.train_res[0]
        self.fovy_range_min = np.deg2rad(FLAGS.fovy_range[0])
        self.fovy_range_max = np.deg2rad(FLAGS.fovy_range[1])
        self.elevation_range_min = np.deg2rad(FLAGS.elevation_range[0])
        self.elevation_range_max = np.deg2rad(FLAGS.elevation_range[1])
        self.angle_front = np.deg2rad(FLAGS.front_threshold)

    def _gif_scene(self, itr):
        fovy = np.deg2rad(45)
        proj_mtx = util.perspective(
            fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        ang = (itr / 100) * np.pi * 2
        rotate_x = np.deg2rad(20)
        prompt_index = 0
        mv = util.translate(
            0, 0, -3) @ (util.rotate_x(-rotate_x) @ util.rotate_y(ang))
        normal_rotate = util.rotate_y_1(0)
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None, ...], prompt_index

    def _validate_scene(self, itr):
        fovy = np.deg2rad(45)
        proj_mtx = util.perspective(
            fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        ang = (itr / 4) * np.pi * 2
        rotate_x = np.random.uniform(-np.pi/4, np.pi/18)
        prompt_index = 0
        mv = util.translate(
            0, 0, -3) @ (util.rotate_x(rotate_x) @ util.rotate_y(ang))
        normal_rotate = util.rotate_y_1(0)
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None, ...], prompt_index

    def _train_scene(self, itr):
        fovy = np.random.uniform(self.fovy_range_min, self.fovy_range_max)
        proj_mtx = util.perspective(
            fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        if self.FLAGS.gpu_number == 8:
            if self.FLAGS.local_rank in [0, 4]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1, 5]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2, 6]:  # back
                rotate_y = np.random.uniform(np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3, 7]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 7:
            if self.FLAGS.local_rank in [0, 4]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1, 5]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2, 6]:  # back
                rotate_y = np.random.uniform(np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 6:
            if self.FLAGS.local_rank in [0, 4]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1, 5]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2]:  # back
                rotate_y = np.random.uniform(np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 5:
            if self.FLAGS.local_rank in [0, 4]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2]:  # back
                rotate_y = np.random.uniform(np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 4:
            if self.FLAGS.local_rank in [0]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2]:  # back
                rotate_y = np.random.uniform(np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 3:
            if self.FLAGS.local_rank in [0]:
                rotate_y = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
            elif self.FLAGS.local_rank in [1]:
                rotate_y = np.random.uniform(np.deg2rad(60), np.deg2rad(180))
            elif self.FLAGS.local_rank in [2]:
                rotate_y = np.random.uniform(np.deg2rad(-60), np.deg2rad(-180))
        # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
        elif self.FLAGS.gpu_number == 2:
            if self.FLAGS.local_rank in [0]:
                rotate_y = np.random.uniform(np.deg2rad(-90), np.deg2rad(90))
            elif self.FLAGS.local_rank in [1]:
                rotate_y = np.random.uniform(np.deg2rad(90), np.deg2rad(270))
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        else:
            # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
            rotate_y = np.random.uniform(np.deg2rad(-180), np.deg2rad(180))

        rotate_x = - \
            np.random.uniform(self.elevation_range_min,
                              self.elevation_range_max)
        # angle_front = np.deg2rad(45)
        prompt_index = get_view_direction(
            thetas=rotate_x, phis=rotate_y, front=self.angle_front)
        cam_radius = 3
        x = np.random.uniform(-self.FLAGS.camera_random_jitter,
                              self.FLAGS.camera_random_jitter)
        y = np.random.uniform(-self.FLAGS.camera_random_jitter,
                              self.FLAGS.camera_random_jitter)
        mv = util.translate(
            x, y, -cam_radius) @ (util.rotate_x(rotate_x) @ util.rotate_y(rotate_y))
        if ((itr+1)/self.FLAGS.batch) <= self.FLAGS.coarse_iter:
            rotate_y1 = np.random.uniform(0, np.pi*2)
            rotate_x1 = np.random.uniform(-np.pi, np.pi)
            normal_rotate = util.rotate_y_1(
                rotate_y1) @ util.rotate_x_1(rotate_x1)
        else:
            normal_rotate = util.rotate_y_1(0)@util.rotate_x_1(0)
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        if self.FLAGS.semantic_guidance or self.FLAGS.structure_guidance:
            assert self.FLAGS.surround_view_num != 0
            surround_view_num = self.FLAGS.surround_view_num
            surround_view_fovy = np.deg2rad(45)
            surround_view_proj_mtx = util.perspective(
                surround_view_fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
            sourround_rotate_x = 0
            sourround_normal_rotate = util.rotate_y_1(0)
            ngle_step = (2 * np.pi) / surround_view_num
            surround_view_dict_temp = {}

            for i in range(surround_view_num):
                surround_view_ang = i * ngle_step
                surround_view_mv = util.translate(
                    0, 0, -3) @ (util.rotate_x(sourround_rotate_x) @ util.rotate_y(surround_view_ang))
                surround_view_mvp = surround_view_proj_mtx @ surround_view_mv
                surround_view_campos = torch.linalg.inv(
                    surround_view_mv)[:3, 3]
                prompt_index = get_view_direction(
                    thetas=sourround_rotate_x, phis=surround_view_ang, front=self.angle_front)
                surround_view_dict_temp[f'view_{i}'] = {
                    'mvp': surround_view_mvp[None, ...],
                    'mv': surround_view_mv[None, ...],
                    'campos': surround_view_campos[None, ...],
                    'resolution': self.FLAGS.train_res,
                    'spp': self.FLAGS.spp,
                    'normal_rotate': sourround_normal_rotate[None, ...],
                    'prompt_index': prompt_index,
                }

            surround_view_dict = {
                'mvp': torch.cat([item['mvp'] for item in surround_view_dict_temp.values()], dim=0),
                'mv': torch.cat([item['mv'] for item in surround_view_dict_temp.values()], dim=0),
                'campos': torch.cat([item['campos'] for item in surround_view_dict_temp.values()], dim=0),
                'resolution': self.FLAGS.train_res,
                'spp': self.FLAGS.spp,
                'normal_rotate': torch.cat([item['normal_rotate'] for item in surround_view_dict_temp.values()], dim=0),
                'prompt_index': np.array([item['prompt_index'] for item in surround_view_dict_temp.values()], dtype=np.int32),
            }
        else:
            surround_view_dict = None

        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None, ...], prompt_index, surround_view_dict

    def __len__(self):
        if self.gif == True:
            return 100
        else:
            return 4 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        surround = None
        if self.gif:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index = self._gif_scene(
                itr)
        elif self.validate:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index = self._validate_scene(
                itr)
        else:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index, surround = self._train_scene(
                itr)

        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': iter_res,
            'spp': iter_spp,
            'normal_rotate': normal_rotate,
            'prompt_index': prompt_index,
            'surround': surround
        }

    def collate(self, batch):
        iter_res, iter_spp, surround = batch[0]['resolution'], batch[
            0]['spp'], batch[0]['surround']

        return {
            'mv': torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp': torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos': torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution': iter_res,
            'spp': iter_spp,
            'normal_rotate': torch.cat(list([item['normal_rotate'] for item in batch]), dim=0),
            'prompt_index': np.array([item['prompt_index'] for item in batch], dtype=np.int32),
            'surround': surround,
        }


@torch.no_grad()
def get_view_direction(thetas, phis, front):
    #                   phis [B,];  -pi~pi        thetas: [B,] -pi/2~pi/2
    # front = 0         [-front, front)
    # side (left) = 1   [front, pi - front)
    # back = 2          [pi - front, pi) or [-pi, -pi+front)
    # side (right) = 3  [-pi+front, - front)

    if (phis >= -front) and (phis < front):
        prompt_index = 0
    elif (phis >= front) and (phis < np.pi - front):
        prompt_index = 1
    elif (phis >= np.pi - front) or (phis < -np.pi + front):
        prompt_index = 2
    elif (phis >= -np.pi + front) and (phis < -front):
        prompt_index = 3

    return prompt_index
