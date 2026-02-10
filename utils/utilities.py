import math
import torch
import shutil
import random
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch.distributed as dist
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class RandomSharpness:
    """在[min_factor, max_factor]范围内随机选择锐化强度"""
    def __init__(self, min_factor=1.0, max_factor=3.0, p=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 在范围内随机采样一个锐化因子
            factor = random.uniform(self.min_factor, self.max_factor)
            return transforms.functional.adjust_sharpness(img, factor)
        return img


@torch.jit.script
class AverageMeterRefine(object):
    """Computes and stores the average and current value"""
    def __init__(self, dtype: torch.dtype, device_str: str = "cuda"):
        device = torch.device(device_str)
        self.val_ori = torch.tensor(0., dtype=dtype, device=device)
        self.avg_ori = torch.tensor(0., dtype=dtype, device=device)
        self.sum_ori = torch.tensor(0., dtype=dtype, device=device)
        self.count_ori = torch.tensor(0., dtype=dtype,device=device)
        
        self.val =torch.tensor(0., dtype=dtype, device=device)
        self.avg =torch.tensor(0., dtype=dtype, device=device)
        self.sum =torch.tensor(0., dtype=dtype, device=device)
        self.count =torch.tensor(0., dtype=dtype,device=device)

    def reset(self):
        self.val = self.val_ori 
        self.avg = self.avg_ori 
        self.sum = self.sum_ori 
        self.count = self.count_ori 

    def update(self, val: torch.Tensor, n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # def all_reduce(self, device='cpu'):
    #     # if torch.cuda.is_available():
    #     #     device = torch.device("cuda")
    #     # else:
    #     #     device = torch.device("cpu")
    #     total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    #     dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    #     self.sum, self.count = total.tolist()
    #     self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

def save_model(state, is_best, current_ckp_name='checkpoint.pth.tar', best_ckp_name='model_best.pth.tar'):
    """ save checkpoint during training """
    torch.save(state, current_ckp_name)
    if is_best:
        shutil.copyfile(current_ckp_name, best_ckp_name)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["state_dict"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["state_dict"] = averaged_params
    return new_state

def adjust_learning_rate(optimizer, init_lr, epoch, decay_points):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr
    if epoch in decay_points:
        cur_lr *= 0.1

        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

def MaskOutROI(image, roi, value_scale=255):
    # solder_pad_roi is a list of 2d-arrays

    mask = np.zeros_like(image, dtype=np.int32)
    cv2.fillPoly(mask, roi, (value_scale, value_scale, value_scale))

    roi_np = np.vstack(roi)
    xmin, ymin = np.min(roi_np, 0)
    xmax, ymax = np.max(roi_np, 0)

    masked_image = image*mask
    masked_image_crop = masked_image[ymin:ymax, xmin:xmax]
    return masked_image_crop, mask

# TODO 用于测试，不需要实现
import io
def compress_png_by_PIL(x:Image.Image, q):
    if x.mode == 'L':
        gray = True
        img_jpg = x  # 保持灰度图像
    else:
        gray = False
        img_jpg = x.convert('RGB')  # 转换为灰度图像
    # img_jpg = x.convert('RGB')

    # 将RGB图像保存到内存中的字节流
    with io.BytesIO() as output:
        img_jpg.save(output, format='JPEG', quality=q) # 指定JPEG质量
        jpeg_data = output.getvalue() # 获取JPEG字节数据

    # 将字节数据转换为NumPy数组``
    nparr = np.frombuffer(jpeg_data, np.uint8)

    # 使用cv2.imdecode将字节数据解码为图像, 此时的imgcv与x是一样的图像,
    # 即cv2.imwrite(imgcv)与x.save的可视化结果一致，但是训练要的是np.array(x)
    # 所以imgcv还要做一步cv2.cvtColor
    if not gray:
        imgcv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_cvt = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)

        img_t = torch.from_numpy(img_cvt).permute(2, 0, 1).contiguous()
    else:
        imgcv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 
        img_t = torch.from_numpy(imgcv).unsqueeze(0)
    return img_t

def compress_png_by_opencv(x:np.array, q):
    # 检查输入图像是否为灰度图像
    if len(x.shape) == 2:  # 灰度图像
        gray = True
    else:
        gray = False
        # 如果是彩色图像，转换为灰度图像
    
    _, jpeg_buffer = cv2.imencode('.jpg', x, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    # 使用cv2.imdecode将字节数据解码为图像, imgcv和x可视化结果一样
    if not gray:
        imgcv = cv2.imdecode(jpeg_buffer, cv2.IMREAD_COLOR)

        img_t = torch.from_numpy(imgcv).permute(2, 0, 1).contiguous()
    else:
        imgcv = cv2.imdecode(jpeg_buffer, cv2.IMREAD_GRAYSCALE)
        img_t = torch.from_numpy(imgcv).unsqueeze(0)
    return img_t

def compress_img(img: Image.Image, p, select_p):
    if select_p >= 0.5:
        compress_img = compress_png_by_PIL(img, p)
    else:
        # read_image(os.path.join(self.root_folder, img1_path)) = np.array(img).transpose(2,0,1)
        # 所以imgcv必须等于np.array(img)
        img_array = np.array(img)
        compress_img = compress_png_by_opencv(img_array, p)
    
    return compress_img

def compress_img_opencv(img: Image.Image, p):
    img_array = np.array(img)
    compress_img = compress_png_by_opencv(img_array, p)

    return compress_img

def compress_img_PIL(img: Image.Image, p):
    compress_img = compress_png_by_PIL(img, p)
    
    return compress_img


def diff_rgb_white(s1_X, s1_match_X_img, return_PIL = False):
    S1_diff = s1_X - s1_match_X_img

    s1_X = S1_diff - S1_diff.min()

    s1_X = s1_X / 510 # 255 * 2
    s1_match_X_img = s1_match_X_img / 255

    if return_PIL:
        s1_X = Image.fromarray((s1_X * 255).astype(np.uint8))
        s1_match_X_img = Image.fromarray((s1_match_X_img * 255).astype(np.uint8))
    return s1_X, s1_match_X_img

def diff_rgb_white_abs(s1_X, s1_match_X_img, return_PIL = False):
    s1_h, s1_w, _ = s1_X.shape
    s1_match_h, s1_match_w, _ = s1_match_X_img.shape
    h_s = min(s1_h, s1_match_h)
    w_s = min(s1_w, s1_match_w)

    s1_X = abs(s1_X[:h_s, :w_s, :] - s1_match_X_img[:h_s, :w_s, :])

    if return_PIL:
        s1_X = Image.fromarray((s1_X).astype(np.uint8))
        s1_match_X_img = Image.fromarray((s1_match_X_img).astype(np.uint8))
        return s1_X, s1_match_X_img

    s1_X = s1_X / 255
    s1_match_X_img = s1_match_X_img / 255
    return s1_X, s1_match_X_img

def max_rgb_white(s1_X, s1_match_X_img, return_PIL = False):
    if isinstance(s1_X, torch.Tensor):
        s1_X = torch.maximum(s1_X, s1_match_X_img)
    elif isinstance(s1_X, np.ndarray):
        s1_X = np.maximum(s1_X, s1_match_X_img)

    if return_PIL:
        s1_X = Image.fromarray(s1_X.astype(np.uint8))
        s1_match_X_img = Image.fromarray((s1_match_X_img).astype(np.uint8))

    else:
        s1_X = s1_X / 255
        s1_match_X_img = s1_match_X_img / 255

    return s1_X, s1_match_X_img

def add_rgb_white(s1_X, s1_match_X_img, return_PIL = False):
    s1_X = s1_X + s1_match_X_img

    s1_X = s1_X / 510
    s1_match_X_img = s1_match_X_img / 255
    if return_PIL:
        s1_X = Image.fromarray((s1_X * 255).astype(np.uint8))
        s1_match_X_img = Image.fromarray((s1_match_X_img * 255).astype(np.uint8))

    return s1_X, s1_match_X_img

# the following relevant for inference using onnx or torch model
import cv2
import os
class TransformImageFusion():
    def __init__(self, img_path, img_type = 'png', rs_img_size_h=256, rs_img_size_w=256, gray_mean = None, gray_std = None,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], transform=None, sharpness_save = None, fusion_type = 'diff'):
        self.scale = np.float32(1.0 / 255.0)
        shape = (1, 1, 3)
        self.rs_img_size_h = rs_img_size_h
        self.rs_img_size_w = rs_img_size_w

        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

        if gray_mean:
            self.gray_mean = np.array(gray_mean).reshape((1,1)).astype('float32')
            self.gray_std = np.array(gray_mean).reshape((1,1)).astype('float32')

        if '_rgb_' in img_path:
            s1_match_img_path = img_path.replace('_rgb_', '_white_')
            if fusion_type == 'diff_white_rgb' or fusion_type == 'cat_HW_white_rgb':
                img_path_tp = img_path
                img_path = s1_match_img_path
                s1_match_img_path = img_path_tp

        elif '_white_' in img_path:
            s1_match_img_path = img_path.replace('_white_', '_rgb_')

            if fusion_type == 'diff_rgb_white' or fusion_type == 'cat_HW_rgb_white':
                img_path_tp = img_path
                img_path = s1_match_img_path
                s1_match_img_path = img_path_tp

        elif '_RGB_' in img_path:
            s1_match_img_path = img_path.replace('_RGB_', '_WHITE_')
            if fusion_type == 'diff_white_rgb' or fusion_type == 'cat_HW_white_rgb':
                img_path_tp = img_path
                img_path = s1_match_img_path
                s1_match_img_path = img_path_tp
            
        elif '_WHITE_' in img_path:
            s1_match_img_path = img_path.replace('_WHITE_', '_RGB_')

            if fusion_type == 'diff_rgb_white' or fusion_type == 'cat_HW_rgb_white':
                img_path_tp = img_path
                img_path = s1_match_img_path
                s1_match_img_path = img_path_tp

        # load img data
        img = np.array(Image.open(img_path)).astype(np.float32)
        if fusion_type == 'cat_gray' or fusion_type == 'merge_rb_G':
            if '_white_' in s1_match_img_path or '_WHITE_' in s1_match_img_path:
                s1_X_img = np.array(Image.open(s1_match_img_path).convert('L')).astype(np.float32)
            else:
                s1_X_img = np.array(Image.open(img_path).convert('L')).astype(np.float32)
                img = np.array(Image.open(s1_match_img_path)).astype(np.float32)

        else:
            s1_X_img = np.array(Image.open(s1_match_img_path)).astype(np.float32)

        if 'diff' in fusion_type and 'abs' not in fusion_type:
            img, s1_X_img = diff_rgb_white(img, s1_X_img, return_PIL = True)

        elif fusion_type == 'diff_rgb_white_abs':   
            img, s1_X_img = diff_rgb_white_abs(img, s1_X_img, return_PIL = True)
        elif fusion_type == 'max':
            img, s1_X_img = max_rgb_white(img, s1_X_img, return_PIL = True)
        elif fusion_type == 'add':
            img, s1_X_img = add_rgb_white(img, s1_X_img, return_PIL = True)

        # 这里在transform里面做的transpose
        if transform is not None:
            img = transform(img)
            s1_X_img = transform(s1_X_img)

        if isinstance(img, Image.Image):
            img = np.array(img)
            s1_X_img = np.array(s1_X_img)

        # if fusion_type == 'cat':
        #     pass

        # if img.shape[-1] == 4:
        #     self.img = img[:,:,:3]
        # else:
        self.fusion_type = fusion_type
        self.img = img
        self.s1_X_img = s1_X_img


    def transform(self):
        img =  cv2.resize(self.img, (self.rs_img_size_w, self.rs_img_size_h), interpolation = cv2.INTER_NEAREST)

        img = img.astype('float32') * self.scale # 归一化
        img = (img - self.mean) / self.std
        # w, h, c = img.shape
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))

        s1_X_img =  cv2.resize(self.s1_X_img, (self.rs_img_size_w, self.rs_img_size_h), interpolation = cv2.INTER_NEAREST)
        s1_X_img = s1_X_img.astype('float32') * self.scale

        if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
            s1_X_img = (s1_X_img - self.gray_mean) / self.gray_std
            s1_X_img = np.expand_dims(s1_X_img, axis=-1)
        else:
            s1_X_img = (s1_X_img - self.mean) / self.std

            # w, h, c = s1_X_img.shape
        s1_X_img = np.expand_dims(s1_X_img, axis=0)
        s1_X_img = np.transpose(s1_X_img, (0, 3, 1, 2))

        if self.fusion_type == 'cat' or self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
            if  self.fusion_type == 'merge_rb_G':
                img = img[:, [0, 2], :, :]
            img = np.concatenate([img, s1_X_img], axis=1) # 通道维度拼接
        elif 'cat_HW' in self.fusion_type:
            img = np.concatenate([img, s1_X_img], axis=2) # 通道维度拼接
            
        return img


class TransformImage():
    def __init__(self, img_path, img_type = 'png', rs_img_size_h=256, rs_img_size_w=256,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], transform=None, sharpness_save = None):
        self.scale = np.float32(1.0 / 255.0)
        shape = (1, 1, 3)
        self.rs_img_size_h = rs_img_size_h
        self.rs_img_size_w = rs_img_size_w

        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

        if img_type =='png':
            # img_path = img_path.replace('.jpg', '.png')
            img = Image.open(img_path)
 
            # self.img = img
        elif img_type == 'jpg':
            img_nojpg = Image.open(img_path)
            img_jpg = np.array(img_nojpg)
            img_jpg = cv2.cvtColor(img_jpg, cv2.COLOR_BGR2RGB)

            jpg_path = img_path.replace('.png', '.jpg')
            cv2.imwrite(jpg_path, img_jpg)

            img = Image.open(jpg_path)

            # 比较opevcv读取数据和PIL读取数据是否一致，结果时一致的
            # imgcv = cv2.imread(jpg_path)
            os.system(f'sudo rm {jpg_path}')
        # read_image(os.path.join(self.root_folder, img1_path)) = np.array(Image.open(path)).transpose(2,0,1),
        # 这里在transform里面做的transpose
        if transform is not None:
            # img_name = os.path.basename(img_path)

            # ori_img_save_path = os.path.join(sharpness_save + '-ori', img_name)
            # img.save(ori_img_save_path)
            img = transform(img)
            # tran_img_save_path = os.path.join(sharpness_save, img_name)
            # img.save(tran_img_save_path)

        if isinstance(img, Image.Image):
            img = np.array(img)

        if img.shape[-1] == 4:
            self.img = img[:,:,:3]
        else:
            self.img = img

    def transform(self):
        img =  cv2.resize(self.img, (self.rs_img_size_w, self.rs_img_size_h), interpolation = cv2.INTER_NEAREST)

        img = img.astype('float32') * self.scale
        img = (img - self.mean) / self.std
        w, h, c = img.shape
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        return img

def split(a, n):
    # split array a into n approximately equal splits
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == '__main__':

    from torchvision.io import read_image
    import os
    import pandas as pd
    rs_img_size = 224
    part_name_list = ['pins']
    processed_defect_data_folder = '/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data/'
    date = '230314'
    for part_name in part_name_list:
        annotation_path = os.path.join(processed_defect_data_folder, 'merged_annotation', date,
                                       f'annotation_labels_{part_name}.csv')
        annotation_df = pd.read_csv(annotation_path, index_col=0).reset_index(drop=True)
        counter = 0
        for insp_image_path in annotation_df[f'{part_name}_file_path']:
            counter += 1
            if counter % 500 == 0:
                print(counter)
            insp_image = TransformImage(img_path=insp_image_path, rs_img_size=rs_img_size).transform()
            s1_X = read_image(insp_image_path)
            if s1_X.shape[0] == 4:
                s1_X = s1_X[:3]
    # import matplotlib.pyplot as plt
    # import cv2
    # import os
    # img_folder = '/home/robinru/shiyuan_projects/data/aoi_defect_data_20220906'
    # region = 'pad'
    # L_ref = 'AD082-H.0616R_20201203135858855_NG_L2_12NH_2503.png'
    # C_ref = 'AD008C-K.001_20201111121845907_NG_COMP1000_C10_1000.png'
    # R_ref = 'AD008.0161R_20201030184714117_NG_R3_1.01.0269R_1055.png'
    # R_ref = 'AD008.0161R_20201030164304612_NG_R3_1.01.0269R_1055.png'
    # for img_ref in [L_ref, C_ref, R_ref]:
    #     img_sub = cv2.imread(f'{img_folder}/{region}/{region}_{img_ref}', cv2.IMREAD_COLOR)
    #     img_show = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
    #     plt.imshow(img_show, cmap='gray')
    #     plt.title(f'{img_ref}')
    #     plt.show()
    # import os
    #
    # region_list =  ['component', 'package', 'pad', 'pins']
    # seed = 42
    # rs_img_size = 224
    # backbone_arch_list = ['mobilenetv3small_pretrained', 'mobilenetv3large_pretrained',
    #                       'fcdropoutmobilenetv3small_pretrained', 'fcdropoutmobilenetv3large_pretrained',
    #                       'resnet18', 'resnet18_pretrained',
    #                       'fcdropoutresnet18', 'fcdropoutresnet18_pretrained',
    #                       ]
    # for region in region_list:
    #     for backbone_arch in backbone_arch_list:
    #         # backbone_arch = backbone_arch_list[0]
    #         if region == 'component':
    #             defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4,
    #                            'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
    #                            'tombstone': 8, 'solder_ball': 10, 'others': 11}  # v1.31, v1.32
    #
    #         elif region == 'package':
    #             # mclass shouldn't know what's wrong component
    #             defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'others': 11}  # v1.32
    #
    #         elif region == 'pad':
    #             defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5, 'pseudosolder': 6,
    #                            'solder_ball': 10, 'others': 11}  # v1.32
    #
    #         elif region == 'pins':
    #             defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5, 'pseudosolder': 6,
    #                            'solder_shortage': 7, 'others': 11}  # v1.32
    #         n_class = len(defect_code)
    #         version_name = 'v1.45fp16'
    #         ckp_folder = '../models/checkpoints'
    #         version_folder = 'v1.45'
    #         checkpoint_list = [os.path.join(ckp_folder, version_folder,
    #                                       f'{region}{backbone_arch}rs{rs_img_size}s{seed}c{n_class}_ckp_best{version_name}top{i}.pth.tar') for i in range(4)]
    #         new_state =  average_checkpoints(checkpoint_list)
    #         checkpoint_path = os.path.join(ckp_folder, version_folder,
    #                                       f'{region}{backbone_arch}rs{rs_img_size}s{seed}c{n_class}_ckp_best{version_name}ema.pth.tar')
    #         torch.save(new_state, checkpoint_path)