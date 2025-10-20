import math
import zlib
import struct
# import qrcode
import os
import numpy as np
import cv2
# import base64
from tqdm import tqdm
import reedsolo
import config as cg
from myqr import qr


""""""
# 大致流程如下：[分组 -> 补长] -> rs -> 块头 -> qr -> 视频
""""""


def read_and_divide(file_path, bytes_per_frame = cg.bytes_per_frame, rs_mode=cg.rs_mode):
    """
    读取一个二进制文件，并切分为数据块
    
    参数：
        file_path (str): 文件路径
        bytes_per_frame(int): 每块字节数
        rs_mode(bool): 是否开启rs模式
    返回:
        list[bytes]: 原始数据块
        total(int): 块总数
    """
    raw_data_blocks = []
    try:
        with open(file_path, 'rb') as file:
            # 循环读取指定大小的数据块
            while True:
                # 读取指定字节数
                block = file.read(bytes_per_frame)
                if not block:  # 读取到文件末尾
                    break
                raw_data_blocks.append(block)

            # 只有rs模式才需要补长
            if rs_mode:
                # 末块补长
                last_block = raw_data_blocks[-1]
                last_block_length = len(last_block)
                # 处理末块，可能出现大小刚好是 bytes_per_frame - 1 的情况，但是不管了，直接报错
                if last_block_length != bytes_per_frame:
                    if last_block_length == bytes_per_frame - 1:
                        raise ValueError("the last_block_length is bytes_per_frame - 1")
                    # 补齐长度
                    padding_length = bytes_per_frame - last_block_length
                    if padding_length > 0:
                        last_block = last_block + b"z" * padding_length
                        raw_data_blocks[-1] = last_block
                    # 写入长度信息
                    length_bytes = struct.pack(">H", last_block_length)
                    last_block = last_block[:-2] + length_bytes
                    raw_data_blocks[-1] = last_block
                    # print(f"pad length :{last_block_length}")
        # print(f"len(raw_data_blocks) is :{len(raw_data_blocks)}")
        return raw_data_blocks, len(raw_data_blocks)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found in read and divide: {file_path}")
    except IOError as e:
        raise IOError(f"io error in read and divide: {str(e)}")


def encode_rs_per_group(group_blocks, bytes_per_frame = cg.bytes_per_frame, rs_factor=cg.rs_factor):
    """
    对单组数据块进行垂直rs编码生成冗余块

    参数：
        group_blocks[bytes]: 单组原始数据块
        bytes_per_frame(int): 每块字节数
        rs_factor(double): 冗余率
    返回：
        list[bytes]: 该组冗余数据块
    """
    k = len(group_blocks)
    nsym = math.ceil(k * rs_factor)
    # n = k + nsym
    # 垂直编码
    coder = reedsolo.RSCodec(nsym)

    rs_blocks = [bytearray(bytes_per_frame) for _ in range(nsym)]

    for i in range(bytes_per_frame):
        column_data = bytearray([block[i] for block in group_blocks])
        encoded_column = coder.encode(column_data)
        parity_bytes = encoded_column[k:]
        
        for j in range(nsym):
            rs_blocks[j][i] = parity_bytes[j]

    return [bytes(block) for block in rs_blocks]


def encode_rs(raw_data_blocks, rs_group_size = cg.rs_group_size):
    """
    ### 生成并添加全局冗余块 首先记录最后一块字节长度并补齐同一长度 然后再rs

    参数：
        raw_data_blocks[bytes]: 原始数据块
    返回：
        blocks[bytes]: 添加了rs冗余块的数据块
    """
    # 添加rs块
    rs_blocks = []
    # 按组切分
    for i in tqdm(range(0, len(raw_data_blocks), rs_group_size), 
                  desc="添加rs块", 
                  total=math.ceil(len(raw_data_blocks) / rs_group_size)):
        group_blocks = raw_data_blocks[i : i + rs_group_size]
        rs_blocks_per_group = encode_rs_per_group(group_blocks)
        rs_blocks.extend(rs_blocks_per_group)
    
    return raw_data_blocks + rs_blocks


def generate_qr_sequence(blocks, total, output_dir="frames_encode"):
    """
    将数据块序列编码为二维码图片

    参数：
        blocks(list[bytes]): 已经加了块头的数据块，后续可能加上冗余块
        output_dir(str): 输出目录（保存二维码图片）
    返回：
        list[str]: 每个二维码图片的文件路径
    """
    # # 创建输出目录
    # os.makedirs(output_dir, exist_ok=True)
    
    file_paths = []
    qrcoder = qr()
    for i, block in tqdm(enumerate(blocks), desc="生成图片", total=len(blocks)):
        file_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        file_paths.append(file_path)
        qrcoder.add_data(block, i, total)
        qrcoder.make(file_path)
        # # 设置qr参数
        # qr = qrcode.QRCode(
        # error_correction=qrcode.ERROR_CORRECT_M,    # 15% 纠错
        # box_size=5,
        # border=4
        # )

        # encoded_block = base64.b64encode(block)
        # if test_mode:
        #     print(f"encode_block size : {len(encoded_block)}")
        # qr.add_data(encoded_block)
        # qr.make(fit=False)

        # img = qr.make_image(fill_color="black", back_color="white")

        # file_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        # img.save(file_path)
        # file_paths.append(file_path)
        # if test_mode:
        #     print(f"len is {len(encoded_block)}")
        #     print(f"qr version is {qr.version}")

    return file_paths


def images_to_video(image_paths, output_path, fps = 60, frame_repeat = 2):
    """
    将图片序列以固定帧率，每个图片重复固定次数，生成视频
    ps: 这部分都是ai写的

    参数：
        imageput_dir(str): 图片序列路径
        output_path(str): 视频输出位置，包含后缀
        fps(int): 输出帧率
        frame_repeat(int): 每帧重复次数
    """
    # 设置分辨率
    frame = cv2.imread(image_paths[0])
    h, w, _ = frame.shape
    total = len(image_paths)

    video = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (w, h)
        )
    
    def write_empty_frame(video, empty_frame, repeat):
        for _ in range(repeat):
            video.write(empty_frame)
    
    white_frame = np.full((h, w, 3), 255, dtype=np.uint8)
    write_empty_frame(video, white_frame, 2)

    for image in tqdm(image_paths, desc="生成视频", total=total):
        frame = cv2.imread(image)
        for _ in range(frame_repeat):   # 重复写入
            video.write(frame)
        # write_empty_frame(video, white_frame, 2)

    write_empty_frame(video, white_frame, 2)
    video.release()


# 别问为什么有两种函数注释模式，因为我才知道还有这种写法，前面的就懒得改了
def cal_speed(file_path, video_path):
    """
    计算传输速度

    :param file_path: 原始文件路径
    :param video_path: 视频文件路径
    :return: 编码速度kbps
    """
    file_size = os.path.getsize(file_path) * 8 / 1000   # kb
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    # print(f"duration is {duration}, file_size is {file_size}")
    return file_size / duration