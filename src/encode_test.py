import zlib
import struct
import qrcode
import os
import shutil
import cv2
import base64
from tqdm import tqdm


""""""
# 参数配置
bytes_per_frame = 1536  # 单块数据量

test_mode = False
# FLAG_START = b'S'    # 起始标志
# FLAG_DATA = b'D'     # 数据标志
# FLAG_END = b'E'      # 结束标志
# FLAG_REDUNDANT = b'R'# 冗余标志
""""""


def read_and_divide(file_path):
    """
    读取一个二进制文件，并切分为数据块
    
    参数：
        file_path (str): 文件路径
    返回:
        list[bytes]: 原始数据块
    """
    data_blocks = []
    try:
        with open(file_path, 'rb') as file:
            # 循环读取指定大小的数据块
            while True:
                # 读取指定字节数
                block = file.read(bytes_per_frame)
                if not block:  # 读取到文件末尾
                    break
                data_blocks.append(block)
                
        return data_blocks
        
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found in read and divide: {file_path}")
    except IOError as e:
        raise IOError(f"io error in read and divide: {str(e)}")


def add_header(data_blocks):
    """
    给所有数据块添加块头 
        去除 因为可以用超过index来表示冗余 编号表示起始和终止->定义标识符为bytes类型 分为起始、数据、终止、冗余
    
    参数：
        data_blocks: 原始bytes数据块
    返回：
        list[bytes]: 含块头的数据块
    """
    total = len(data_blocks)
    headed_blocks = []
    for index, block in tqdm(enumerate(data_blocks), desc="添加数据块头", total=total):
        # # 确定当前块的标志
        # if index == 0:
        #     flag = FLAG_START
        # elif index == total - 1:
        #     flag = FLAG_END
        # else:
        #     flag = FLAG_DATA
        # 添加块头
        # headed_block = add_header_per_block(block, flag, index, total)
        headed_block = add_header_per_block(block, index, total)
        headed_blocks.append(headed_block)

    return headed_blocks


# def add_header_per_block(block, flag, index, total):
def add_header_per_block(block, index, total):
    """
    给每个数据块进行crc并添加块头 块构成为([标识符])[块编号/总块数][CRC][数据]

    参数：
        block(bytes)    : 数据块
        # flag(bytes)     : 标识符
        index(int)      : 块编号
        total(int)      : 总块数
    返回：
        bytes: 含块头的数据块
    """
    # 计算crc
    crc = zlib.crc32(block) & 0xffffffff
    crc_bytes = struct.pack(">I", crc)
    # 拼接编号
    order = struct.pack(">HH", index, total)

    if test_mode:
        print(f"index is : {index}, total is : {total}, crc is : {crc}")
    return order + crc_bytes + block


def generate_global_rs():
    """生成并添加全局冗余块"""
    


def generate_qr_sequence(blocks, output_dir="frames_encode"):
    """
    将数据块序列编码为二维码图片

    参数：
        blocks(list[bytes]): 已经加了块头的数据块，后续可能加上冗余块
        output_dir(str): 输出目录（保存二维码图片）
    返回：
        list[str]: 每个二维码图片的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = []
    total = len(blocks)

    for i, block in tqdm(enumerate(blocks), desc="生成二维码图片", total=total):
        # 设置qr参数
        qr = qrcode.QRCode(
        version=40,                                 # 固定最大版本
        error_correction=qrcode.ERROR_CORRECT_M,    # 15% 纠错
        box_size=10,
        border=4
        )

        encoded_block = base64.b64encode(block)
        if test_mode:
            print(f"encode_block size : {len(encoded_block)}")
        qr.add_data(encoded_block)
        qr.make(fit=False)

        img = qr.make_image(fill_color="black", back_color="white")

        file_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        img.save(file_path)
        file_paths.append(file_path)

    return file_paths


def images_to_video(image_paths, output_path, fps = 60, frame_repeat = 4):
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

    for image in tqdm(image_paths, desc="生成视频", total=total):
        frame = cv2.imread(image)
        for _ in range(frame_repeat):   # 重复写入
            video.write(frame)

    video.release()


def main():
    import time

    # if os.path.exists("test/frames_encode"):
    #     shutil.rmtree("test/frames_encode")
    # data_blocks = read_and_divide("test/input.bin")
    # headed_blocks = add_header(data_blocks)
    # image_paths = generate_qr_sequence(headed_blocks, "test/frames_encode")
    # images_to_video(image_paths, "test/output.mp4")
    input_file_path = "test/jiheon.jpg"
    output_file_path = "test/output.mp4"
    frames_file_path = "test/frames_encode"

    # 记录程序开始时间
    start_time = time.time()
    
    if os.path.exists(frames_file_path):
        shutil.rmtree(frames_file_path)
    
    # 记录各阶段开始时间（可选，用于分析各步骤耗时）
    read_start = time.time()
    data_blocks = read_and_divide(input_file_path)
    read_end = time.time()
    
    header_start = time.time()
    headed_blocks = add_header(data_blocks)
    header_end = time.time()
    
    qr_start = time.time()
    image_paths = generate_qr_sequence(headed_blocks, frames_file_path)
    qr_end = time.time()
    
    video_start = time.time()
    images_to_video(image_paths, output_file_path)
    video_end = time.time()
    
    # 记录程序结束时间
    end_time = time.time()
    
    # 计算总耗时
    total_time = end_time - start_time
    
    # 打印计时结果
    print(f"\n程序运行完成")
    print(f"文件读取与分块: {read_end - read_start:.2f} 秒")
    print(f"添加块头: {header_end - header_start:.2f} 秒")
    print(f"生成二维码: {qr_end - qr_start:.2f} 秒")
    print(f"生成视频: {video_end - video_start:.2f} 秒")
    print(f"总耗时: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()