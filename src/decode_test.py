import cv2
import struct
import zlib
from pyzbar.pyzbar import decode, ZBarSymbol
import os
import base64
from tqdm import tqdm


""""""
# 参数配置

test_mode = False
""""""


def decode_frames(file_path):
    """
    读取视频每一帧并直接解码保存为raw_data_blocks

    参数：
        file_path(str): 待解码的视频文件路径
    返回：
        list[bytes]: 二维码读取出来的数据块
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("cannot open a video file")
        exit()

    raw_data_blocks = []

    pbar = tqdm(desc="解码视频", unit="帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 灰度化
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 二值化
        # _, frame = cv2.threshold(frame, 50, 205, cv2.THRESH_BINARY)
        # 读取二维数据
        results = decode(frame, symbols=[ZBarSymbol.QRCODE])
        if results:
            decoded_block = base64.b64decode(results[0].data)
            raw_data_blocks.append(decoded_block)

        pbar.update()

    return raw_data_blocks


def parse_header(raw_block):
    """
    解析块头

    参数：
        raw_block(bytes): 待解析的块
    返回：
        index(int): 块编号
        total(int): 总块数
        crc(int): crc
        block(bytes): 块数据
    """
    index, total = struct.unpack(">HH", raw_block[0:4])
    crc = struct.unpack(">I", raw_block[4:8])[0]
    block = raw_block[8:]
    if test_mode:
        print(f"index is : {index}, total is : {total}, crc is : {crc}")
    return index, total, crc, block


def parse_blocks(raw_data_blocks):
    """
    解析块并验证crc 按编号保留首个crc通过的块 分离出rs恢复块(即index从total开始的所有块)

    参数：
        raw_blocks[bytes]: 待解析的块
    返回：
        total_blocks(int): 总块数
        blocks_dict: 已去重并校验过crc的块字典
        rs_blocks[bytes]: 冗余块
    """
    total_blocks = -1
    blocks_dict = {}
    rs_blocks = []
    for raw_block in tqdm(raw_data_blocks, desc="解析数据块", total=len(raw_data_blocks)):
        index, total, crc, block = parse_header(raw_block)
        # 去重
        if index in blocks_dict:
            continue
        # 校验
        if zlib.crc32(block) & 0xffffffff != crc:
            continue
        # 校验通过后初始化总块数
        if total_blocks == -1:
            total_blocks = total
        # 分离冗余块和数据块
        if index >= total:
            rs_blocks.append((index, block))  # 保留编号和数据
        else:
            blocks_dict[index] = block

    return total_blocks, blocks_dict, rs_blocks


def check_blocks(blocks_dict, total):
    """
    遍历查询缺失块，若可能则进行恢复，若无法恢复，返回缺失块数量或索引列表
    
    参数：
        blocks_dict: 已去重并校验过crc的块字典
    返回：
        blocks[bytes] or None: 完整数据块表 缺失返回None
    """
    missing = [i for i in range(total) if i not in blocks_dict]

    if missing:
        # 输出缺失块信息
        print(f"检测到缺失块，共 {len(missing)} 个")
        print(f"缺失块索引：{missing}")
        return None  # 有缺失时返回None
    else:
        return [blocks_dict[i] for i in range(total)]


def restore_global_rs():
    """用冗余块进行恢复"""


def reconstructed_file(blocks, file_name):
    """
    按编号拼接数据，并写入文件

    参数：
        blocks[bytes]: 完整数据块表
        file_name(str): 输出文件名（含后缀）
    """
    if blocks is None or not blocks:
        print("error in reconstructed_file by blocks")
        return
    try:
        with open(file_name, "wb") as file:
            for block in tqdm(blocks, desc="写入文件", total=len(blocks)):
                file.write(block)
    except Exception as e:
        print(f"error in reconstructed_file while writing file:{str(e)}")


def main():
    import time
    input_file_path = "test/output.mp4"
    output_file_path = "test/jiheon_decoded.jpg"
    start_time = time.time()

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # raw_data_blocks = decode_frames("test/output.mp4")
    # total, blocks_dict, rs_blocks = parse_blocks(raw_data_blocks)
    # blocks = check_blocks(blocks_dict, total)
    # if blocks is None:
    #     return
    # reconstructed_file(blocks, "test/output.bin")

    decode_start = time.time()
    raw_data_blocks = decode_frames(input_file_path)
    decode_end = time.time()

    parse_start = time.time()
    total, blocks_dict, rs_blocks = parse_blocks(raw_data_blocks)
    parse_end = time.time()

    check_start = time.time()
    blocks = check_blocks(blocks_dict, total)
    check_end = time.time()

    if blocks is None:
        print("数据块不完整，无法重建文件")
        return

    reconstruct_start = time.time()
    reconstructed_file(blocks, output_file_path)
    reconstruct_end = time.time()

    # 总用时
    total_time = time.time() - start_time
    print(f"\n程序运行完成")
    print(f"解码视频帧耗时: {decode_end - decode_start:.4f} 秒")
    print(f"解析数据块耗时: {parse_end - parse_start:.4f} 秒")
    print(f"检查数据块耗时: {check_end - check_start:.4f} 秒")
    print(f"重建文件耗时: {reconstruct_end - reconstruct_start:.4f} 秒")
    print(f"总程序运行时间: {total_time:.4f} 秒")

if __name__ == "__main__":
    main()