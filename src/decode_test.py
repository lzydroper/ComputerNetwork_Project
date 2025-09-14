import cv2
import struct
import zlib
from pyzbar.pyzbar import decode, ZBarSymbol
import os
import base64
from tqdm import tqdm
import reedsolo
import config as cg
import math


""""""
# 大致流程如下：[写入 <- 还原] <- rs <- 块头 <- [base64 <- qr] <- 视频
# 参数配置
bytes_per_frame = cg.bytes_per_frame  # 单块数据量
rs_group_size = cg.rs_group_size     # rs块分组大小
rs_factor = cg.rs_factor        # 冗余率
rs_mode = cg.rs_mode         # rs
test_mode = cg.test_mode       # 测试模式
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

    if not raw_data_blocks:
        raise RuntimeError("decode frame error: cannot read any qrcode")

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
        parsed_block(bytes): 块数据
    """
    index, total = struct.unpack(">HH", raw_block[0:4])
    crc = struct.unpack(">I", raw_block[4:8])[0]
    parsed_block = raw_block[8:]

    if test_mode:
        print(f"index is : {index}, total is : {total}, crc is : {crc}")
    return index, total, crc, parsed_block


def parse_blocks(raw_blocks):
    """
    解析块并验证crc 按编号保留首个crc通过的块

    参数：
        raw_blocks[bytes]: 待解析的块
    返回：
        total_data_blocks(int): 总数据块数
        total_rs_blocks(int): 总rs块数
        parsed_blocks[bytes or None]: 已去重并校验过crc的块表 若缺失则为None 0-total为数据块 total以后为rs块
    """
    total_data_blocks = -1
    total_rs_blocks = -1
    parsed_blocks = []
    for raw_block in tqdm(raw_blocks, desc="解析块头", total=len(raw_blocks)):
        index, total, crc, parsed_block = parse_header(raw_block)
        # 去重
        if total_data_blocks != -1 and parsed_blocks[index] is not None:
            continue
        # 校验
        if zlib.crc32(parsed_block) & 0xffffffff != crc:
            continue
        # 校验通过后初始化总块数
        if total_data_blocks == -1:
            total_data_blocks = total
            total_rs_blocks = math.ceil(total_data_blocks * rs_factor)
            parsed_blocks = [None] * (total_data_blocks + total_rs_blocks)

        parsed_blocks[index] = parsed_block

    return total_data_blocks, total_rs_blocks, parsed_blocks


def check_blocks(parsed_blocks, total_data_blocks, total_rs_blocks):
    """
    遍历查询缺失块，若可能则进行恢复，若无法恢复，返回缺失块数量或索引列表
    
    参数：
        parsed_blocks[bytes or None]: 已去重并校验过crc的块表 若缺失则为None 0-total为数据块 total以后为rs块
        total_data_blocks(int): 总数据块数
        total_rs_blocks(int): 总rs块数
    返回：
        data_blocks[bytes] or None: 完整数据块表 缺失返回None
    """
    parsed_data_blocks = parsed_blocks[:total_data_blocks]
    missing = [i for i, block in enumerate(parsed_data_blocks) if block is None]
    if len(missing) > 0:
        print(f"need blocks :{total_data_blocks}, but find missing block :{len(missing)}")
        print(f"missing block index is :{missing}")
        return decode_rs(parsed_blocks, total_data_blocks, total_rs_blocks)
    else:
        return parsed_blocks[:total_data_blocks]


def decode_rs_per_group(group_blocks, data_blocks_num, rs_blocks_num):
    """
    对单子块进行rs解码还原

    参数：
        group_blocks[bytes or None]: 该组块 前半部分为数据块 后半部分为rs块
        data_blocks_num(int): 原始数据块数目
        rs_blocks_num(int): 冗余块数目
    返回：
        restored_grou[bytes]: 还原的该组原始数据
    """
    coder = reedsolo.RSCodec(rs_blocks_num)

    erase_pos = [i for i, block in enumerate(group_blocks) if block is None]

    repaired_columns = []
    for i in range(bytes_per_frame):
        column_data = bytearray([block[i] if block is not None else 0 for block in group_blocks])

        try:
            decoded_column, _, _ = coder.decode(column_data, erase_pos=erase_pos)
            repaired_columns.append(decoded_column[:data_blocks_num])
        except reedsolo.ReedSolomonError as e:
            raise RuntimeError(f"列 {i} 解码失败: {e}")
    # 利用zip将列数据转置为行数据，再转换为bytes
    return [bytes(bytearray(col)) for col in zip(*repaired_columns)]


def decode_rs(parsed_blocks, total_data_blocks, total_rs_blocks):
    """
    用冗余块进行恢复，先分组，再判断每组的可恢复程度，若都能恢复再进行恢复，否则直接报错吧

    参数：
        parsed_blocks[bytes or None]: 已去重并校验过crc的块表 若缺失则为None 0-total为数据块 total以后为rs块
        total_data_blocks(int): 总数据块数
        total_rs_blocks(int): 总rs块数
    返回：
        data_blocks[bytes]: 没报错的话会直接返回还原的数据块
    """
    # 总体计算有效块数目是否足够
    if total_rs_blocks < parsed_blocks.count(None):
        raise RuntimeError("restore rs: data block is not enough, cannot restore")
    data_blocks = []
    # 分组处理
    rs_group_blocks_num = math.ceil(rs_group_size * rs_factor)
    for i in tqdm(range(0, total_data_blocks, rs_group_size), 
                  desc="按组添加rs块", 
                  total=math.ceil(total_data_blocks / rs_group_size)):
        r = min(i + rs_group_size, total_data_blocks)
        group_data_blocks = parsed_blocks[i : r]
        # 检验该组是否需要还原
        if group_data_blocks.count(None) == 0:
            data_blocks.extend(group_data_blocks)
            continue
        j = i + total_data_blocks
        group_rs_blocks = parsed_blocks[j : j + rs_group_blocks_num]
        group_blocks = group_data_blocks + group_rs_blocks
        # 检验单组有效块数目是否足够
        if len(group_rs_blocks) < group_blocks.count(None):
            raise RuntimeError("restore rs: data block is not enough, cannot restore")
        # 还原
        restored_group = decode_rs_per_group(group_blocks, len(group_data_blocks), len(group_rs_blocks))
        data_blocks.extend(restored_group)

    return data_blocks


def reconstructed_file(blocks, file_name):
    """
    按编号拼接数据，并写入文件

    参数：
        blocks[bytes]: 完整数据块表
        file_name(str): 输出文件名（含后缀）
    """
    if blocks is None or not blocks:
        raise RuntimeError("error in reconstructed_file by blocks")
    
    # 处理末块
    last_block = blocks[-1]
    last_block_length = struct.unpack(">H", last_block[-2:])[0]
    last_block = last_block[:last_block_length]
    blocks[-1] = last_block
    if test_mode:
        print(f"pad length :{last_block_length}")

    try:
        with open(file_name, "wb") as file:
            for block in tqdm(blocks, desc="写入文件", total=len(blocks)):
                file.write(block)
    except Exception as e:
        print(f"error in reconstructed_file while writing file:{str(e)}")


def main():
    import time
    input_file_path = "test/output.mp4"
    output_file_path = "test/decoded_xmu.txt"
    start_time = time.time()

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    decode_start = time.time()
    raw_data_blocks = decode_frames(input_file_path)
    decode_end = time.time()

    parse_start = time.time()
    total_data_blocks, total_rs_blocks, parsed_blocks = parse_blocks(raw_data_blocks)
    parse_end = time.time()

    # 模拟破坏，把第一块置为None
    parsed_blocks[0] = None
    # print(f"parsed_blocks length is :{len(parsed_blocks)}")

    check_start = time.time()
    blocks = check_blocks(parsed_blocks, total_data_blocks, total_rs_blocks)
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