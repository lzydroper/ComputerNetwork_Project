import cv2
import struct
import zlib
import os
from tqdm import tqdm
import reedsolo
import config as cg
import math
import numpy as np
from myqr import qr


""""""
# 大致流程如下：[写入 <- 还原（最后一块补长还原）] <- rs <- 块头 <- qr <- 视频
""""""
# 报错信息集合
messages = []


def read_and_divide(input_file_path, output_dir):
    """
    读取视频文件的每一帧，并将其保存到指定目录，返回所有帧的路径列表

    参数：
        file_path(str): 待读取的视频文件路径
        output_dir(str): 保存每一帧图片的目录
    返回：
        list[str]: 所有帧的路径列表
    """
    file_paths = []
    cap = cv2.VideoCapture(input_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="读视频帧")
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        file_paths.append(frame_path)
        pbar.update(1)
    pbar.close()
    cap.release()
    return file_paths


def decode_frames(file_paths, workspace, rs_factor=cg.rs_factor, debug=False):
    """
    读取图片并直接解码保存为raw_data_blocks

    参数：
        file_paths(list[str]): 待解码的图片路径列表
        workspace(str): 工作目录
        rs_factor(double): 冗余率
        debug(bool): 是否开启调试模式
    返回：
        tuple(int, int, list[bytes]): 二维码读取出来的数据块总数、rs块总数、数据块列表
    """
    total_data_blocks = -1
    total_rs_blocks = -1
    parsed_blocks = []
    qr_decoder = qr()
    # 捕获异常并记录到日志文件
    import logging
    logging.basicConfig(filename=f"{workspace}/debug_decoding.log", level=logging.DEBUG, format='%(message)s')
    errors = 0
    for frame_index in tqdm(range(len(file_paths)), desc="解二维码"):
        try:
            data, index, total, error_blocks = qr_decoder.decode(file_paths[frame_index], workspace, debug)
            if error_blocks > 0:
                errors += error_blocks
                raise RuntimeError(f"Reed-Solomon decoding failed: {error_blocks} errors corrected.")
            # print(f"index is :{index}, total is :{total}")
            if total_data_blocks == -1:
                total_data_blocks = total
                total_rs_blocks = math.ceil(total_data_blocks * rs_factor)
                parsed_blocks = [None] * (total_data_blocks + total_rs_blocks)
            if parsed_blocks[index] is not None:
                continue
            parsed_blocks[index] = data
        except Exception as e:
            if debug:
                log_msg = f"{frame_index:06d} fail for reason: {e}"
                logging.debug(log_msg)
            continue
    return total_data_blocks, total_rs_blocks, parsed_blocks


def check_blocks(parsed_blocks, total_data_blocks, total_rs_blocks):
    """
    遍历查询缺失块，若可能则进行恢复，若无法恢复，返回缺失块数量或索引列表
    
    参数：
        parsed_blocks[bytes or None]: 块表 若缺失则为None 0-total为数据块 total以后为rs块
        total_data_blocks(int): 总数据块数
        total_rs_blocks(int): 总rs块数
    返回：
        data_blocks[bytes] or None: 完整数据块表 缺失返回None
    """
    # print(f"len(parsed_blocks) is :{len(parsed_blocks)}, total_data_blocks is :{total_data_blocks}, total_rs_blocks is :{total_rs_blocks}")
    if parsed_blocks is None or total_data_blocks == -1 or total_rs_blocks == -1:
        raise RuntimeError("parsed_blocks is None or total_data_blocks is -1")
    parsed_data_blocks = parsed_blocks[:total_data_blocks]
    missing = [i for i, block in enumerate(parsed_data_blocks) if block is None]
    if len(missing) > 0:
        messages.append(f"检测到 {len(missing)} 个数据块丢失，尝试使用Reed-Solomon码进行恢复。")
        messages.append(f"缺失块索引: {missing}")
        # print(f"need blocks :{total_data_blocks}, but find missing block :{len(missing)}")
        # print(f"missing block index is :{missing}")
        return decode_rs(parsed_blocks, total_data_blocks, total_rs_blocks)
    else:
        return parsed_data_blocks.copy()


def decode_rs_per_group(group_blocks, data_blocks_num, rs_blocks_num, bytes_per_frame=cg.bytes_per_frame):
    """
    对单子块进行rs解码还原

    参数：
        group_blocks[bytes or None]: 该组块 前半部分为数据块 后半部分为rs块
        data_blocks_num(int): 原始数据块数目
        rs_blocks_num(int): 冗余块数目
        bytes_per_frame(int): 每块字节数
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
            messages.append(f"警告: 组内一列数据解码失败，该组数据将丢失。")
            return [None] * data_blocks_num
            # raise RuntimeError(f"列 {i} 解码失败: {e}")
    # 利用zip将列数据转置为行数据，再转换为bytes
    return [bytes(bytearray(col)) for col in zip(*repaired_columns)]


def decode_rs(parsed_blocks, total_data_blocks, total_rs_blocks, rs_group_size=cg.rs_group_size, rs_factor=cg.rs_factor):
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
        needed = parsed_blocks.count(None) - total_rs_blocks
        messages.append(f"警告: 整体冗余块不足，无法完全恢复。缺失: {parsed_blocks.count(None)}, 冗余: {total_rs_blocks}。至少还需要 {needed} 个块。")
        # print(f"restore rs: data block is not enough, cannot restore, still need {needed}")
        # exit(1)
    data_blocks = []
    # 分组处理
    rs_group_blocks_num = math.ceil(rs_group_size * rs_factor)
    pbar = tqdm(range(0, total_data_blocks, rs_group_size), 
                desc="翻译rs块", 
                total=math.ceil(total_data_blocks / rs_group_size))
    # for i in tqdm(range(0, total_data_blocks, rs_group_size), 
    #               desc="翻译rs块", 
    #               total=math.ceil(total_data_blocks / rs_group_size)):
    for i in pbar:
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
            pbar.set_postfix_str(f"组 {i} 无法恢复 (冗余不足)")
            messages.append(f"警告: 第 {i} 组数据无法恢复，因冗余块不足。")
            data_blocks.extend([None] * len(group_data_blocks))
            continue
            # raise RuntimeError("restore rs: data block is not enough, cannot restore")
        # 还原
        restored_group = decode_rs_per_group(group_blocks, len(group_data_blocks), len(group_rs_blocks))
        if restored_group and restored_group[0] is None:
            pbar.set_postfix_str(f"组 {i} 恢复失败 (解码错误)")
        data_blocks.extend(restored_group)

    return data_blocks


def reconstructed_file(blocks, file_name, bytes_per_frame=cg.bytes_per_frame):
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
    if last_block is not None:
        try:
            last_block_length = struct.unpack(">H", last_block[-2:])[0]
            last_block = last_block[:last_block_length]
            blocks[-1] = last_block
        except (struct.error, IndexError):
            # --- MODIFICATION START ---
            messages.append("警告: 无法解析最后一个数据块的长度，将使用其完整大小。")
    # last_block_length = struct.unpack(">H", last_block[-2:])[0]
    # last_block = last_block[:last_block_length]
    # blocks[-1] = last_block

    null_block = b'\x00' * bytes_per_frame
    lost_blocks_count = 0

    try:
        with open(file_name, "wb") as file:
            for block in tqdm(blocks, desc="写入文件", total=len(blocks)):
                if block is not None:
                    file.write(block)
                else:
                    file.write(null_block)
                    lost_blocks_count += 1
                # file.write(block)
    except Exception as e:
        print(f"error in reconstructed_file while writing file:{str(e)}")
    
    if lost_blocks_count > 0:
        # --- MODIFICATION START ---
        messages.append(f"文件生成完成，但有 {lost_blocks_count} 个数据块丢失并被零字节填充。文件可能已损坏。")
        # --- MODIFICATION END ---
    else:
        messages.append("文件成功生成，所有数据块均已恢复。")
    
    return messages