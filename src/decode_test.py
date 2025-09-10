from pyzbar.pyzbar import decode


def decode_frames():
    """读取视频每一帧并直接解码保存为data_blocks"""


def parse_header():
    """解析块头"""


def parse_blocks():
    """解析块并验证crc 按编号保留首个crc通过的块 分离出rs恢复块(即index从total开始的所有块)"""


def check_blocks():
    """遍历查询缺失块，若可能则进行恢复，若无法恢复，返回缺失块数量或索引列表"""


def restore_global_rs():
    """用冗余块进行恢复"""


def reconstructed_file():
    """按编号拼接数据，并写入文件"""