from myqr import qr

bytes_per_frame = (qr.NSIZE - qr.NSYMB) * qr.BLOCK  # 单块数据量
rs_group_size = 200     # rs块分组大小
rs_factor = 0.15        # 冗余率
# 最大0.275
rs_mode = True         # rs
test_mode = False       # 测试模式