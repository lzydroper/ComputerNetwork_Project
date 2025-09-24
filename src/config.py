bytes_per_frame = 1720  # 单块数据量
# 理论最高了，上限是2311，目前塞了2304
rs_group_size = 200     # rs块分组大小
rs_factor = 0.15        # 冗余率
# 最大0.275
rs_mode = True         # rs
test_mode = False       # 测试模式