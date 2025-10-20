from myqr import qr
# 参数配置，其实没什么用，理论上应该用什么别的方式，但我尚且不会，而且这个改成用命令参数解析了
bytes_per_frame = (qr.NSIZE - qr.NSYMB) * qr.BLOCK  # 单块数据量
rs_group_size = 200     # rs块分组大小
rs_factor = 0.15        # 冗余率
# 最大0.275     <--     改用自己的以后没有最大，可以自己随便拉大
rs_mode = True         # rs
test_mode = False       # 测试模式