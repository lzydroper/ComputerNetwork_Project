import math
from PIL import Image, ImageDraw
import numpy as np
from reedsolo import RSCodec
import cv2

class qr():
    # --- 核心参数 ---
    W, H = 192, 108       # 模块数（宽 × 高）
    M = 10                # 每模块像素
    BORDER = 3            # 边框宽度
    # TIMING_LINE_POS = 10  # 时序线位置

    # rs参数
    NSYMB = 8
    NSIZE = 36
    BLOCK = 64

    # --- 颜色与掩码值 ---
    WHITE = 0
    BLACK = 1
    WRITABLE = 0
    PROTECTED = 1
    mask_patterns = [
    lambda y, x: (y + x) % 2 == 0,
    lambda y, x: y % 2 == 0,
    lambda y, x: x % 3 == 0,
    lambda y, x: (y + x) % 3 == 0,
    lambda y, x: (math.floor(y / 2) + math.floor(x / 3)) % 2 == 0,
    lambda y, x: ((y * x) % 2 + (y * x) % 3) == 0,
    lambda y, x: (((y * x) % 2 + (y * x) % 3) % 2) == 0,
    lambda y, x: (((y + x) % 2 + (y * x) % 3) % 2) == 0,
    ]

    # --- 数据网格与保护区 ---
    grid = None           # 数据网格 (0/1)
    mask = None           # 保护区掩码 (0/1)

    # 初始化
    def __init__(self):
        self.generate_empty_grid()
        self.generate_data_mask()

    # --- 编码部分 ---
    def empty_grid(self, fill=WHITE):
        return np.full((self.H, self.W), fill, dtype=np.uint8)
    
    def place_block(self, y0, x0, w, h, value):
        """在模块坐标 (x0,y0) 放置 w*h 的方块 (覆盖)，坐标以模块为单位"""
        self.grid[y0:y0+h, x0:x0+w] = value

    def place_finder_with_separator(self, top_left_y, top_left_x):
        """
        在模块坐标 (top_left_x, top_left_y) 放置 Finder。
        Finder 样式 7x7主体:
        外层 7x7 -> 黑
        内层 5x5 -> 白
        中心 3x3 -> 黑
        另外在 7x7 外侧一圈留白（隔离带）: 1 模块宽（调用方应确保隔离带在边界内）
        """
        # 外层 7x7 黑
        self.place_block(top_left_y, top_left_x, 7, 7, self.BLACK)
        # 中间 5x5 白
        self.place_block(top_left_y + 1, top_left_x + 1, 5, 5, self.WHITE)
        # 中心 3x3 黑
        self.place_block(top_left_y + 2, top_left_x + 2, 3, 3, self.BLACK)

    def generate_data_mask(self):
        """生成可写掩码矩阵 0为可写 1为不可写"""
        self.mask = self.empty_grid(self.WRITABLE)

        # 1. 边框
        self.mask[:self.BORDER, :] = self.PROTECTED
        self.mask[-self.BORDER:, :] = self.PROTECTED
        self.mask[:, :self.BORDER] = self.PROTECTED
        self.mask[:, -self.BORDER:] = self.PROTECTED

        # 2. Finder + 隔离区
        self.mask[self.BORDER:self.BORDER + 8, self.BORDER:self.BORDER + 8] = self.PROTECTED # 左上
        self.mask[self.BORDER:self.BORDER + 8, self.W - self.BORDER - 8:self.W - self.BORDER] = self.PROTECTED # 右上
        self.mask[self.H - self.BORDER - 8:self.H - self.BORDER, self.BORDER:self.BORDER + 8] = self.PROTECTED # 左下
        self.mask[self.H - self.BORDER - 8:self.H - self.BORDER, self.W - self.BORDER - 8:self.W - self.BORDER] = self.PROTECTED # 右下

        # 3. 空白区 31*8
        self.mask[self.BORDER:self.BORDER + 8, self.BORDER + 8:self.BORDER + 8 + 31] = self.PROTECTED

        # 4. 信息区 6*6
        self.mask[self.BORDER + 8:self.BORDER + 8 + 6, self.BORDER:self.BORDER + 6] = self.PROTECTED

        # writable_indices = np.where(self.mask == self.WRITABLE)
        # print(f"可写模块数: {writable_indices[0].size}")

    def generate_empty_grid(self):
        """生成空白二维码 只包含 Finder、分隔带"""
        self.grid = self.empty_grid()

        # 放置 Finder（左上、右上、左下、右下）
        self.place_finder_with_separator(self.BORDER, self.BORDER)
        self.place_finder_with_separator(self.BORDER, self.W - self.BORDER - 7)
        self.place_finder_with_separator(self.H - self.BORDER - 7, self.BORDER)
        self.place_finder_with_separator(self.H - self.BORDER - 7, self.W - self.BORDER - 7)

    def cal_and_apply_mask(self, bits) -> int:
        """计算并应用掩码 返回最佳掩码id"""
        def cal_penalty(grid):
            """计算掩码模式的 penalty score"""
            penalty = 0
            # 规则1: 行/列连续相同颜色
            for row in grid:
                for i in range(len(row) - 4):
                    if len(set(row[i:i+5])) == 1:
                        penalty += 3 + (len(set(row[i:i+6])) == 1) # N1
            for col in grid.T:
                for i in range(len(col) - 4):
                    if len(set(col[i:i+5])) == 1:
                        penalty += 3 + (len(set(col[i:i+6])) == 1) # N1
            # 规则2: 2x2 块
            for y in range(self.H - 1):
                for x in range(self.W - 1):
                    if grid[y, x] == grid[y+1, x] == grid[y, x+1] == grid[y+1, x+1]:
                        penalty += 3 # N2
            # 3.  Finder 模式
            pattern1 = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]) # 黑白黑黑黑白黑 + 4个白
            pattern2 = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]) # 4个白 + 黑白黑黑黑白黑
            # 检查行
            for y in range(self.H):
                for x in range(self.W - 10):
                    if np.array_equal(grid[y, x:x+11], pattern1) or np.array_equal(grid[y, x:x+11], pattern2):
                        penalty += 40
            # 检查列
            for x in range(self.W):
                for y in range(self.H - 10):
                    if np.array_equal(grid[y:y+11, x], pattern1) or np.array_equal(grid[y:y+11, x], pattern2):
                        penalty += 40
            # 规则4: 黑白比例
            black_modules = np.sum(grid)
            total_modules = self.W * self.H
            ratio = black_modules / total_modules
            penalty += int(abs(ratio - 0.5) * 100 / 5) * 10 # N4
            return penalty
        candidates = []
        for mask_id, mask_func in enumerate(self.mask_patterns):
            # 对每个掩码模式，根据规则进行黑白翻转
            tmp_grid = self.grid.copy()
            idx = 0
            for y in range(self.H):
                for x in range(self.W):
                    if self.mask[y, x] == self.WRITABLE:
                        bit = bits[idx]
                        if mask_func(y, x):
                            bit = 1 - bit # flip bit
                        tmp_grid[y, x] = bit
                        idx += 1
            # 计算当前掩码模式的 penalty score
            penalty = cal_penalty(tmp_grid)
            candidates.append((penalty, mask_id, tmp_grid))

        # 选择 penalty 最小的掩码模式
        _, best_mask_id, best_grid = min(candidates, key=lambda x: x[0])
        self.grid = best_grid
        # print(f"best_mask_id: {best_mask_id}")
        return best_mask_id

    def place_info(self, index, total, mask_id):
        """放置信息区 (12:18, 4:10) 共36bit:
        index (16bit) + total (16bit) + mask_id (4bit)
        """
        bits = (
            f"{index:016b}" +
            f"{total:016b}" +
            f"{mask_id:04b}"
        )
        i = 0
        for y in range(self.BORDER + 8, self.BORDER + 8 + 6):
            for x in range(self.BORDER, self.BORDER + 6):
                self.grid[y, x] = int(bits[i])
                i += 1

    def encode_rs(self, data):
        """
        对数据进行rs编码

        :param data: 数据
        :return: 编码后的bit int列表
        """
        # 分成BLOCK个 每个BLOCK中ndata原数据 NSYMB冗余
        # 即原始数据BLOCK*ndata 分成BLOCK份ndata 每份进行一次rs后再拼回到一起
        # 分出BLOCK个ndata
        ndata = self.NSIZE - self.NSYMB
        blocks = [data[i:i+ndata] for i in range(0, self.BLOCK*ndata, ndata)]
        # 每个NSIZE中ndata原数据 NSYMB冗余
        rs = RSCodec(self.NSYMB, self.NSIZE)
        # 对每个NSIZE进行rs编码
        encoded_blocks = [rs.encode(block) for block in blocks]
        # 拼回到一起
        encoded = b''.join(encoded_blocks)
        # 转为bit int列表
        bits = [int(b) for b in ''.join(f'{byte:08b}' for byte in encoded)]
        return bits

    def add_data(self, data, index, total):
        """
        向二维码添加数据 数据块索引从0开始 并rs、掩码
        :param data: 数据
        :param index: 数据块索引 16bit
        :param total: 数据块总数 16bit
        """
        # 先rs
        bits = self.encode_rs(data)
        # 再掩码
        mask_id = self.cal_and_apply_mask(bits)
        # 放置信息区
        self.place_info(index, total, mask_id)

    def make(self, filename: str):
        """
        生成二维码
        """
        img_w = self.W * self.M
        img_h = self.H * self.M

        img = Image.new('1', (img_w, img_h), 1) # '1'表示1位像素，黑白
        draw = ImageDraw.Draw(img)
        # 绘制二维码
        for y in range(self.H):
            for x in range(self.W):
                color = "white" if self.grid[y, x] == self.WHITE else "black"
                shape = [(x * self.M, y * self.M), ((x + 1) * self.M -1, (y + 1) * self.M - 1)]
                draw.rectangle(shape, fill=color)

        img.save(filename)

    def test_mask(self, filename: str):
        """
        测试掩码
        """
        img_w = self.W * self.M
        img_h = self.H * self.M

        img = Image.new('1', (img_w, img_h), 1) # '1'表示1位像素，黑白
        draw = ImageDraw.Draw(img)
        # 绘制二维码
        for y in range(self.H):
            for x in range(self.W):
                color = "white" if self.mask[y, x] == self.WRITABLE else "black"
                shape = [(x * self.M, y * self.M), ((x + 1) * self.M -1, (y + 1) * self.M - 1)]
                draw.rectangle(shape, fill=color)

        img.save(filename)

    # --- 解码部分 ---
    def _find_finders(self, image):
        """使用OpenCV在图像中寻找三个定位符"""

        # 2. 寻找轮廓
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"finders contours: {len(contours)}")

        # 3. 筛选定位符
        finder_candidates = []
        for i, contour in enumerate(contours):
            # 一个定位符有两层子轮廓
            # hierarchy[i] -> [Next, Previous, First_Child, Parent]
            child_idx = hierarchy[0][i][2]
            if child_idx == -1: continue
            grandchild_idx = hierarchy[0][child_idx][2]
            if grandchild_idx == -1: continue

            # 确保它没有兄弟轮廓，即它是唯一的子轮廓
            if hierarchy[0][child_idx][0] == -1 and hierarchy[0][child_idx][1] == -1:
                 if hierarchy[0][grandchild_idx][0] == -1 and hierarchy[0][grandchild_idx][1] == -1:
                    finder_candidates.append(contour)
        
        if len(finder_candidates) < 4:
            raise Exception(f"Could not find 4 finder patterns, found {len(finder_candidates)}, candidates: {finder_candidates}")

        # 计算每个候选定位符的中心点
        centers = []
        for contour in finder_candidates:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append(np.array([cx, cy]))
        
        if len(centers) < 4:
             raise Exception("Could not calculate centers for 4 finder patterns.")

        # 4. 识别左上、右上、左下角
        centers = np.array(centers)
        centroid = np.mean(centers, axis=0)
        cx, cy = centroid[0], centroid[1]
        tl, tr, bl, br = None, None, None, None
        # 左上角: x+y 最小
        tl = centers[np.argmin(np.sum(centers, axis=1))]
        # 右下角: x+y 最大
        br = centers[np.argmax(np.sum(centers, axis=1))]
        # 右上角: x-y 最大
        bl = centers[np.argmax(np.diff(centers, axis=1))]
        # 左下角: x-y 最小
        tr = centers[np.argmin(np.diff(centers, axis=1))]
        if any(p is None for p in [tl, tr, bl, br]):
            raise Exception("Could not classify all four corner points relative to the centroid.")
        
        # print(f"[Decoder-Debug] Finder Centers: tl={tl}, tr={tr}, bl={bl}, br={br}")
        return tl, tr, bl, br
    
    def _warp_and_sample_grid(self, image, finder_centers):
        """根据四个定位点进行精确的透视变换，并采样整个模块网格"""
        target_w, target_h = self.W * self.M, self.H * self.M
        tl, tr, bl, br = finder_centers
        src_pts = np.float32([tl, tr, br, bl])
        dst_tl_coord = (self.BORDER + 3.5, self.BORDER + 3.5)
        dst_tr_coord = (self.W - self.BORDER - 3.5, self.BORDER + 3.5)
        dst_br_coord = (self.W - self.BORDER - 3.5, self.H - self.BORDER - 3.5)
        dst_bl_coord = (self.BORDER + 3.5, self.H - self.BORDER - 3.5)

        # 将理论模块坐标转换为理论像素坐标
        dst_pts = np.float32([
            [dst_tl_coord[0] * self.M, dst_tl_coord[1] * self.M],
            [dst_tr_coord[0] * self.M, dst_tr_coord[1] * self.M],
            [dst_br_coord[0] * self.M, dst_br_coord[1] * self.M],
            [dst_bl_coord[0] * self.M, dst_bl_coord[1] * self.M]
        ])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (target_w, target_h))
        _, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # debug_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        # for center in dst_pts:
        #     cv2.circle(debug_img, (int(center[0]), int(center[1])), 10, (0,0,255), 2)
        # cv2.imwrite("debug_think_finders.png", debug_img)
        # """根据定位点进行透视变换，并采样整个模块网格"""
        # tl, tr, bl = finder_centers

        # # 目标点坐标 (在理想的、未旋转的图像中)
        # # 我们使用定位点中心的模块坐标
        # dst_tl = (self.BORDER + 3.5, self.BORDER + 3.5)
        # dst_tr = (self.W - self.BORDER - 3.5, self.BORDER + 3.5)
        # dst_bl = (self.BORDER + 3.5, self.H - self.BORDER - 3.5)
        
        # # 为了进行透视变换，需要第四个点。我们可以通过向量计算得出右下角
        # # (tr - tl) 是顶部向量, (bl - tl) 是左侧向量
        # # br_estimated = tl + (tr - tl) + (bl - tl)
        # br_estimated = tr + bl - tl
        # dst_br = (self.W - self.BORDER - 3.5, self.H - self.BORDER - 3.5)

        # # 定义源点和目标点
        # src_pts = np.float32([tl, tr, bl, br_estimated])
        # # 目标图像尺寸，为了采样精度，可以适当放大
        # target_res = max(self.W, self.H) * self.M
        # dst_pts = np.float32([dst_tl, dst_tr, dst_bl, dst_br]) * (target_res / max(self.W, self.H))
        
        # # 计算透视变换矩阵并应用
        # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warped = cv2.warpPerspective(image, matrix, (target_res, int(target_res * self.H/self.W)))

        # debug_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        # for center in dst_pts:
        #     cv2.circle(debug_img, (int(center[0]), int(center[1])), 10, (0,0,255), 2)
        # cv2.imwrite("debug_think_finders.png", debug_img)

        # 采样网格
        grid = np.zeros((self.H, self.W), dtype=np.uint8)
        module_h = warped.shape[0] / self.H
        module_w = warped.shape[1] / self.W

        for y in range(self.H):
            for x in range(self.W):
                # 在每个模块的中心区域采样，避免边缘噪声
                roi_y_start = int((y + 0.25) * module_h)
                roi_y_end = int((y + 0.75) * module_h)
                roi_x_start = int((x + 0.25) * module_w)
                roi_x_end = int((x + 0.75) * module_w)
                
                roi = warped[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                avg_val = np.mean(roi)
                
                # 低于阈值的是黑色 (1)，否则是白色 (0)
                grid[y, x] = self.BLACK if avg_val > 128 else self.WHITE
                
        return grid, warped
    
    def _read_info_area(self, grid):
        """从网格中读取信息区 (index, total, mask_id)"""
        bits = []
        for y in range(self.BORDER + 8, self.BORDER + 8 + 6):
            for x in range(self.BORDER, self.BORDER + 6):
                bits.append(str(grid[y, x]))
        
        bit_str = "".join(bits)
        
        index = int(bit_str[0:16], 2)
        total = int(bit_str[16:32], 2)
        mask_id = int(bit_str[32:36], 2)
        # print(f"index: {index}, total: {total}, mask_id: {mask_id}")
        
        return index, total, mask_id

    def _extract_and_unmask_data(self, grid, mask_id):
        """提取数据区的比特，并应用反掩码操作"""
        mask_func = self.mask_patterns[mask_id]
        data_bits = []
        
        for y in range(self.H):
            for x in range(self.W):
                if self.mask[y, x] == self.WRITABLE:
                    bit = grid[y, x]
                    # 反向掩码：如果掩码函数为 True，则翻转比特
                    if mask_func(y, x):
                        bit = 1 - bit
                    data_bits.append(bit)
                    
        return data_bits

    def _decode_rs(self, bits):
        byte_list = []
        for i in range(0, len(bits), 8):
            byte_str = "".join(map(str, bits[i:i+8]))
            if len(byte_str) < 8: continue # 忽略末尾不足8位的比特
            byte_list.append(int(byte_str, 2))
        encoded_data = bytes(byte_list)

        blocks = [encoded_data[i:i+self.NSIZE] for i in range(0, len(encoded_data), self.NSIZE)]
        rs = RSCodec(self.NSYMB, self.NSIZE)
        decoded_blocks = []
        error_blocks = 0
        corre_blocks = 0
        total_blocks = 0
        for i, block in enumerate(blocks):
            if len(block) < self.NSIZE: continue
            total_blocks += 1
            try:
                result = rs.decode(block)
                if result[2]:
                    corre_blocks += 1
                decoded_blocks.append(result[0])
            except Exception:
                error_blocks += 1
                # 填充错误块为空数据或标记，避免数据错位
                decoded_blocks.append(b'\x00' * (self.NSIZE - self.NSYMB))

        return b"".join(decoded_blocks), error_blocks, corre_blocks
    
    def decode(self, image_path: str, workspace: str, debug = False):
        """
        解码主函数：从图像文件完整解码出数据。
        
        :param image_path: 包含二维码的图像文件路径
        :return: 一个元组 (data, index, total)，其中 data 是原始的 bytes 数据
        """
        import os
        frame_index = os.path.basename(image_path).split("_")[1].split(".")[0]
        # 1. 加载并预处理图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image from {image_path}")
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        if debug:
            cv2.imwrite(f"{workspace}/debug_{frame_index}_1_blurred.png", blurred)
        # 2. 二值化图像，使其只有纯黑和纯白，便于轮廓检测
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        # _, binary_image = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
        if debug:
            cv2.imwrite(f"{workspace}/debug_{frame_index}_2_binary.png", binary_image)
        # 3. 形态学操作清理噪点 (非常重要!)
        # 开运算：先腐蚀后膨胀，可以移除小的白色噪点
        # 闭运算：先膨胀后腐蚀，可以填充内部小的黑色空洞
        kernel = np.ones((3,3),np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        if debug:
            cv2.imwrite(f"{workspace}/debug_{frame_index}_3_cleaned.png", cleaned)
        # 4. 找到四个定位点
        # finder_centers = self._find_finders(cleaned)
        # debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # for center in finder_centers:
        #     cv2.circle(debug_img, (int(center[0]), int(center[1])), 10, (0,0,255), 2)
        # cv2.imwrite("debug_4_finders.png", debug_img)
        try:
            finder_centers = self._find_finders(cleaned)
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # 如果成功找到4个点，将它们画出来
            for center in finder_centers:
                cv2.circle(debug_img, (tuple(center.astype(int))), 25, (0, 0, 255), 5) # 用元组形式
            if debug:
                cv2.imwrite(f"{workspace}/debug_{frame_index}_4_finders_SUCCESS.png", debug_img)
        except Exception as e:
            if debug:
                # print(f"Finder detection failed: {e}")
                # --- 可视化调试代码 ---
                # 异常发生时，我们手动重新运行查找逻辑，并把找到的画出来
                contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                finder_candidates = []
                for i, contour in enumerate(contours):
                    child_idx = hierarchy[0][i][2]
                    if child_idx == -1: continue
                    grandchild_idx = hierarchy[0][child_idx][2]
                    if grandchild_idx == -1: continue
                    if hierarchy[0][child_idx][0] == -1 and hierarchy[0][child_idx][1] == -1 and \
                    hierarchy[0][grandchild_idx][0] == -1 and hierarchy[0][grandchild_idx][1] == -1:
                        finder_candidates.append(contour)

                # 在原始图像上把找到的候选轮廓（比如3个）用绿色画出来
                debug_fail_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_fail_img, finder_candidates, -1, (0, 255, 0), 3)
                cv2.imwrite(f"{workspace}/debug_{frame_index}_4_finders_FAILED_candidates.png", debug_fail_img)
                if len(finder_candidates) > 0:
                    raise RuntimeError(f"finder detection failed, only found {len(finder_candidates)} candidates")
                else:
                    raise RuntimeError(f"it may be an empty frame.")
                # print("已生成 debug_4_finders_FAILED_candidates.png 以显示找到的候选轮廓。")
            # --- 调试代码结束 ---
            return # 提前退出函数
        # 3. 图像校正并采样网格数据
        # print("Step 2: Warping image and sampling grid...")
        grid, warped = self._warp_and_sample_grid(blurred, finder_centers)
        if debug:
            cv2.imwrite(f"{workspace}/debug_{frame_index}_5_warped.png", warped)
        if debug:
            sampled_viz = np.kron(grid, np.ones((10,10), dtype=np.uint8)) * 255
            cv2.imwrite(f"{workspace}/debug_{frame_index}_6_sampled_grid.png", sampled_viz)
        # print("Grid sampling complete.")
        
        # 4. 读取信息区
        index, total, mask_id = self._read_info_area(grid)
        if index < 0 or total <= 0 or mask_id < 0 or mask_id >= 8:
            raise ValueError(f"Invalid decoded info: Index={index}, Total={total}, MaskID={mask_id}")
        
        # 5. 提取数据并反掩码
        # print("Step 4: Extracting data and applying unmask...")
        data_bits = self._extract_and_unmask_data(grid, mask_id)
        # print(f"Extracted {len(data_bits)} data bits.")
        
        # 6. RS解码
        # print("Step 5: Reed-Solomon decoding...")
        data, error_blocks, corre_blocks = self._decode_rs(data_bits)
        if debug:
            import logging
            logging.basicConfig(filename=f"{workspace}/debug_decoding.log", level=logging.DEBUG, format='%(message)s')
            log_msg = f"{frame_index} whitch index is {index} Total blocks: {self.BLOCK}, \
              Failed blocks: {error_blocks} ({(error_blocks/self.BLOCK*100):.2f}%)\
              Corrected blocks: {corre_blocks} ({(corre_blocks/self.BLOCK*100):.2f}%)"
            logging.debug(log_msg)
        
        return data, index, total, error_blocks
    

def encode_test(index = 1, total = 10, output_img="qr_encoded.png", output_data="original_data.bin"):
    """
    生成随机数据并编码为QR码
    :param index: 当前QR码索引
    :param total: 总QR码数量
    :param data_size: 随机数据大小(字节)
    :param output_img: 生成的QR码图片路径
    :param output_data: 原始数据保存路径
    :return: 生成的文件路径元组 (图片路径, 数据路径)
    """
    import os
    # 创建QR实例
    my_qr = qr()
    # my_qr.test_mask("test.png")
    
    # 生成随机数据并保存
    data_size = (qr.NSIZE - qr.NSYMB) * qr.BLOCK
    sample_data = os.urandom(data_size)
    with open(output_data, 'wb') as f:
        f.write(sample_data)
    print(f"已生成原始数据并保存至: {output_data}")
    
    # 生成QR码
    my_qr.add_data(sample_data, index=index, total=total)
    my_qr.make(output_img)
    print(f"已生成QR码图片并保存至: {output_img}")
    
    return output_img, output_data

def decode_test(input_img, original_data_path):
    """
    解码QR码并与原始数据比对
    :param input_img: 待解码的QR码图片路径
    :param original_data_path: 原始数据文件路径
    :return: 验证结果 (布尔值)
    """
    # 创建QR实例
    my_qr = qr()
    
    # 解码QR码
    decoded_data, decoded_index, decoded_total = my_qr.decode(input_img)
    print(f"解码完成 - 索引: {decoded_index}/{decoded_total}")
    
    # 读取原始数据
    with open(original_data_path, 'rb') as f:
        original_data = f.read()
    
    # 验证结果
    print("\n--- 验证结果 ---")
    is_data_match = (decoded_data == original_data)
    print(f"数据一致性: {'✅ 匹配' if is_data_match else '❌ 不匹配'}")


if __name__ == '__main__':
    qr_index = 1
    qr_total = 10
    img_path = f"qr_part_{qr_index}_of_{qr_total}.png"
    data_path = f"original_data_{qr_index}_of_{qr_total}.bin"

    # encode_test(index=qr_index, total=qr_total, output_img=img_path, output_data=data_path)

    to_decode_img = img_path
    to_decode_img = "to_decode_img.jpg"
    decode_test(to_decode_img, data_path)
