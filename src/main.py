import os
import shutil
import time
from datetime import datetime
import encoder as en
import decoder as de



"""
    莫名其妙的tkinter会有找不到tcl的问题 我目前是直接把tcl的文件放在.venv下 如果你们也有问题可以尝试这么做
    或者在最上面取消注释下面的代码 并把路径改成自己的Python的tcl所在位置
"""
# import os
# os.environ['TCL_LIBRARY'] = r"D:\.Software\Python\tcl\tcl8.6"
# os.environ['TK_LIBRARY'] = r"D:\.Software\Python\tcl\tk8.6"

def select_file():
    """
    弹出文件选择窗口
    """
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择文件",
        initialdir="./workspace",
        filetypes=[("所有文件", "*.*")]
    )
    if file_path:
        return file_path
    else:
        print("error: select nothing or cancel")
        exit(1)


def encode(args):
    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)

    file_path = select_file()
    if args.t:
        start_time = time.time()
        read_start = time.time()
    
    raw_data_blocks, total = en.read_and_divide(file_path)
    if args.t:
        read_end = time.time()
        rs_start = time.time()

    blocks = en.encode_rs(raw_data_blocks)
    if args.t:
        rs_end = time.time()
    #     header_start = time.time()

    # headed_blocks = en.add_header(blocks, total)
    # if args.t:
    #     header_end = time.time()
        qr_start = time.time()

    frames_file_path = workspace + "/frames_encode"
    if os.path.exists(frames_file_path):
        shutil.rmtree(frames_file_path)
    os.makedirs(frames_file_path, exist_ok=True)
    image_paths = en.generate_qr_sequence(blocks, total, frames_file_path)
    if args.t:
        qr_end = time.time()
        video_start = time.time()

    output_file_path = workspace + "/ouput"
    os.makedirs(output_file_path, exist_ok=True)
    output_file = f"{output_file_path}/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.mp4"
    en.images_to_video(image_paths, output_file, args.fps, args.repeat)
    if args.t:
        video_end = time.time()
        end_time = time.time()
        total_time = end_time - start_time
        print()
        print(f"文件读取: {read_end - read_start:.2f} 秒")
        print(f"添加rs块: {rs_end - rs_start:.2f} 秒")
        # print(f"添加块头: {header_end - header_start:.2f} 秒")
        print(f"生成图片: {qr_end - qr_start:.2f} 秒")
        print(f"生成视频: {video_end - video_start:.2f} 秒")
        print(f"总计耗时: {total_time:.2f} 秒")
        print(f"编码速度: {en.cal_speed(file_path, image_paths):.2f} kbps")
    
    print(f"\n程序运行完成, 文件输出至“{output_file}”")


def decode(args):
    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)
    file_path = select_file()
    if args.one:
        import myqr
        qr = myqr.qr()
        qr.decode(file_path, workspace, debug=True)
    else:
        if args.t:
            start_time = time.time()
            read_start = time.time()
        frames_file_path = workspace + "/frames_decode"
        if os.path.exists(frames_file_path):
            shutil.rmtree(frames_file_path)
        os.makedirs(frames_file_path, exist_ok=True)
        image_paths = de.read_and_divide(file_path, frames_file_path)
        if args.t:
            read_end = time.time()
            decode_start = time.time()
        
        total_data_blocks, total_rs_blocks, parsed_blocks = de.decode_frames(image_paths, frames_file_path, debug=args.d)
        if args.t:
            decode_end = time.time()
            check_start = time.time()
        
        blocks = de.check_blocks(parsed_blocks, total_data_blocks, total_rs_blocks)
        if blocks is None:
            print("数据块不完整，无法重建文件")
            return
        if args.t:
            check_end = time.time()
            reconstruct_start = time.time()
        
        output_file_path = workspace + "/ouput"
        os.makedirs(output_file_path, exist_ok=True)
        output_file = f"{output_file_path}/decoded_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.bin"
        try:
            de.reconstructed_file(blocks, output_file)
        except:
            print("数据块不完整，无法重建文件")
            return
        if args.t:
            reconstruct_end = time.time()
            end_time = time.time()
            total_time = end_time - start_time
            print()
            print(f"文件读取: {read_end - read_start:.2f} 秒")
            print(f"解码二维码: {decode_end - decode_start:.2f} 秒")
            print(f"检查数据块: {check_end - check_start:.2f} 秒")
            print(f"重建文件: {reconstruct_end - reconstruct_start:.2f} 秒")
            print(f"总计耗时: {total_time:.2f} 秒")
        
        print(f"\n程序运行完成, 文件输出至“{output_file}”")



def main():
    import argparse as ap
    pa = ap.ArgumentParser(description="文件/二维码视频互转")
    spa = pa.add_subparsers(dest="mode", required=True, 
                            help="操作模式(encode/decode)")
    # encoder arg
    pa_en = spa.add_parser("encode", aliases="e", 
                           help="文件编码为二维码视频")
    pa_en.add_argument("--fps", type=int, default=60, choices=[30, 60],
                       help="帧率(默认60)")
    pa_en.add_argument("--repeat", type=int, default=4, choices=range(2,60),
                       help="重复次数(默认4)")
    
    # decoder arg
    pa_de = spa.add_parser("decode", aliases="d", 
                           help="二维码视频解码为文件")
    pa_de.add_argument("--rs", type=bool, default=True, 
                       help="是否添加恢复信息(默认True)")
    pa_de.add_argument("--one", type=bool, default=False,
                       help="是否只解码一帧(默认False)")
    
    # public
    for p in [pa_en, pa_de]:
        p.add_argument('-t', action='store_true', default=False,
                      help='开启计时功能(默认：关闭)')
        p.add_argument('-d', action='store_true', default=False,
                      help='开启调试模式(默认：关闭)')
        
    # handle
    args = pa.parse_args()
    if args.mode in ["encode", "e"]:
        print("en")
        encode(args)
    elif args.mode in ["decode", "d"]:
        print("de")
        decode(args)


if __name__ == "__main__":
    main()