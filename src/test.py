import cv2
import struct
import zlib
from pyzbar.pyzbar import decode
import os

def parse_header(file_path):
    img = cv2.imread(file_path)
    raw_block = decode(img)[0].data
    index, total = struct.unpack(">HH", raw_block[0:4])
    crc = struct.unpack(">I", raw_block[4:8])[0]
    block = raw_block[8:]
    print(f"index is : {index}, total is : {total}, crc is : {crc}")
    # return index, total, crc, block

if __name__ == "__main__":
    file_path = "test/frames_encode/frame_00001.png"
    parse_header(file_path)