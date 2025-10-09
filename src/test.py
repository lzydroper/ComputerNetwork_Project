# test_rs.py

from reedsolo import RSCodec, ReedSolomonError
import os
import numpy as np

# --- 1. 定义与您主程序一致的RS参数 ---
NSIZE = 36  # 总块长 (字节)
NSYMB = 4   # 纠错码长度 (字节)
NDATA = NSIZE - NSYMB # 数据长度 (字节)

print(f"--- Reed-Solomon Codec Test ---")
print(f"Block Total Size (NSIZE): {NSIZE} bytes")
print(f"ECC Size (NSYMB): {NSYMB} bytes")
print(f"Data Size (NDATA): {NDATA} bytes")
print(f"Correction Capability: {NSYMB // 2} bytes per block")
print("-" * 35)

# 初始化 RS 编解码器
rs = RSCodec(NSYMB, NSIZE)

# --- 2. 生成一段随机的原始数据 ---
# os.urandom(NDATA) 生成的是 bytes, RSCodec 需要 bytearray 或 list of ints
original_data = bytearray(os.urandom(NDATA))
print(f"Original Data (first 10 bytes): {list(original_data[:10])}...")
print(f"Original Data Length: {len(original_data)} bytes")
print("-" * 35)


# --- 3. 编码 ---
try:
    encoded_data = rs.encode(original_data)
    print("✅ Encoding successful.")
    print(f"Encoded Data Block Length: {len(encoded_data)} bytes (should be {NSIZE})")
    # 编码后的数据是一个 bytearray
    # print(f"Encoded Data (full): {list(encoded_data)}")
    print("-" * 35)
except Exception as e:
    print(f"❌ Encoding failed: {e}")
    exit()

# ===================================================================
# --- 4. 测试 1: 解码完美无损的数据 ---
# ===================================================================
print("\n>>> Test 1: Decoding a perfect, undamaged block...")
try:
    result_perfect = rs.decode(encoded_data)
    
    print(f"✅ Decoding successful.")
    print(f"   - Return value type: {type(result_perfect)}")
    print(f"   - Number of returned values: {len(result_perfect)}")
    
    if len(result_perfect) == 3:
        print("   - Result: As expected, returned 2 values (data, ecc).")
        decoded_data, _, err_pos = result_perfect
        # 验证数据是否一致
        if decoded_data == original_data:
            print("   - Data verification: PASSED. Decoded data matches original.")
        else:
            print("   - Data verification: FAILED. Decoded data does not match original.")
    else:
        print(f"   - Result: UNEXPECTED! Returned {len(result_perfect)} values instead of 2.")

except ReedSolomonError as e:
    print(f"❌ Decoding failed unexpectedly: {e}")
print("-" * 35)

# ===================================================================
# --- 5. 测试 2: 解码可修复的损坏数据 ---
# ===================================================================
print("\n>>> Test 2: Decoding a block with 2 correctable byte errors...")
# 创建一个副本以进行修改
corrupted_data_correctable = bytearray(encoded_data)

# 手动损坏 2 个字节
error_positions_inflicted = [5, 20]
corrupted_data_correctable[error_positions_inflicted[0]] = 0xAA  # 改变第 5 个字节
corrupted_data_correctable[error_positions_inflicted[1]] = 0xBB  # 改变第 20 个字节
print(f"   - Inflicted errors at byte positions: {error_positions_inflicted}")

try:
    result_correctable = rs.decode(corrupted_data_correctable)

    print(f"✅ Decoding successful.")
    print(f"   - Return value type: {type(result_correctable)}")
    print(f"   - Number of returned values: {len(result_correctable)}")

    if len(result_correctable) == 3:
        print("   - Result: As expected, returned 3 values (data, ecc, errata_pos).")
        decoded_data, _, errata_pos = result_correctable
        print(f"   - Detected error positions (errata_pos): {list(errata_pos)}")
        # 验证修复的位置是否正确
        if sorted(list(errata_pos)) == sorted(error_positions_inflicted):
             print("   - Error position verification: PASSED.")
        else:
             print("   - Error position verification: FAILED.")
        # 验证数据是否一致
        if decoded_data == original_data:
            print("   - Data verification: PASSED. Decoded data matches original.")
        else:
            print("   - Data verification: FAILED.")
    else:
        print(f"   - Result: UNEXPECTED! Returned {len(result_correctable)} values instead of 3.")
        
except ReedSolomonError as e:
    print(f"❌ Decoding failed unexpectedly: {e}")
print("-" * 35)

# ===================================================================
# --- 6. 测试 3: 解码不可修复的损坏数据 ---
# ===================================================================
print("\n>>> Test 3: Decoding a block with 3 uncorrectable byte errors...")
# 创建一个副本
corrupted_data_uncorrectable = bytearray(encoded_data)

# 手动损坏 3 个字节
error_positions_inflicted_3 = [2, 15, 30]
corrupted_data_uncorrectable[error_positions_inflicted_3[0]] = 0x11
corrupted_data_uncorrectable[error_positions_inflicted_3[1]] = 0x22
corrupted_data_uncorrectable[error_positions_inflicted_3[2]] = 0x33
print(f"   - Inflicted errors at byte positions: {error_positions_inflicted_3}")

try:
    rs.decode(corrupted_data_uncorrectable)
    print("❌ Decoding SUCCEEDED UNEXPECTEDLY. This should have failed.")

except ReedSolomonError as e:
    print(f"✅ Decoding failed as expected.")
    print(f"   - Caught exception: {e}")
print("-" * 35)