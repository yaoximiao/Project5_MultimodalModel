import os

data_dir = 'D:/curriculum/AI/Project5/P5_data/data'

max_length = 0  
min_length = float('inf')  # 初始化为无穷大
max_length_file = None  
min_length_file = None  
encodings = ['utf-8', 'gbk', 'latin1', 'iso-8859-1'] 

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        text_path = os.path.join(data_dir, filename)
        for enc in encodings:
            try:    
                with open(text_path, 'r', encoding=enc) as f:
                    text = f.read().strip()
                    length = len(text)
                    if length >= max_length:
                        max_length = length
                        max_length_file = filename
                    if length <= min_length:
                        min_length = length
                        min_length_file = filename
            except UnicodeDecodeError:
                continue

print(f"最大字符串长度: {max_length}, 文件名: {max_length_file}")
print(f"最小字符串长度: {min_length}, 文件名: {min_length_file}")