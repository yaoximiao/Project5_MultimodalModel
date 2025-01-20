from PIL import Image
import os

# 数据集路径
data_dir = 'D:/curriculum/AI/Project5/P5_data/data'

# 初始化变量
max_width = 0  # 最大宽度
max_height = 0  # 最大高度
min_width = float('inf')  # 最小宽度
min_height = float('inf')  # 最小高度

# 遍历数据集中的所有图片
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        # 加载图像
        image_path = os.path.join(data_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # 更新最大宽度和高度
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
            
            # 更新最小宽度和高度
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
        except Exception as e:
            print(f"无法加载图像 {filename}: {e}")

# 输出结果
print(f"最大宽度: {max_width}")
print(f"最大高度: {max_height}")
print(f"最小宽度: {min_width}")
print(f"最小高度: {min_height}")