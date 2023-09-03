# 原图像和标注所在文件夹
origin_image_mask = r"D:\liver_tumor_data\volume_label"

# 预处理结果存放位置
train_image = r"D:\liver_tumor_data\aiage_liver\train_image"
liver_mask = r"D:\liver_tumor_data\aiage_liver\train_liver_mask"

# 具体预处理哪个图像
items = list(range(131))

# 线程数量
processes=5

# 选取肝脏周边的切片
expand_slice = 6

# 像素阈值
upper = 255
lower = 0

# 切割后patch大小
patch_size = (16, 256, 256)

# 步长
numberxy = 192
numberz = 12

# 数据增强
# 图像旋转
rotate = True
# 图像翻转
flip = True