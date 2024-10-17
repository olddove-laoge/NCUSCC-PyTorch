#这是实现读取csv文件并绘图的代码

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn的样式
sns.set(style="whitegrid")

# 读取CSV文件
df = pd.read_csv('training_data.csv')

# 设置图表大小
plt.figure(figsize=(25, 15))  # 调整图表大小以适应三个子图

# 定义颜色列表，确保颜色数量与数据类别数量匹配
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#ff7f0e','#d62728','#9467bd','#8c564b','#1f77b4','#ff7f0e','#d62728','#9467bd','#8c564b','#ffffe0','#98df8a']

# 绘制Training Time柱状图
plt.subplot(3, 1, 1)
sns.barplot(x='Optimization Technique', y='Training Time (seconds)', data=df, palette=colors)
plt.title('Training Time by Optimization Technique', fontsize=18)
plt.xlabel('Optimization Technique', fontsize=16)
plt.ylabel('Training Time (seconds)', fontsize=16)
plt.grid(True)  # 添加网格线

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height():.2f}', 
             ha='center', va='bottom', fontsize=12, color='black')

# 绘制Accuracy柱状图
plt.subplot(3, 1, 2)
sns.barplot(x='Optimization Technique', y='Accuracy (%)', data=df, palette=colors)
plt.title('Accuracy by Optimization Technique', fontsize=18)
plt.xlabel('Optimization Technique', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.grid(True)  # 添加网格线

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height()}%', 
             ha='center', va='bottom', fontsize=12, color='black')

# 绘制Memory Usage柱状图
plt.subplot(3, 1, 3)
sns.barplot(x='Optimization Technique', y='Memory Usage (MB)', data=df, palette=colors)
plt.title('Memory Usage by Optimization Technique', fontsize=18)
plt.xlabel('Optimization Technique', fontsize=16)
plt.ylabel('Memory Usage (MB)', fontsize=16)
plt.grid(True)  # 添加网格线

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height():.2f} MB', 
             ha='center', va='bottom', fontsize=12, color='black')

# 调整子图间距
plt.tight_layout()

# 保存图表，设置透明度
plt.savefig('optimization_techniques_comparison.png', dpi=300, transparent=True)

# 显示图表
plt.show()
