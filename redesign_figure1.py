"""
重新设计DG-STAN论文图1: 模型整体架构图
严格按照论文描述的5个组件和流程顺序绘制
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse, Arc
from matplotlib.collections import PatchCollection
import numpy as np

# 设置全局样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'cm'

# 定义配色方案
COLORS = {
    # 输入/输出
    'input_fill': '#E8F5E9',
    'input_border': '#2E7D32',
    'output_fill': '#E3F2FD', 
    'output_border': '#1565C0',
    
    # 静态图模块 - 蓝色系
    'static_fill': '#E3F2FD',
    'static_border': '#1565C0',
    
    # 动态图模块 - 橙色系
    'dynamic_fill': '#FFF3E0',
    'dynamic_border': '#E65100',
    
    # 图融合模块 - 红色系
    'fusion_fill': '#FFEBEE',
    'fusion_border': '#C62828',
    
    # 时间卷积模块 - 绿色系
    'temporal_fill': '#E8F5E9',
    'temporal_border': '#2E7D32',
    
    # 时空注意力模块 - 紫色系
    'attention_fill': '#F3E5F5',
    'attention_border': '#6A1B9A',
    
    # 优化模块 - 青色系
    'optim_fill': '#E0F7FA',
    'optim_border': '#00838F',
    
    # 其他
    'arrow': '#37474F',
    'text': '#212121',
    'white': '#FFFFFF',
    'gray': '#ECEFF1',
    'light_gray': '#FAFAFA',
}


def draw_module_box(ax, x, y, width, height, title, fill_color, border_color, 
                    subtitle=None, components=None, fontsize=11):
    """绘制模块框"""
    # 主框
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=fill_color, edgecolor=border_color,
                         linewidth=2.5, alpha=0.95)
    ax.add_patch(box)
    
    # 标题
    title_y = y + height - 0.4
    ax.text(x + width/2, title_y, title,
            ha='center', va='top', fontsize=fontsize, fontweight='bold',
            color=border_color)
    
    # 副标题
    if subtitle:
        ax.text(x + width/2, title_y - 0.35, subtitle,
                ha='center', va='top', fontsize=8,
                color='#666', style='italic')
    
    # 组件列表
    if components:
        comp_y = y + height - 1.0
        for i, comp in enumerate(components):
            comp_box = FancyBboxPatch((x + 0.15, comp_y - i*0.55), width - 0.3, 0.45,
                                      boxstyle="round,pad=0.01,rounding_size=0.08",
                                      facecolor=COLORS['white'], edgecolor=border_color,
                                      linewidth=1.2, alpha=0.9)
            ax.add_patch(comp_box)
            ax.text(x + width/2, comp_y - i*0.55 + 0.22, comp,
                    ha='center', va='center', fontsize=8, color=COLORS['text'])


def draw_arrow(ax, start, end, color='#37474F', style='-|>', linewidth=2,
               connectionstyle="arc3,rad=0", linestyle='-'):
    """绘制箭头"""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            mutation_scale=15,
                            color=color,
                            linewidth=linewidth,
                            connectionstyle=connectionstyle,
                            linestyle=linestyle)
    ax.add_patch(arrow)


def draw_small_box(ax, x, y, width, height, text, fill_color, border_color, fontsize=9):
    """绘制小框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=fill_color, edgecolor=border_color,
                         linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize, color=COLORS['text'])


def create_figure1():
    """
    创建图1: DG-STAN整体架构图
    按照论文描述的流程: 输入 → 静态图+动态图 → 图融合 → 多尺度时间卷积 → 时空注意力 → 输出
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # ========== 标题 ==========
    ax.text(8, 9.6, 'DG-STAN: Dynamic Graph Spatio-Temporal Attention Network',
            ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['text'])
    ax.text(8, 9.2, 'Overall Architecture',
            ha='center', va='center', fontsize=12, color='#666', style='italic')
    
    # ========== (1) 输入模块 ==========
    draw_small_box(ax, 0.3, 4.0, 1.8, 1.8, 
                   'Input\n$X \\in \\mathbb{R}^{T \\times N \\times F}$\n\nTraffic Data',
                   COLORS['input_fill'], COLORS['input_border'], fontsize=9)
    
    # ========== (2) 静态图构建模块 ==========
    static_x, static_y = 2.8, 5.8
    draw_module_box(ax, static_x, static_y, 2.8, 2.8,
                    '(1) Static Graph',
                    COLORS['static_fill'], COLORS['static_border'],
                    subtitle='Construction',
                    components=['Road Network', 'Distance Weighting', '$A \\in \\mathbb{R}^{N \\times N}$'])
    
    # ========== (3) 动态图生成模块 ==========
    dynamic_x, dynamic_y = 2.8, 1.2
    draw_module_box(ax, dynamic_x, dynamic_y, 2.8, 2.8,
                    '(2) Dynamic Graph',
                    COLORS['dynamic_fill'], COLORS['dynamic_border'],
                    subtitle='Generation',
                    components=['Feature Aggregation', 'Similarity + Sigmoid', '$A_t^{dyn} \\in \\mathbb{R}^{N \\times N}$'])
    
    # ========== (4) 图门控融合模块 ==========
    fusion_x, fusion_y = 6.2, 3.0
    draw_module_box(ax, fusion_x, fusion_y, 2.6, 3.8,
                    '(3) Graph Gating',
                    COLORS['fusion_fill'], COLORS['fusion_border'],
                    subtitle='Fusion',
                    components=['Concat [A, $A_t^{dyn}$]', 'MLP', '$\\sigma(\\cdot)$', 
                               '$G \\in [0,1]^{N \\times N}$', '$A_{fusion}$'])
    
    # ========== (5) 多尺度时间卷积模块 ==========
    temporal_x, temporal_y = 9.4, 5.2
    draw_module_box(ax, temporal_x, temporal_y, 3.0, 3.4,
                    '(4) Multi-Scale',
                    COLORS['temporal_fill'], COLORS['temporal_border'],
                    subtitle='Temporal Convolution',
                    components=['Conv1D (K=3)', 'Conv1D (K=5)', 'Conv1D (K=11)', 
                               'Gated Attention'])
    
    # ========== (6) 时空注意力模块 ==========
    attention_x, attention_y = 9.4, 1.0
    draw_module_box(ax, attention_x, attention_y, 3.0, 3.6,
                    '(5) Spatio-Temporal',
                    COLORS['attention_fill'], COLORS['attention_border'],
                    subtitle='Attention',
                    components=['Spatial Attention', 'Temporal Attention', 
                               'Multi-Head', 'Residual Connection'])
    
    # ========== (7) 输出/优化模块 ==========
    output_x, output_y = 13.0, 3.2
    draw_module_box(ax, output_x, output_y, 2.6, 3.4,
                    'Output',
                    COLORS['output_fill'], COLORS['output_border'],
                    subtitle='Prediction',
                    components=['GRU Decoder', 'MAE Loss', 'AdamW + OneCycleLR',
                               '$\\hat{Y} \\in \\mathbb{R}^{T\' \\times N \\times F}$'])
    
    # ========== 绘制数据流箭头 ==========
    arrow_color = '#455A64'
    
    # 输入 → 静态图
    draw_arrow(ax, (2.1, 5.2), (2.8, 6.8), color=arrow_color, connectionstyle="arc3,rad=0.2")
    # 输入 → 动态图
    draw_arrow(ax, (2.1, 4.6), (2.8, 3.0), color=arrow_color, connectionstyle="arc3,rad=-0.2")
    
    # 静态图 → 图融合
    draw_arrow(ax, (5.6, 6.5), (6.2, 5.8), color=COLORS['static_border'], connectionstyle="arc3,rad=0.15")
    # 动态图 → 图融合
    draw_arrow(ax, (5.6, 3.0), (6.2, 4.0), color=COLORS['dynamic_border'], connectionstyle="arc3,rad=-0.15")
    
    # 图融合 → 时空注意力 (提供融合图)
    draw_arrow(ax, (8.8, 4.5), (9.4, 3.5), color=COLORS['fusion_border'], 
               connectionstyle="arc3,rad=-0.1", linestyle='--')
    
    # 输入 → 多尺度时间卷积 (原始特征)
    draw_arrow(ax, (2.1, 5.0), (9.4, 6.5), color=arrow_color, connectionstyle="arc3,rad=0.15")
    
    # 多尺度时间卷积 → 时空注意力
    draw_arrow(ax, (10.9, 5.2), (10.9, 4.6), color=COLORS['temporal_border'])
    
    # 时空注意力 → 输出
    draw_arrow(ax, (12.4, 2.8), (13.0, 4.2), color=COLORS['attention_border'], 
               connectionstyle="arc3,rad=0.1")
    
    # ========== 添加流程标注 ==========
    # 标注箭头含义
    ax.text(1.8, 6.0, '$X$', fontsize=9, color='#666', style='italic')
    ax.text(1.8, 3.8, '$X$', fontsize=9, color='#666', style='italic')
    ax.text(5.8, 7.3, '$A$', fontsize=9, color=COLORS['static_border'], style='italic')
    ax.text(5.8, 2.5, '$A_t^{dyn}$', fontsize=9, color=COLORS['dynamic_border'], style='italic')
    ax.text(8.5, 3.8, '$A_{fusion}$', fontsize=9, color=COLORS['fusion_border'], style='italic')
    ax.text(5.5, 6.0, '$H$', fontsize=9, color='#666', style='italic')
    ax.text(11.2, 4.9, '$H_{temporal}$', fontsize=9, color=COLORS['temporal_border'], style='italic')
    ax.text(12.5, 3.2, '$H_{out}$', fontsize=9, color=COLORS['attention_border'], style='italic')
    
    # ========== 添加图例 ==========
    legend_y = 0.3
    legend_items = [
        (COLORS['static_fill'], COLORS['static_border'], 'Static Graph'),
        (COLORS['dynamic_fill'], COLORS['dynamic_border'], 'Dynamic Graph'),
        (COLORS['fusion_fill'], COLORS['fusion_border'], 'Graph Fusion'),
        (COLORS['temporal_fill'], COLORS['temporal_border'], 'Temporal Conv'),
        (COLORS['attention_fill'], COLORS['attention_border'], 'Attention'),
    ]
    
    for i, (fill, border, label) in enumerate(legend_items):
        x_pos = 1.0 + i * 2.8
        rect = Rectangle((x_pos, legend_y), 0.4, 0.4, 
                         facecolor=fill, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, legend_y + 0.2, label, fontsize=9, va='center')
    
    # ========== 添加流程说明 ==========
    ax.text(8, 0.1, 
            'Data Flow: Input → Static/Dynamic Graph Construction → Graph Gating Fusion → Multi-Scale Temporal Conv → Spatio-Temporal Attention → Output',
            ha='center', va='center', fontsize=9, color='#666', style='italic')
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("重新设计DG-STAN论文图1: 模型整体架构图")
    print("=" * 60)
    
    output_dir = '/data_ssd/other_models/IEEE'
    
    # 创建图1
    print("\n生成图1: 模型整体架构图...")
    fig1 = create_figure1()
    
    # 保存为PNG和PDF
    fig1.savefig(f'{output_dir}/1.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig1.savefig(f'{output_dir}/figure1_architecture_v2.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig1)
    
    print(f"   保存: {output_dir}/1.png")
    print(f"   保存: {output_dir}/figure1_architecture_v2.pdf")
    
    # 备份旧图
    import shutil
    import os
    old_path = f'{output_dir}/1_old_original.png'
    if not os.path.exists(old_path):
        # 如果还没有备份原始图，先备份
        if os.path.exists(f'{output_dir}/1_old.png'):
            shutil.copy(f'{output_dir}/1_old.png', old_path)
            print(f"   原始图已备份到: {old_path}")
    
    print("\n" + "=" * 60)
    print("图1重新设计完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
