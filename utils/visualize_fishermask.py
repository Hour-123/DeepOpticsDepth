import sys
import os

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from configs import config


def visualize_phase_masks_from_files(file_path=None, output_dir=None):
    """从文件路径可视化相位掩膜（兼容旧格式，从文件名推断通道）。

    Args:
        file_path: 高度图文件路径（包含 _R_, _G_, _B_ 标记）
        output_dir: 输出目录，默认为文件所在目录
    """
    if file_path is None:
        raise ValueError("file_path is required")

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'

    # 从 config 读取物理常数
    wavelengths = {
        'Red (610nm)': config.WAVELENGTHS[0],
        'Green (530nm)': config.WAVELENGTHS[1],
        'Blue (470nm)': config.WAVELENGTHS[2]
    }
    n_material = config.REFRACTIVE_INDEX
    delta_n = n_material - 1.0

    # 从传入的文件路径推断其他通道的文件路径
    try:
        if '_R_' in file_path:
            base_name = file_path.replace('_R_', '_{}_')
        elif '_G_' in file_path:
            base_name = file_path.replace('_G_', '_{}_')
        elif '_B_' in file_path:
            base_name = file_path.replace('_B_', '_{}_')
        else:
            # 兼容旧格式: 尝试读取单文件3行数据
            height_data = np.loadtxt(file_path)
            if height_data.ndim == 2 and height_data.shape[0] == 3:
                height_maps = height_data.reshape(3, 23, 23)
                channel_info = [
                    ('Red (610nm)', wavelengths['Red (610nm)'], height_maps[0]),
                    ('Green (530nm)', wavelengths['Green (530nm)'], height_maps[1]),
                    ('Blue (470nm)', wavelengths['Blue (470nm)'], height_maps[2])
                ]
            else:
                print(f"无法识别文件格式: {file_path}")
                return None

        if '_R_' in file_path or '_G_' in file_path or '_B_' in file_path:
            # 读取三个独立文件
            height_r = np.loadtxt(base_name.format('R'))
            height_g = np.loadtxt(base_name.format('G'))
            height_b = np.loadtxt(base_name.format('B'))

            channel_info = [
                ('Red (610nm)', wavelengths['Red (610nm)'], height_r),
                ('Green (530nm)', wavelengths['Green (530nm)'], height_g),
                ('Blue (470nm)', wavelengths['Blue (470nm)'], height_b)
            ]

    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    return _plot_phase_masks(channel_info, delta_n, output_dir, os.path.basename(file_path))


def visualize_phase_masks_from_checkpoint(checkpoint_dir, output_dir=None):
    """从 checkpoint 目录加载并可视化相位掩膜（用于 test.py 调用）。

    Args:
        checkpoint_dir: checkpoint 目录，包含 HeightMap_R.txt, HeightMap_G.txt, HeightMap_B.txt
        output_dir: 输出目录，默认为 checkpoint_dir

    Returns:
        output_path: 生成的可视化图片路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = checkpoint_dir

    # 构建文件路径
    height_map_r_path = os.path.join(checkpoint_dir, 'HeightMap_R.txt')
    height_map_g_path = os.path.join(checkpoint_dir, 'HeightMap_G.txt')
    height_map_b_path = os.path.join(checkpoint_dir, 'HeightMap_B.txt')

    # 检查文件是否存在
    for path in [height_map_r_path, height_map_g_path, height_map_b_path]:
        if not os.path.exists(path):
            print(f"Warning: Height map file not found: {path}")
            return None

    # 从 config 读取物理常数
    wavelengths = {
        'Red (610nm)': config.WAVELENGTHS[0],
        'Green (530nm)': config.WAVELENGTHS[1],
        'Blue (470nm)': config.WAVELENGTHS[2]
    }
    n_material = config.REFRACTIVE_INDEX
    delta_n = n_material - 1.0

    # 加载高度图
    try:
        height_r = np.loadtxt(height_map_r_path)
        height_g = np.loadtxt(height_map_g_path)
        height_b = np.loadtxt(height_map_b_path)

        channel_info = [
            ('Red (610nm)', wavelengths['Red (610nm)'], height_r),
            ('Green (530nm)', wavelengths['Green (530nm)'], height_g),
            ('Blue (470nm)', wavelengths['Blue (470nm)'], height_b)
        ]
    except Exception as e:
        print(f"读取高度图失败: {e}")
        return None

    return _plot_phase_masks(channel_info, delta_n, output_dir, "HeightMap")


def _plot_phase_masks(channel_info, delta_n, output_dir, title_name):
    """绘制相位掩膜图。

    Args:
        channel_info: 列表，每个元素为 (channel_name, wavelength, height_map)
        delta_n: 折射率差
        output_dir: 输出目录
        title_name: 标题中显示的名称

    Returns:
        output_path: 生成的图片路径
    """
    # 创建可视化窗口
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Multi-Channel Phase Mask Visualization ($\\phi(x,y)$ in radians)\n{title_name}",
                 fontsize=16)

    # 计算并绘制每个通道的相位
    for i, (name, lam, height_map) in enumerate(channel_info):
        # 计算相位公式: phi = (2 * pi / lambda) * delta_n * h
        phase_map = (2 * np.pi / lam) * delta_n * height_map

        # 绘制 2D 热力图，固定 colorbar 范围为 [0, 2π]
        im = axes[i].imshow(phase_map, cmap='viridis', interpolation='nearest', vmin=0, vmax=2*np.pi)
        axes[i].set_title(name)
        axes[i].set_xlabel("x pixel")
        axes[i].set_ylabel("y pixel")

        # 添加侧边颜色条，显示弧度值 [0, 2π]
        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04,
                           ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        cbar.set_label('Phase (rad)')
        cbar.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存结果图
    output_path = os.path.join(output_dir, 'phase_masks_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Phase mask visualization saved to: {output_path}")
    plt.close()

    return output_path


def visualize_phase_masks(file_path=None, output_dir=None, checkpoint_dir=None):
    """可视化相位掩膜的主函数（兼容多种输入方式）。

    Args:
        file_path: 单个文件路径（用于旧格式，包含 _R_/G/B 标记）
        output_dir: 输出目录
        checkpoint_dir: checkpoint 目录（用于 test.py 直接调用）

    Returns:
        output_path: 生成的图片路径
    """
    if checkpoint_dir is not None:
        return visualize_phase_masks_from_checkpoint(checkpoint_dir, output_dir)
    elif file_path is not None:
        return visualize_phase_masks_from_files(file_path, output_dir)
    else:
        raise ValueError("Either checkpoint_dir or file_path must be provided")


if __name__ == "__main__":
    # 默认文件路径（用于直接运行此脚本）
    FILE_PATH = 'FisherMask_R_Phi_-10to10_Iter2999.txt'
    visualize_phase_masks_from_files(FILE_PATH)
