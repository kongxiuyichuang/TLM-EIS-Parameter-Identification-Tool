# -*- coding: utf-8 -*-
"""
TLM 传输线模型 EIS 参数辨识工具
Transmission Line Model - EIS Parameter Identification Tool
Created by 锂电之路
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.optimize import least_squares
import os
import sys
import warnings
import webbrowser


def resource_path(relative_path):
    """获取资源绝对路径，兼容开发环境和 PyInstaller 打包。"""
    try:
        base_path = sys._MEIPASS          # PyInstaller 运行时临时目录
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

warnings.filterwarnings('ignore')

# ───────────────── 全局样式配置 ─────────────────
FONT_FAMILY = "Microsoft YaHei UI"
MONO_FONT = "Consolas"

COLORS = {
    'bg':            '#f0f2f5',
    'panel':         '#ffffff',
    'primary':       '#1a237e',
    'accent':        '#2962ff',
    'success':       '#2e7d32',
    'danger':        '#c62828',
    'warning':       '#f57f17',
    'text':          '#212121',
    'text_secondary':'#616161',
    'border':        '#e0e0e0',
    'raw_marker':    '#bdbdbd',
    'sel_marker':    '#2962ff',
    'fit_line':      '#c62828',
    'res_re':        '#2e7d32',
    'res_im':        '#f57f17',
}

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# ───────────────── 物理模型核心 ─────────────────

def _safe_sinh_coth(x):
    """稳定计算 sinh(x) 和 coth(x)（支持复数）"""
    ax = np.abs(x)

    # --- 小量区 ---
    if ax < 1e-6:
        x2 = x * x
        sinh = x * (1 + x2/6 + x2*x2/120)
        coth = 1/x + x/3 - x*x2/45
        return sinh, coth

    # --- 大量区 ---
    if ax > 50:
        # sinh ≈ 0.5 exp(x)
        sinh = 0.5 * np.exp(x)
        coth = 1.0 + 2*np.exp(-2*x)  # 保留尾项更精确
        return sinh, coth

    # --- 中间区 ---
    exp_p = np.exp(x)
    exp_n = np.exp(-x)
    sinh = (exp_p - exp_n) / 2
    coth = (exp_p + exp_n) / (exp_p - exp_n)

    return sinh, coth

def _safe_inv_sinh_coth(x):
    """
    稳健计算 1/sinh(x) 和 coth(x)，防止复数指数溢出。
    原理：当 Re(x) 很大时，sinh(x) -> inf, 但 1/sinh(x) -> 0，后者在数值上是安全的。
    """
    # 使用实部来判断是否会发生指数溢出
    re_x = np.real(x)
    abs_x = np.abs(x)

    # # --- 1. 小量区 (泰勒展开防止除以0) ---
    # if abs_x < 1e-6:
        
    #     # 1/sinh(x) ≈ 1/x - x/6
    #     inv_sinh = 1.0/x - x/6.0
    #     # coth(x) ≈ 1/x + x/3
    #     coth = 1.0/x + x/3.0
    #     return inv_sinh, coth

    # --- 2. 大量区 (防止 exp(x) 溢出) ---
    # 当 re_x > 50 时，exp(50) 约 5e21，1/sinh 已经极小，直接使用渐进公式
    if re_x > 50:
        # 1/sinh(x) = 2 / (exp(x) - exp(-x)) ≈ 2 * exp(-x)
        inv_sinh = 2.0 * np.exp(-x)
        # coth(x) = (exp(x) + exp(-x)) / (exp(x) - exp(-x)) ≈ 1 + 2*exp(-2x)
        coth = 1.0 + 2.0 * np.exp(-2.0 * x)
        return inv_sinh, coth

    # --- 3. 中间常规区 ---
    ep = np.exp(x)
    en = np.exp(-x)
    inv_sinh = 2.0 / (ep - en)
    coth = (ep + en) / (ep - en)

    return inv_sinh, coth

def tlm_impedance(f, delta, R0, Re, Ri, Cdl, p, Rct=None):
    """
    带有数值稳定性优化的 TLM 阻抗计算
    """
    omega = 2.0 * np.pi * f
    j = 1j

    # 计算界面阻抗 Zt
    if Rct is None:
        Zt = 1.0 / (Cdl * (omega * j) ** p)
    else:
        Zt = Rct / (1.0 + Cdl * (omega *j ) ** p * Rct)

    # 计算传播常数 gamma
    gamma = np.sqrt((Re + Ri) / Zt)
    gd = gamma * delta

    # 获取稳健的中间项：inv_sinh_val = 1/sinh(gd)
    inv_sinh_val, coth_val = _safe_inv_sinh_coth(gd)

    coeff = Re * Ri / (Re + Ri)
    
    # 第一项：欧姆阻抗与分布电阻基础项
    term1 = R0 + coeff * delta
    
    # 第二项：交互项 (通过乘法使用 inv_sinh，彻底避免 Inf 错误)
    # 原式：coeff * (2.0 / (gamma * sinh(gd)))
    term2 = (coeff * 2.0 / gamma) * inv_sinh_val
    
    # 第三项：终端项
    # 原式：(1/gamma) * ((Re^2+Ri^2)/(Re+Ri)) * coth(gd)
    #term3 = (1.0 / gamma) * ((Re**2 + Ri**2) / (Re + Ri)) * coth_val
    term3 = (1.0 / gamma) * ((Re**2 + Ri**2) / (Re + Ri)) * (1/np.tanh(gd))
    return term1 + term2 + term3


# ───────────────── GUI 应用 ─────────────────
class TLMFitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("对称电池 EIS-TLM参数辨识工具V1.0")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLORS['bg'])
        self.root.minsize(900, 580)

        # ── 数据状态 ──
        self.raw_data = None          # (N,3) [f, Real, Imag]
        self.filtered_data = None     # 筛选后数据
        self.fit_results = None       # list of fitted params (full)
        self.z_fit = None             # np.array complex
        self.res_re = None            # real residuals
        self.res_im = None            # imag residuals
        self.fit_stats = {}           # dict of stats

        self._setup_style()
        self._build_ui()
        self._draw_welcome()

    # ── 样式 ──
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background=COLORS['bg'])
        style.configure("TFrame", background=COLORS['bg'])
        style.configure("Panel.TFrame", background=COLORS['panel'])
        style.configure("TLabel", background=COLORS['bg'], font=(FONT_FAMILY, 10),
                        foreground=COLORS['text'])
        style.configure("Panel.TLabel", background=COLORS['panel'],
                        font=(FONT_FAMILY, 10), foreground=COLORS['text'])
        style.configure("Bold.TLabel", font=(FONT_FAMILY, 10, "bold"))
        style.configure("Title.TLabel", font=(FONT_FAMILY, 12, "bold"),
                        foreground=COLORS['primary'])
        style.configure("TButton", font=(FONT_FAMILY, 10))
        style.configure("TLabelframe", background=COLORS['bg'])
        style.configure("TLabelframe.Label", font=(FONT_FAMILY, 10, "bold"),
                        foreground=COLORS['primary'])
        style.configure("TCheckbutton", background=COLORS['panel'],
                        font=(FONT_FAMILY, 9))

    # ── UI 框架 ──
    def _build_ui(self):
        # ─── 主分隔区 ───
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左侧 - 全部控制面板 (垂直堆叠)
        left = ttk.Frame(self.main_pane)
        self.main_pane.add(left, weight=1)
        self._build_left_panel(left)

        # 右侧 - 纯绘图区
        right = ttk.Frame(self.main_pane)
        self.main_pane.add(right, weight=4)
        self._build_plot_panel(right)

    def _build_left_panel(self, parent):
        """垂直堆叠所有控制组件于左侧面板。"""
        # ─── 1. 数据导入与模型选择 ───
        ctrl_lf = ttk.LabelFrame(parent, text="数据导入与模型选择", padding=8)
        ctrl_lf.pack(fill=tk.X, pady=(0, 6))

        self.btn_import = ttk.Button(ctrl_lf, text="📂 导入数据 (csv/txt/xlsx)",
                                     command=self._load_data)
        self.btn_import.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(ctrl_lf, text="📖 使用文档", command=self._open_document).pack(
            fill=tk.X, pady=(0, 6))

        ttk.Separator(ctrl_lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # 模型选择
        model_row = ttk.Frame(ctrl_lf, style="Panel.TFrame")
        model_row.pack(fill=tk.X)
        ttk.Label(model_row, text="模型选择:", style="Bold.TLabel",
                  background=COLORS['panel']).pack(side=tk.LEFT, padx=(0, 8))
        self.model_var = tk.StringVar(value="Non-Faradaic")
        ttk.Radiobutton(model_row, text="Non-Faradaic", variable=self.model_var,
                        value="Non-Faradaic",
                        command=self._on_model_change).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(model_row, text="Faradaic", variable=self.model_var,
                        value="Faradaic",
                        command=self._on_model_change).pack(side=tk.LEFT, padx=3)

        # 公式渲染区
        self.eq_frame = ttk.Frame(ctrl_lf, height=80)
        self.eq_frame.pack(fill=tk.X, pady=(6, 0))
        self.eq_frame.pack_propagate(False)
        self.eq_fig = Figure(figsize=(3.8, 0.7), dpi=80, facecolor=COLORS['panel'])
        self.eq_canvas = FigureCanvasTkAgg(self.eq_fig, master=self.eq_frame)
        self.eq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ─── 2. 频率筛选 (紧接导入与模型选择下方) ───
        self.filter_lf = ttk.LabelFrame(parent, text="频率筛选 (Hz)", padding=(6, 4))
        self.filter_lf.pack(fill=tk.X, pady=(0, 6))

        row1 = ttk.Frame(self.filter_lf)
        row1.pack(fill=tk.X, pady=1)
        ttk.Label(row1, text="f_min:", width=6).pack(side=tk.LEFT)
        self.slider_start = ttk.Scale(row1, from_=-3, to=6,
                                      orient=tk.HORIZONTAL, length=140,
                                      command=self._on_filter)
        self.slider_start.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.lbl_start = ttk.Label(row1, text="0.001", width=8)
        self.lbl_start.pack(side=tk.LEFT)

        row2 = ttk.Frame(self.filter_lf)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="f_max:", width=6).pack(side=tk.LEFT)
        self.slider_end = ttk.Scale(row2, from_=-3, to=6,
                                    orient=tk.HORIZONTAL, length=140,
                                    command=self._on_filter)
        self.slider_end.set(6)
        self.slider_end.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.lbl_end = ttk.Label(row2, text="1e6", width=8)
        self.lbl_end.pack(side=tk.LEFT)

        # ─── 3. 可滚动区域：参数配置 + 按钮 + 统计 + 版权 ───
        scroll_container = ttk.Frame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(scroll_container, bg=COLORS['bg'],
                                       highlightthickness=0)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL,
                                  command=self.scroll_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        # 滚动内容框架
        scroll_inner = ttk.Frame(self.scroll_canvas)
        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0), window=scroll_inner, anchor="nw")

        # 内容宽度跟随 canvas 宽度
        def _on_canvas_configure(event):
            self.scroll_canvas.itemconfig(
                self.scroll_window, width=event.width)
        self.scroll_canvas.bind("<Configure>", _on_canvas_configure)

        # 滚动区域自动更新
        def _on_inner_configure(event):
            self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all"))
        scroll_inner.bind("<Configure>", _on_inner_configure)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            self.scroll_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")
        self.scroll_canvas.bind("<Enter>",
            lambda e: self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scroll_canvas.bind("<Leave>",
            lambda e: self.scroll_canvas.unbind_all("<MouseWheel>"))

        # ─── 3a. 拟合参数配置 ───
        param_lf = ttk.LabelFrame(scroll_inner, text="拟合参数配置", padding=8)
        param_lf.pack(fill=tk.X, pady=(0, 6))

        self.param_inner = ttk.Frame(param_lf, style="Panel.TFrame")
        self.param_inner.pack(fill=tk.X)

        self.param_widgets = {}
        self.param_names = []
        self._render_params()

        # 按钮
        btn_frame = ttk.Frame(param_lf, style="Panel.TFrame")
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        self.btn_fit = ttk.Button(btn_frame, text="▶ 运行拟合",
                                  command=self._run_fit)
        self.btn_fit.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_export = ttk.Button(btn_frame, text="📊 导出结果",
                                     command=self._export_data)
        self.btn_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # ─── 3b. 统计信息 ───
        stats_lf = ttk.LabelFrame(scroll_inner, text="拟合统计信息", padding=5)
        stats_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        txt_frame = ttk.Frame(stats_lf)
        txt_frame.pack(fill=tk.BOTH, expand=True)

        # 统计文本框使用固定像素高度，避免在滚动区域内过度拉伸
        self.stats_text = tk.Text(txt_frame, height=8, font=(MONO_FONT, 9),
                                  bg=COLORS['panel'], fg=COLORS['text'],
                                  relief=tk.SOLID, borderwidth=1, wrap=tk.NONE)
        self.stats_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        sb_text = ttk.Scrollbar(txt_frame, command=self.stats_text.yview)
        sb_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=sb_text.set)

        copy_btn = ttk.Button(stats_lf, text="📋 复制统计信息",
                              command=lambda: self.root.clipboard_append(
                                  self.stats_text.get("1.0", tk.END)))
        copy_btn.pack(anchor="e", pady=(4, 0))

        # ─── 3c. 版权与打赏 ───
        self.author_frame = ttk.Frame(scroll_inner, padding=(6, 4),
                                      style="Panel.TFrame")
        self.author_frame.pack(fill=tk.X)

        ttk.Label(self.author_frame, text="Created by 锂电之路",
                  font=(FONT_FAMILY, 10, "bold italic"),
                  background=COLORS['panel'],
                  foreground=COLORS['primary']).pack(anchor="w", pady=(0, 4))

        qr_row = ttk.Frame(self.author_frame, style="Panel.TFrame")
        qr_row.pack(fill=tk.X, pady=(2, 0))

        donate_col = ttk.Frame(qr_row, style="Panel.TFrame")
        donate_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._load_qr(resource_path("qrcode.png"),
                      donate_col, "buy me a coffee", fallback_text="[打赏码]")

        gzh_col = ttk.Frame(qr_row, style="Panel.TFrame")
        gzh_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._load_qr(resource_path("gzh_qrcode.png"),
                      gzh_col, "更多信息关注公众号", fallback_text="[公众号码]")

    def _load_qr(self, path, parent, caption, fallback_text="[QR]"):
        """加载并显示 100×100 二维码图片 + 底部文字标签。"""
        if os.path.exists(path):
            try:
                from PIL import Image, ImageTk
                img = Image.open(path).resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl = ttk.Label(parent, image=photo)
                lbl.image = photo             # 保持引用防止 GC
                lbl.pack()
            except Exception:
                ttk.Label(parent, text=fallback_text,
                          foreground=COLORS['text_secondary']).pack()
        else:
            ttk.Label(parent, text=fallback_text,
                      foreground=COLORS['text_secondary']).pack()

        ttk.Label(parent, text=caption,
                  font=(FONT_FAMILY, 8, "italic"),
                  background=COLORS['panel'],
                  foreground=COLORS['text_secondary']).pack()

    def _open_document(self):
        """在系统默认浏览器中打开使用文档。"""
        webbrowser.open(resource_path("document.html"))

    def _render_params(self):
        """模型切换时重建整个参数表格 (参考 refresh_params 模式)。"""
        # 清空旧控件
        for widget in self.param_inner.winfo_children():
            widget.destroy()

        # 表头
        headers = ["参数", "下界", "初始值", "上界", "固定"]
        col_widths = [9, 10, 12, 10, 4]
        for j, (h, w) in enumerate(zip(headers, col_widths)):
            ttk.Label(self.param_inner, text=h, font=(FONT_FAMILY, 9, "bold"),
                      width=w, anchor="center").grid(row=0, column=j, padx=2, pady=4)

        model = self.model_var.get()
        if model == "Non-Faradaic":
            self.param_names = ["δ (cm)", "R\u2080 (\u03a9)", "R\u2091 (\u03a9/cm)",
                                "R\u1d62 (\u03a9/cm)", "C_dl (F/cm)", "p"]
            self.configs = [
                (0.09, 0.1, 0.11),       # delta
                (1e-2, 5e-1, 1.0),        # R0
                (0.1, 50.0, 5000.0),      # Re
                (1.0, 200.0, 10000.0),    # Ri
                (1e-5, 1e-3, 0.1),          # Cdl
                (0.5,  0.95, 1.0),          # p
            ]
        else:
            self.param_names = ["δ (cm)", "R\u2080 (\u03a9)", "R\u2091 (\u03a9/cm)",
                                "R\u1d62 (\u03a9/cm)", "C_dl (F/cm)", "p",
                                "R_ct (\u03a9\u00b7cm)"]
            self.configs = [
                (0.09, 0.1, 0.1),
                (1e-2, 5e-1, 1.0),
                (0.1, 50.0, 5000.0),
                (1.0, 200.0, 10000.0),
                (1e-5, 1e-3, 0.1),
                (0.5,  0.95, 1.0),
                (10.0, 5000.0, 1e6),       # Rct
            ]

        self.param_widgets = {}
        for i, (pname, (lb, init, ub)) in enumerate(zip(self.param_names, self.configs)):
            r = i + 1  # data rows start from row 1
            ttk.Label(self.param_inner, text=pname,
                      font=(FONT_FAMILY, 10),
                      background=COLORS['panel']).grid(
                row=r, column=0, sticky="w", padx=(6, 2), pady=2)

            ent_l = ttk.Entry(self.param_inner, width=11, justify="center")
            ent_l.insert(0, str(lb))
            ent_l.grid(row=r, column=1, padx=2, pady=2)

            ent_v = ttk.Entry(self.param_inner, width=13, justify="center")
            ent_v.insert(0, str(init))
            ent_v.grid(row=r, column=2, padx=2, pady=2)

            ent_u = ttk.Entry(self.param_inner, width=11, justify="center")
            ent_u.insert(0, str(ub))
            ent_u.grid(row=r, column=3, padx=2, pady=2)

            fix_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(self.param_inner, variable=fix_var)
            chk.grid(row=r, column=4)

            self.param_widgets[pname] = {
                'lower': ent_l, 'value': ent_v, 'upper': ent_u, 'fixed': fix_var
            }

        self._update_formula_fig()

    def _on_model_change(self):
        self._render_params()

    def _update_formula_fig(self):
        """Render the TLM + Zt formula using matplotlib mathtext."""
        self.eq_fig.clear()
        ax = self.eq_fig.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor(COLORS['panel'])

        model = self.model_var.get()
        if model == "Non-Faradaic":
            zt_eq = r"$Z_t = \frac{1}{j(\omega C_{dl})^p}$"
        else:
            zt_eq = r"$Z_t = \frac{R_{ct}}{1 + j(\omega C_{dl})^p R_{ct}}$"

        full = (
            r"$Z = R_0 + \frac{R_e R_i}{R_e+R_i}\delta + "
            r"\frac{R_e R_i}{R_e+R_i}\cdot\frac{2}{\gamma\sinh(\delta\gamma)} + "
            r"\frac{1}{\gamma}\cdot\frac{R_e^2+R_i^2}{R_e+R_i}\coth(\delta\gamma)$"
            + "\n"
            + zt_eq + r",$\quad \gamma = \sqrt{(R_e+R_i) / Z_t}$"
        )
        ax.text(0.02, 0.5, full, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', color=COLORS['primary'])
        self.eq_fig.tight_layout(pad=0.3)
        self.eq_canvas.draw()

    def _build_plot_panel(self, parent):
        # 使用 Notebook 创建两个重叠的绘图页
        self.plot_nb = ttk.Notebook(parent)
        self.plot_nb.pack(fill=tk.BOTH, expand=True)

        # ── Tab 1: 实验数据预览 ──
        self.tab_raw = ttk.Frame(self.plot_nb)
        self.plot_nb.add(self.tab_raw, text="  实验数据预览  ")
        self.fig_raw = Figure(dpi=100, facecolor='white', layout='constrained')
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.tab_raw)
        self._setup_plot_tab(self.tab_raw, self.canvas_raw, self.fig_raw)

        # ── Tab 2: 拟合辨识结果 ──
        self.tab_fit = ttk.Frame(self.plot_nb)
        self.plot_nb.add(self.tab_fit, text="  拟合辨识结果  ")
        self.fig_fit = Figure(dpi=100, facecolor='white', layout='constrained')
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=self.tab_fit)
        self._setup_plot_tab(self.tab_fit, self.canvas_fit, self.fig_fit)

        self.plot_nb.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _setup_plot_tab(self, frame, canvas, fig):
        """工具栏 + 绘图区 + 窗口缩放自动刷新。"""
        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 监听容器 frame 尺寸变化 → 自动刷新绘图组
        frame.bind("<Configure>",
                   lambda e: self._resize_handler(fig, canvas, frame))

    def _resize_handler(self, fig, canvas, container):
        """动态读取容器像素尺寸，同步 Figure 英寸大小并重绘。"""
        w = container.winfo_width()
        h = container.winfo_height() - 45       # 扣除工具栏高度
        if w > 100 and h > 100:
            dpi = fig.get_dpi()
            fig.set_size_inches(w / dpi, h / dpi)
            canvas.draw_idle()

    def _on_tab_changed(self, event):
        """切换标签页时刷新对应 Figure 尺寸。"""
        idx = self.plot_nb.index("current")
        if idx == 0:
            self._resize_handler(self.fig_raw, self.canvas_raw, self.tab_raw)
        else:
            self._resize_handler(self.fig_fit, self.canvas_fit, self.tab_fit)

    # ── 初始欢迎图 ──
    def _draw_welcome(self):
        for fig, canvas in [(self.fig_raw, self.canvas_raw),
                            (self.fig_fit, self.canvas_fit)]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Import data to begin",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=18, color='#9e9e9e')
            ax.axis('off')
            canvas.draw()

    # ── 数据导入 ──
    def _load_data(self):
        fp = filedialog.askopenfilename(
            title="选择 EIS 数据文件",
            filetypes=[
                ("支持的文件", "*.csv *.txt *.xlsx *.xls"),
                ("CSV", "*.csv"), ("TXT", "*.txt"),
                ("Excel", "*.xlsx *.xls"),
            ]
        )
        if not fp:
            return

        try:
            ext = os.path.splitext(fp)[1].lower()
            if ext in ('.xlsx', '.xls'):
                df = pd.read_excel(fp)
            else:
                df = pd.read_csv(fp, sep=None, engine='python')

            data = df.iloc[:, :3].values.astype(float)
            # 按频率降序排列（EIS 惯例）
            data = data[np.argsort(data[:, 0])[::-1]]

            self.raw_data = data
            self.filtered_data = data.copy()
            self.fit_results = None
            self.z_fit = None
            self.res_re = None
            self.res_im = None

            n = len(data)
            f_min_log = np.log10(np.min(data[:, 0]))
            f_max_log = np.log10(np.max(data[:, 0]))
            pad = (f_max_log - f_min_log) * 0.05
            self.slider_start.config(from_=f_min_log - pad, to=f_max_log + pad)
            self.slider_end.config(from_=f_min_log - pad, to=f_max_log + pad)
            self.slider_start.set(f_min_log - pad)
            self.slider_end.set(f_max_log + pad)
            self.lbl_start.config(text=f"{10**f_min_log:.2g}")
            self.lbl_end.config(text=f"{10**f_max_log:.2g}")

            self.stats_text.delete("1.0", tk.END)

            self._plot_experimental()
            # 延迟同步，确保 tk 布局已落定
            self.root.after(150, lambda: self._resize_handler(
                self.fig_raw, self.canvas_raw, self.tab_raw))
            print(f"[INFO] Loaded {n} data points from {os.path.basename(fp)}")

        except Exception as e:
            messagebox.showerror("导入错误", f"文件读取失败:\n{e}")

    # ── 频率筛选 ──
    def _on_filter(self, _=None):
        if self.raw_data is None:
            return
        f_log_start = float(self.slider_start.get())
        f_log_end = float(self.slider_end.get())

        # 确保起点 < 终点
        if f_log_start > f_log_end:
            f_log_start = f_log_end
            self.slider_start.set(f_log_start)

        f_start = 10.0 ** f_log_start
        f_end = 10.0 ** f_log_end

        self.lbl_start.config(text=f"{f_start:.2g}")
        self.lbl_end.config(text=f"{f_end:.2g}")

        # 按频率范围筛选
        freqs = self.raw_data[:, 0]
        mask = (freqs >= f_start) & (freqs <= f_end)
        if np.sum(mask) < 2:
            return
        self.filtered_data = self.raw_data[mask].copy()

        # 实时刷新两页绘图
        self._plot_experimental()
        if self.fit_results is not None:
            self._plot_fit_results()

    # ── 实验数据绘图 (4 panels, Raw_Data tab) ──
    def _plot_experimental(self):
        if self.raw_data is None:
            return

        self.fig_raw.clear()
        f_r, re_r, im_r = self.raw_data.T
        f_f, re_f, im_f = self.filtered_data.T

        gs = self.fig_raw.add_gridspec(2, 2)

        # Nyquist
        ax1 = self.fig_raw.add_subplot(gs[0, 0])
        ax1.plot(re_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5,
                 label='Raw data')
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5,
                 label='Selected')
        ax1.set_xlabel("Z' (\u03a9)")
        ax1.set_ylabel("-Z'' (\u03a9)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')

        # Bode
        ax2 = self.fig_raw.add_subplot(gs[0, 1])
        ax2.loglog(f_r, np.sqrt(re_r**2 + im_r**2), 'o',
                   color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax2.loglog(f_f, np.sqrt(re_f**2 + im_f**2), 'o',
                   color=COLORS['sel_marker'], ms=5, label='Selected')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("|Z| (\u03a9)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8)
        ax2.grid(True, alpha=0.3, which='both')

        # Real vs f
        ax3 = self.fig_raw.add_subplot(gs[1, 0])
        ax3.semilogx(f_r, re_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5,
                     label='Raw')
        ax3.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5,
                     label='Selected')
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Z' (\u03a9)")
        ax3.set_title("Real Impedance vs f", fontweight='bold')
        ax3.legend(framealpha=0.8)
        ax3.grid(True, alpha=0.3)

        # -Imag vs f
        ax4 = self.fig_raw.add_subplot(gs[1, 1])
        ax4.semilogx(f_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5,
                     label='Raw')
        ax4.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5,
                     label='Selected')
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("-Z'' (\u03a9)")
        ax4.set_title("-Imag Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8)
        ax4.grid(True, alpha=0.3)

        self.canvas_raw.draw()

    # ── 拟合运行 ──
    def _run_fit(self):
        if self.filtered_data is None or len(self.filtered_data) < 2:
            messagebox.showwarning("数据不足", "请导入数据并确保筛选后至少保留 2 个点")
            return

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "正在拟合，请稍候...\n")
        self.root.update()

        try:
            f_arr = self.filtered_data[:, 0]
            z_exp = self.filtered_data[:, 1] + 1j * self.filtered_data[:, 2]
            model = self.model_var.get()

            # 分离自由/固定参数
            free_vals, lb_list, ub_list, free_idx = [], [], [], []
            all_init = []

            for i, pname in enumerate(self.param_names):
                w = self.param_widgets[pname]
                v0 = float(w['value'].get())
                lo = float(w['lower'].get())
                hi = float(w['upper'].get())
                fixed = w['fixed'].get()

                all_init.append(v0)
                if not fixed:
                    free_vals.append(v0)
                    lb_list.append(lo)
                    ub_list.append(hi)
                    free_idx.append(i)

            if not free_vals:
                messagebox.showwarning("参数错误", "请至少保留一个参数不被固定")
                return

            def full_params(x_free):
                p = list(all_init)
                for idx, val in zip(free_idx, x_free):
                    p[idx] = val
                return p

            def objective(x_free):
                p_full = full_params(x_free)
                if model == "Non-Faradaic":
                    z_model = np.array(
                        [tlm_impedance(f, *p_full[:6]) for f in f_arr])
                else:
                    z_model = np.array(
                        [tlm_impedance(f, *p_full[:6], Rct=p_full[6])
                         for f in f_arr])
                res = z_exp - z_model
                wgt = 1.0 / np.abs(z_exp)                      # 模长加权
                return np.concatenate([np.real(res) * wgt,
                                       np.imag(res) * wgt])

            # 最小二乘
            result = least_squares(
                objective, free_vals,
                bounds=(lb_list, ub_list),
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=8000, method='trf'
            )

            final_p = full_params(result.x)
            self.fit_results = final_p

            # 覆写参数框
            for i, pname in enumerate(self.param_names):
                w = self.param_widgets[pname]
                w['value'].delete(0, tk.END)
                w['value'].insert(0, f"{final_p[i]:.6e}")

            # 计算拟合阻抗 & 残差
            if model == "Non-Faradaic":
                self.z_fit = np.array(
                    [tlm_impedance(f, *final_p[:6]) for f in f_arr])
            else:
                self.z_fit = np.array(
                    [tlm_impedance(f, *final_p[:6], Rct=final_p[6])
                     for f in f_arr])
            self.res_re = np.real(z_exp - self.z_fit)
            self.res_im = np.imag(z_exp - self.z_fit)

            # ── 统计量 ──
            wgt = 1.0 / np.abs(z_exp)
            chi_sq = np.sum((self.res_re * wgt)**2 + (self.res_im * wgt)**2)
            ss_res = np.sum(self.res_re**2 + self.res_im**2)
            ss_tot = np.sum(
                (np.real(z_exp) - np.mean(np.real(z_exp)))**2 +
                (np.imag(z_exp) - np.mean(np.imag(z_exp)))**2
            )
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            n_data = 2 * len(f_arr)                              # real + imag
            n_par = len(free_vals)
            dof = max(n_data - n_par, 1)
            chi_red = chi_sq / dof

            self.fit_stats = {
                'r2': r2, 'chi_sq': chi_sq, 'chi_red': chi_red,
                'cost': result.cost, 'nfev': result.nfev,
                'status': result.message, 'n_param': n_par, 'dof': dof,
            }

            # ── 输出信息 ──
            lines = []
            lines.append("=" * 48)
            lines.append(f"  收敛状态 : {result.message}")
            lines.append(f"  代价函数 : {result.cost:.4e}")
            lines.append(f"  卡方 χ²  = {chi_sq:.6e}")
            lines.append(f"  约化 χ²  = {chi_red:.6e}")
            lines.append(f"  R²       = {r2:.6f}")
            lines.append(f"  自由度   = {dof}")
            lines.append(f"  迭代次数 = {result.nfev}")
            lines.append("-" * 48)
            lines.append("  拟合参数值:")
            lines.append("─" * 48)
            for i, pname in enumerate(self.param_names):
                mark = " (固定)" if self.param_widgets[pname]['fixed'].get() else ""
                lines.append(f"  {pname:<22} = {final_p[i]:.6e}{mark}")
            lines.append("=" * 48)

            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, "\n".join(lines))

            self._plot_fit_results()
            self.plot_nb.select(self.tab_fit)    # 自动切换到 Fit-Result 页
            self.root.after(150, lambda: self._resize_handler(
                self.fig_fit, self.canvas_fit, self.tab_fit))

        except Exception as e:
            messagebox.showerror("拟合失败", f"{e}")
            import traceback
            traceback.print_exc()

    # ── 拟合结果绘图 (6 panels, Fit-Result tab) ──
    def _plot_fit_results(self):
        if self.filtered_data is None or self.z_fit is None:
            return

        self.fig_fit.clear()

        f_r, re_r, im_r = self.raw_data.T
        f_f, re_f, im_f = self.filtered_data.T
        re_fit, im_fit = np.real(self.z_fit), np.imag(self.z_fit)

        gs = self.fig_fit.add_gridspec(2, 3)

        # 1. Nyquist
        ax1 = self.fig_fit.add_subplot(gs[0, 0])
        ax1.plot(re_r, -im_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45,
                 label='Raw')
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5,
                 label='Exp (selected)')
        ax1.plot(re_fit, -im_fit, '-', color=COLORS['fit_line'], lw=1.8,
                 label='TLM Fit')
        ax1.set_xlabel("Z' (\u03a9)")
        ax1.set_ylabel("-Z'' (\u03a9)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')

        # 2. Bode
        ax2 = self.fig_fit.add_subplot(gs[0, 1])
        ax2.loglog(f_r, np.sqrt(re_r**2 + im_r**2), 'o',
                   color=COLORS['raw_marker'], ms=3, alpha=0.45, label='Raw')
        ax2.loglog(f_f, np.sqrt(re_f**2 + im_f**2), 'o',
                   color=COLORS['sel_marker'], ms=5, label='Exp (selected)')
        ax2.loglog(f_f, np.sqrt(re_fit**2 + im_fit**2), '-',
                   color=COLORS['fit_line'], lw=1.8, label='TLM Fit')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("|Z| (\u03a9)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8)
        ax2.grid(True, alpha=0.3, which='both')

        # 3. Real residuals % (残差占实验值百分比)
        ax3 = self.fig_fit.add_subplot(gs[0, 2])
        res_re_pct = 100.0 * self.res_re / np.maximum(np.abs(re_f), 1e-12)
        ax3.semilogx(f_f, res_re_pct, 'o-', color=COLORS['res_re'], ms=5,
                     lw=1.2, label='Real residual %')
        ax3.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("\u0394Z' / Z' (%)")
        ax3.set_title("Real Residuals %", fontweight='bold')
        ax3.legend(framealpha=0.8)
        ax3.grid(True, alpha=0.3)

        # 4. Real vs f
        ax4 = self.fig_fit.add_subplot(gs[1, 0])
        ax4.semilogx(f_r, re_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45,
                     label='Raw')
        ax4.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5,
                     label='Exp (selected)')
        ax4.semilogx(f_f, re_fit, '-', color=COLORS['fit_line'], lw=1.8,
                     label='TLM Fit')
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Z' (\u03a9)")
        ax4.set_title("Real Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8)
        ax4.grid(True, alpha=0.3)

        # 5. -Imag vs f
        ax5 = self.fig_fit.add_subplot(gs[1, 1])
        ax5.semilogx(f_r, -im_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45,
                     label='Raw')
        ax5.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5,
                     label='Exp (selected)')
        ax5.semilogx(f_f, -im_fit, '-', color=COLORS['fit_line'], lw=1.8,
                     label='TLM Fit')
        ax5.set_xlabel("Frequency (Hz)")
        ax5.set_ylabel("-Z'' (\u03a9)")
        ax5.set_title("-Imag Impedance vs f", fontweight='bold')
        ax5.legend(framealpha=0.8)
        ax5.grid(True, alpha=0.3)

        # 6. Imag residuals % (残差占实验值百分比)
        ax6 = self.fig_fit.add_subplot(gs[1, 2])
        res_im_pct = 100.0 * self.res_im / np.maximum(np.abs(im_f), 1e-12)
        ax6.semilogx(f_f, res_im_pct, 's-', color=COLORS['res_im'], ms=5,
                     lw=1.2, label='Imag residual %')
        ax6.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax6.set_xlabel("Frequency (Hz)")
        ax6.set_ylabel("\u0394Z'' / Z'' (%)")
        ax6.set_title("Imag Residuals %", fontweight='bold')
        ax6.legend(framealpha=0.8)
        ax6.grid(True, alpha=0.3)

        self.canvas_fit.draw()

    # ── 导出 ──
    def _export_data(self):
        if self.fit_results is None:
            messagebox.showwarning("无结果", "请先运行拟合")
            return

        fp = filedialog.asksaveasfilename(
            title="保存拟合结果",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if not fp:
            return

        try:
            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                # Sheet 1 - 参数与统计
                rows = []
                rows.append(["模型", self.model_var.get(), ""])
                rows.append(["", "", ""])
                for i, pname in enumerate(self.param_names):
                    fixed = self.param_widgets[pname]['fixed'].get()
                    rows.append([pname, f"{self.fit_results[i]:.6e}",
                                 "固定" if fixed else "自由"])
                rows.append(["", "", ""])
                rows.append(["代价函数 Cost", f"{self.fit_stats['cost']:.4e}", ""])
                rows.append(["卡方 χ²", f"{self.fit_stats['chi_sq']:.6e}", ""])
                rows.append(["约化卡方 χ²_red", f"{self.fit_stats['chi_red']:.6e}", ""])
                rows.append(["R²", f"{self.fit_stats['r2']:.6f}", ""])
                rows.append(["自由度", f"{self.fit_stats['dof']}", ""])
                rows.append(["迭代次数", f"{self.fit_stats['nfev']}", ""])
                rows.append(["收敛状态", self.fit_stats['status'], ""])
                df_stats = pd.DataFrame(rows, columns=["项目", "值", "备注"])
                df_stats.to_excel(writer, sheet_name="Fit_Statistics", index=False)

                # Sheet 2 - 阻抗数据
                f_data = self.filtered_data[:, 0]
                df_imp = pd.DataFrame({
                    "f (Hz)": f_data,
                    "Real_exp (\u03a9)": self.filtered_data[:, 1],
                    "Imag_exp (\u03a9)": self.filtered_data[:, 2],
                    "Real_fit (\u03a9)": np.real(self.z_fit),
                    "Imag_fit (\u03a9)": np.imag(self.z_fit),
                })
                df_imp.to_excel(writer, sheet_name="Impedance_Data", index=False)

            messagebox.showinfo("导出成功", f"结果已保存至:\n{fp}")
        except Exception as e:
            messagebox.showerror("导出失败", f"{e}")


# ───────────────── 入口 ─────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = TLMFitterApp(root)
    root.mainloop()
