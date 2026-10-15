# Exported notebook code; may contain IPython commands.
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# تعریف بردارها
x1 = np.array([1.5, 2.0])
x2 = np.array([-1.0, 1.5])
x3 = np.array([2.0, -0.5])
h = np.array([0.8, -0.6])
y = np.array([0.7, 0.3])

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# --- فضای ورودی ---
axs[0].axhline(0, color='black', linewidth=1, alpha=0.5)
axs[0].axvline(0, color='black', linewidth=1, alpha=0.5)
axs[0].quiver(0, 0, x1[0], x1[1], angles='xy', scale_units='xy', scale=1,
              color='#1f77b4', width=0.015, label=r'$x_1 = [1.5,\ 2.0]$')
axs[0].quiver(0, 0, x2[0], x2[1], angles='xy', scale_units='xy', scale=1,
              color='#ff7f0e', width=0.015, label=r'$x_2 = [-1.0,\ 1.5]$')
axs[0].quiver(0, 0, x3[0], x3[1], angles='xy', scale_units='xy', scale=1,
              color='#2ca02c', width=0.015, label=r'$x_3 = [2.0,\ -0.5]$')
axs[0].set_xlim(-2.5, 2.5)
axs[0].set_ylim(-2.5, 2.5)
axs[0].set_aspect('equal')
axs[0].set_title(r'Input Space ($x_t$)' + '\nNo Bounds', fontsize=14, fontweight='bold')
axs[0].set_xlabel(r'$x^{(1)}$')
axs[0].set_ylabel(r'$x^{(2)}$')
axs[0].legend(loc='upper left')
axs[0].grid(True, linestyle='--', alpha=0.6)

# --- فضای وضعیت پنهان ---
axs[1].axhline(0, color='black', linewidth=1, alpha=0.5)
axs[1].axvline(0, color='black', linewidth=1, alpha=0.5)
axs[1].quiver(0, 0, h[0], h[1], angles='xy', scale_units='xy', scale=1,
              color='#9467bd', width=0.015, label=r'$h = [0.8,\ -0.6]$')
rect = patches.Rectangle((-1, -1), 2, 2, linewidth=1.5, edgecolor='red',
                         facecolor='none', linestyle='--',
                         label=r'$\tanh$ Bounds $[-1,\ 1]$')
axs[1].add_patch(rect)
axs[1].set_xlim(-1.5, 1.5)
axs[1].set_ylim(-1.5, 1.5)
axs[1].set_aspect('equal')
axs[1].set_title(r'Hidden State Space ($h_t$)' + '\n' + r'Bounded by $\tanh$ in $[-1,\ 1]$',
                 fontsize=14, fontweight='bold')
axs[1].set_xlabel(r'$h^{(1)}$')
axs[1].set_ylabel(r'$h^{(2)}$')
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.6)

# --- فضای خروجی ---
axs[2].axhline(0, color='black', linewidth=1, alpha=0.5)
axs[2].axvline(0, color='black', linewidth=1, alpha=0.5)
axs[2].quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1,
              color='#d62728', width=0.015, label=r'$y = [0.7,\ 0.3]$')
prob_line_x = np.linspace(0, 1, 100)
axs[2].plot(prob_line_x, 1 - prob_line_x, color='black', linestyle='--',
            alpha=0.7, label=r'$y^{(1)} + y^{(2)} = 1$')
axs[2].set_xlim(-0.2, 1.2)
axs[2].set_ylim(-0.2, 1.2)
axs[2].set_aspect('equal')
axs[2].set_title(r'Output Space ($y_t$)' + '\nSoftmax Probabilities',
                 fontsize=14, fontweight='bold')
axs[2].set_xlabel(r'$y^{(1)}$')
axs[2].set_ylabel(r'$y^{(2)}$')
axs[2].legend(loc='upper right')
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# رسم گراف محاسباتی تمرین ۲
fig, ax = plt.subplots(figsize=(12, 6))

bbox_x = dict(boxstyle='round,pad=1', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=1', facecolor='#9467bd', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=1', facecolor='#2ca02c', alpha=0.25)
bbox_h = dict(boxstyle='round,pad=1', facecolor='#d62728', alpha=0.25)

ax.text(1.5, 4.2, 'Input Space\n$x_t$',
        bbox=bbox_x, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(1.5, 1.2, 'Previous Hidden\n$h_{t-1}$',
        bbox=bbox_h_prev, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(6.0, 2.7, 'Pre-activation\n$z_t$\n$z_t = W_{xz}x_t + W_{hz}h_{t-1}$',
        bbox=bbox_z, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(10.5, 2.7, 'New Hidden\n$h_t$\n' + r'$h_t = \tanh(z_t)$',
        bbox=bbox_h, ha='center', va='center', fontsize=12, fontweight='bold')

# فلش از x_t به z_t
ax.annotate('', xy=(4.2, 3.2), xytext=(2.7, 4.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1f77b4'))
ax.text(3.3, 3.9, r'$W_{xz}$', fontsize=15, color='blue', fontweight='bold', ha='center')

# فلش از h_{t-1} به z_t
ax.annotate('', xy=(4.2, 2.2), xytext=(2.7, 1.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#9467bd'))
ax.text(3.3, 1.4, r'$W_{hz}$', fontsize=15, color='purple', fontweight='bold', ha='center')

# فلش از z_t به h_t
ax.annotate('', xy=(8.7, 2.7), xytext=(7.8, 2.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#d62728'))
ax.text(8.25, 3.0, r'$\tanh$', fontsize=15, color='red', fontweight='bold', ha='center')

ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis('off')
plt.title('RNN Computational Graph (Exercise 2)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# %%
W_xz = np.array([[0.5, 1.2],
                 [-0.8, 0.4]])

W_hz = np.array([[0.9, -0.3],
                 [0.6, 0.7]])

x_sample = x1  # بردار x1
h_prev = np.array([0.0, 0.0])  # h0

z_t = W_xz @ x_sample + W_hz @ h_prev
h_1 = np.tanh(z_t)

print(f'W_xz =\n{W_xz}')
print(f'\nW_hz =\n{W_hz}')
print(f'\nx_1 = {x_sample}')
print(f'h_0 = {h_prev}')
print('-' * 40)
print(f'z_t = W_xz @ x_1 + W_hz @ h_0 = {z_t}')
print(f'h_1 = tanh(z_t) = {h_1}')

# %%
# رسم گراف محاسباتی تمرین ۳ (اضافه شدن مسیر h_t -> y_t)
fig, ax = plt.subplots(figsize=(14, 6))

bbox_x = dict(boxstyle='round,pad=1', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=1', facecolor='#9467bd', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=1', facecolor='#2ca02c', alpha=0.25)
bbox_h = dict(boxstyle='round,pad=1', facecolor='#d62728', alpha=0.25)
bbox_y = dict(boxstyle='round,pad=1', facecolor='#ff7f0e', alpha=0.25)

ax.text(1.5, 4.2, 'Input Space\n$x_t$',
        bbox=bbox_x, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(1.5, 1.2, 'Previous Hidden\n$h_{t-1}$',
        bbox=bbox_h_prev, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(6.0, 2.7, 'Pre-activation\n$z_t$\n$z_t = W_{xz}x_t + W_{hz}h_{t-1}$',
        bbox=bbox_z, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(10.0, 2.7, 'New Hidden\n$h_t$\n' + r'$h_t = \tanh(z_t)$',
        bbox=bbox_h, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(13.5, 2.7, 'Output\n$y_t$\n' + r'$y_t = \mathrm{softmax}(W_{hy} h_t)$',
        bbox=bbox_y, ha='center', va='center', fontsize=12, fontweight='bold')

# فلش از x_t به z_t
ax.annotate('', xy=(4.2, 3.2), xytext=(2.7, 4.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1f77b4'))
ax.text(3.3, 3.9, r'$W_{xz}$', fontsize=14, color='blue', fontweight='bold', ha='center')

# فلش از h_{t-1} به z_t
ax.annotate('', xy=(4.2, 2.2), xytext=(2.7, 1.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#9467bd'))
ax.text(3.3, 1.4, r'$W_{hz}$', fontsize=14, color='purple', fontweight='bold', ha='center')

# فلش از z_t به h_t
ax.annotate('', xy=(8.2, 2.7), xytext=(7.8, 2.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#d62728'))
ax.text(8.0, 3.2, r'$\tanh$', fontsize=14, color='red', fontweight='bold', ha='center')

# فلش از h_t به y_t (تمرین ۳)
ax.annotate('', xy=(12.0, 2.7), xytext=(11.5, 2.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#ff7f0e'))
ax.text(11.75, 3.2, r'$\mathrm{softmax}(W_{hy} \cdot)$', fontsize=12, color='#ff7f0e', fontweight='bold', ha='center')

ax.set_xlim(0, 15)
ax.set_ylim(0, 5.5)
ax.axis('off')
plt.title('RNN Computational Graph (Exercises 2 & 3)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# %%
# تعریف ماتریس W_hy
W_hy = np.array([[1.0, 0.3],
                 [-0.5, 0.8]])

# تابع softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))  # برای پایداری عددی
    return e_x / e_x.sum()

# محاسبه خروجی y_1
logits_1 = W_hy @ h_1
y_1 = softmax(logits_1)

print(f'W_hy =\n{W_hy}')
print(f'\nh_1 = {h_1}')
print('-' * 40)
print(f'W_hy @ h_1 = {logits_1}')
print(f'y_1 = softmax(W_hy @ h_1) = {y_1}')
print(f'\nمجموع y_1 = {y_1.sum():.6f}  (باید ۱ باشد)')

# %%
# رسم گراف محاسباتی کامل RNN (تمرین ۴) — فرمول‌ها درون فضاها
fig, ax = plt.subplots(figsize=(14, 6))

bbox_x = dict(boxstyle='round,pad=1', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=1', facecolor='#9467bd', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=1', facecolor='#2ca02c', alpha=0.25)
bbox_h = dict(boxstyle='round,pad=1', facecolor='#d62728', alpha=0.25)
bbox_y = dict(boxstyle='round,pad=1', facecolor='#ff7f0e', alpha=0.25)

ax.text(1.5, 4.2, '$x_t$',
        bbox=bbox_x, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1.5, 1.2, '$h_{t-1}$',
        bbox=bbox_h_prev, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(6.0, 2.7, '$z_t$\n\n$z_t = W_{xz} \\cdot x_t + W_{hz} \\cdot h_{t-1}$',
        bbox=bbox_z, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(10.0, 2.7, '$h_t$\n\n$h_t = \\tanh(z_t)$',
        bbox=bbox_h, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(13.5, 2.7, '$y_t$\n\n$y_t = \\mathrm{softmax}(W_{hy} \\cdot h_t)$',
        bbox=bbox_y, ha='center', va='center', fontsize=11, fontweight='bold')

# فلش از x_t به z_t
ax.annotate('', xy=(4.2, 3.2), xytext=(2.5, 4.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1f77b4'))
ax.text(3.2, 3.9, r'$W_{xz}$', fontsize=14, color='blue', fontweight='bold', ha='center')

# فلش از h_{t-1} به z_t
ax.annotate('', xy=(4.2, 2.2), xytext=(2.5, 1.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#9467bd'))
ax.text(3.2, 1.4, r'$W_{hz}$', fontsize=14, color='purple', fontweight='bold', ha='center')

# فلش از z_t به h_t
ax.annotate('', xy=(8.3, 2.7), xytext=(7.8, 2.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#d62728'))
ax.text(8.05, 3.3, r'$\tanh$', fontsize=14, color='red', fontweight='bold', ha='center')

# فلش از h_t به y_t
ax.annotate('', xy=(12.0, 2.7), xytext=(11.5, 2.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#ff7f0e'))
ax.text(11.75, 3.3, r'$W_{hy}$', fontsize=14, color='#ff7f0e', fontweight='bold', ha='center')

ax.set_xlim(0, 15)
ax.set_ylim(0, 5.5)
ax.axis('off')
plt.title('RNN Architecture with Formulas (Exercise 4)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# %%
# تمرین ۵: پردازش مرحله‌به‌مرحله سه بردار ورودی

# بردارهای ورودی (از تمرین ۱)
inputs = [x1, x2, x3]

# وضعیت پنهان اولیه
h_prev = np.array([0.0, 0.0])

print('=' * 50)
print('پردازش مرحله‌به‌مرحله RNN')
print('=' * 50)
print(f'h_0 = {h_prev}')
print()

# پردازش هر مرحله
hidden_states = [h_prev]
for t, x_t in enumerate(inputs, 1):
    z_t = W_xz @ x_t + W_hz @ h_prev
    h_t = np.tanh(z_t)
    hidden_states.append(h_t)
    
    print(f'--- مرحله {t} ---')
    print(f'x_{t} = {x_t}')
    print(f'z_{t} = W_xz @ x_{t} + W_hz @ h_{t-1} = {z_t}')
    print(f'h_{t} = tanh(z_{t}) = {h_t}')
    print()
    
    h_prev = h_t

# محاسبه خروجی نهایی y_3
h_3 = hidden_states[3]
logits_3 = W_hy @ h_3
y_3 = softmax(logits_3)

print('--- خروجی نهایی ---')
print(f'W_hy @ h_3 = {logits_3}')
print(f'y_3 = softmax(W_hy @ h_3) = {y_3}')
print(f'مجموع y_3 = {y_3.sum():.6f}')

# %%
# تمرین ۶: رسم فضاهای LSTM (بدون اتصالات)

# بردارهای جدید برای بخش LSTM
lstm_h_prev = np.array([0.5, -0.3])
lstm_c_prev = np.array([-0.2, 0.9])
lstm_x1 = np.array([1.0, 0.5])
lstm_x2 = np.array([-0.7, 1.2])
lstm_x3 = np.array([0.3, -1.0])

fig, axs = plt.subplots(1, 5, figsize=(22, 5))

spaces = [
    (axs[0], r'$x_t$', [lstm_x1, lstm_x2, lstm_x3],
     [r'$x_1=[1.0,\ 0.5]$', r'$x_2=[-0.7,\ 1.2]$', r'$x_3=[0.3,\ -1.0]$'],
     ['#1f77b4', '#ff7f0e', '#2ca02c'], (-1.5, 1.5)),
    (axs[1], r'$h_{t-1}$', [lstm_h_prev],
     [r'$h_{t-1}=[0.5,\ -0.3]$'], ['#9467bd'], (-1.0, 1.0)),
    (axs[2], r'$c_{t-1}$', [lstm_c_prev],
     [r'$c_{t-1}=[-0.2,\ 0.9]$'], ['#e377c2'], (-1.5, 1.5)),
    (axs[3], r'$z_t$', [], [], [], (-1.5, 1.5)),
    (axs[4], r'$y_t$', [], [], [], (-0.2, 1.2)),
]

for ax, title, vecs, labels, colors, lims in spaces:
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    for v, lbl, c in zip(vecs, labels, colors):
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color=c, width=0.02, label=lbl)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    if vecs:
        ax.legend(loc='best', fontsize=9)

plt.suptitle('LSTM Vector Spaces (Exercise 6) — No Connections Yet',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
# تمرین ۸: رسم تمام فضاها — شامل فضاهای قبلی (z_t, y_t) و افزودن f و i
fig, ax = plt.subplots(figsize=(15, 8))

bbox_x = dict(boxstyle='round,pad=0.7', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=0.7', facecolor='#9467bd', alpha=0.25)
bbox_c_prev = dict(boxstyle='round,pad=0.7', facecolor='#e377c2', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=0.7', facecolor='#2ca02c', alpha=0.25)
bbox_f = dict(boxstyle='round,pad=0.7', facecolor='#d62728', alpha=0.25)
bbox_i = dict(boxstyle='round,pad=0.7', facecolor='#ff7f0e', alpha=0.25)
bbox_y = dict(boxstyle='round,pad=0.7', facecolor='#bcbd22', alpha=0.25)

# Inputs (left)
ax.text(1.5, 6.5, '$x_t$', bbox=bbox_x, ha='center', va='center',
        fontsize=14, fontweight='bold')
ax.text(1.5, 4.0, '$h_{t-1}$', bbox=bbox_h_prev, ha='center', va='center',
        fontsize=14, fontweight='bold')
ax.text(1.5, 1.5, '$c_{t-1}$', bbox=bbox_c_prev, ha='center', va='center',
        fontsize=14, fontweight='bold')

# Existing intermediate spaces from RNN section
ax.text(7, 6.5, '$z_t$\n(Pre-activation\nFeature)',
        bbox=bbox_z, ha='center', va='center', fontsize=12, fontweight='bold')

# NEW spaces added in Exercise 8
ax.text(7, 4.0, '$f$\n(Forget Gate)\nNEW',
        bbox=bbox_f, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(7, 1.5, '$i$\n(Input Gate)\nNEW',
        bbox=bbox_i, ha='center', va='center', fontsize=12, fontweight='bold')

# Existing output space (was present from Ex 1-5)
ax.text(12, 4.0, '$y_t$\n(Output)',
        bbox=bbox_y, ha='center', va='center', fontsize=12, fontweight='bold')

# Highlight new boxes with thick frame
for (cx, cy) in [(7, 4.0), (7, 1.5)]:
    rect = patches.Rectangle((cx-1.05, cy-0.7), 2.1, 1.4, linewidth=2.5,
                             edgecolor='red', facecolor='none', linestyle='-',
                             zorder=5)
    ax.add_patch(rect)

ax.text(7, 7.7, 'Pre-existing spaces', color='black', fontsize=10,
        ha='center', style='italic')
ax.text(7, 0.3, 'New gate spaces (Exercise 8)', color='red', fontsize=11,
        ha='center', fontweight='bold')

ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')
plt.title('Exercise 8 — All Spaces So Far + New Gate Spaces $f$ and $i$',
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# %%
# تمرین ۹: نمودار اختصاصی — اتصال x_t و h_{t-1} به z_t, f, i با فرمول‌های جمع
fig, ax = plt.subplots(figsize=(14, 8))

bbox_x = dict(boxstyle='round,pad=0.6', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=0.6', facecolor='#9467bd', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=0.6', facecolor='#2ca02c', alpha=0.25)
bbox_f = dict(boxstyle='round,pad=0.6', facecolor='#d62728', alpha=0.25)
bbox_i = dict(boxstyle='round,pad=0.6', facecolor='#ff7f0e', alpha=0.25)
bbox_main_mem = dict(boxstyle='round,pad=0.6', facecolor='#e377c2', alpha=0.20)

# Inputs
ax.text(1.5, 6, '$x_t$', bbox=bbox_x, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1.5, 2, '$h_{t-1}$', bbox=bbox_h_prev, ha='center', va='center', fontsize=14, fontweight='bold')

# Three spaces with addition formulas inside
ax.text(7, 7, '$z_t$\n$z_t = W_{xz}\\, x_t + W_{hz}\\, h_{t-1}$',
        bbox=bbox_z, ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(7, 4, '$f$\n$f = W_{xf}\\, x_t + W_{hf}\\, h_{t-1}$',
        bbox=bbox_f, ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(7, 1, '$i$\n$i = W_{xi}\\, x_t + W_{hi}\\, h_{t-1}$',
        bbox=bbox_i, ha='center', va='center', fontsize=11, fontweight='bold')

# Hint at "main memory" target via tanh from z_t
ax.text(12, 4, "to main memory $c_t$\n(after $\\tanh$)",
        bbox=bbox_main_mem, ha='center', va='center', fontsize=11, style='italic')

# Arrows from x_t to z_t, f, i
for ty, mat in [(7, 'W_{xz}'), (4, 'W_{xf}'), (1, 'W_{xi}')]:
    ax.annotate('', xy=(5.3, ty), xytext=(2.5, 6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#1f77b4', alpha=0.8))
    midx = (2.5 + 5.3) / 2
    midy = (6 + ty) / 2
    ax.text(midx, midy + 0.15, f'${mat}$', fontsize=11, color='#1f77b4', fontweight='bold', ha='center')

# Arrows from h_{t-1} to z_t, f, i
for ty, mat in [(7, 'W_{hz}'), (4, 'W_{hf}'), (1, 'W_{hi}')]:
    ax.annotate('', xy=(5.3, ty), xytext=(2.5, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#9467bd', alpha=0.8))
    midx = (2.5 + 5.3) / 2
    midy = (2 + ty) / 2
    ax.text(midx, midy - 0.15, f'${mat}$', fontsize=11, color='#9467bd', fontweight='bold', ha='center')

# tanh arrow from z_t toward main memory
ax.annotate('', xy=(10.5, 4.4), xytext=(8.7, 7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2ca02c'))
ax.text(9.6, 5.9, r'$\tanh$', fontsize=13, color='#2ca02c', fontweight='bold', ha='center')

ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')
plt.title('Exercise 9 — Connections to $z_t$, $f$, $i$ with addition formulas',
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# %%
# تمرین ۱۰: نمودار اختصاصی — ساخت حافظه جدید c_t
fig, ax = plt.subplots(figsize=(14, 7))

bbox_c_prev = dict(boxstyle='round,pad=0.5', facecolor='#e377c2', alpha=0.25)
bbox_z = dict(boxstyle='round,pad=0.5', facecolor='#2ca02c', alpha=0.25)
bbox_f = dict(boxstyle='round,pad=0.5', facecolor='#d62728', alpha=0.25)
bbox_i = dict(boxstyle='round,pad=0.5', facecolor='#ff7f0e', alpha=0.25)
bbox_ct = dict(boxstyle='round,pad=0.6', facecolor='#e377c2', alpha=0.45)
bbox_op = dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)

# Source spaces
ax.text(1, 5.5, '$f$', bbox=bbox_f, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1, 3.0, '$c_{t-1}$', bbox=bbox_c_prev, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1, 0.5, '$i$', bbox=bbox_i, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1, -2.0, '$z_t$', bbox=bbox_z, ha='center', va='center', fontsize=14, fontweight='bold')

# Sigmoid / tanh applied
ax.text(4, 5.5, r'$\sigma(f)$', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0'))
ax.text(4, 0.5, r'$\sigma(i)$', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe5cc'))
ax.text(4, -2.0, r'$\tanh(z_t)$', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4f4d4'))

# Element-wise multiply nodes
ax.text(6.8, 4.25, r'$\odot$', ha='center', va='center', fontsize=18,
        bbox=bbox_op, fontweight='bold')
ax.text(6.8, -0.75, r'$\odot$', ha='center', va='center', fontsize=18,
        bbox=bbox_op, fontweight='bold')

# Sum node
ax.text(9.5, 1.75, r'$+$', ha='center', va='center', fontsize=20,
        bbox=bbox_op, fontweight='bold')

# Final c_t
ax.text(12.5, 1.75,
        '$c_t$\n$c_t = \\sigma(f)\\odot c_{t-1}$\n$+\\, \\sigma(i)\\odot \\tanh(z_t)$',
        bbox=bbox_ct, ha='center', va='center', fontsize=11, fontweight='bold')

def arr(x1, y1, x2, y2, color='black', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color))

# f -> sigmoid -> mul (top)
arr(1.6, 5.5, 3.3, 5.5, color='#d62728')
arr(4.7, 5.5, 6.3, 4.5, color='#d62728')
# c_{t-1} -> mul (top)
arr(1.7, 3.0, 6.3, 4.0, color='#e377c2')
# i -> sigmoid -> mul (bottom)
arr(1.6, 0.5, 3.3, 0.5, color='#ff7f0e')
arr(4.7, 0.5, 6.3, -0.5, color='#ff7f0e')
# z_t -> tanh -> mul (bottom)
arr(1.6, -2.0, 3.3, -2.0, color='#2ca02c')
arr(4.7, -2.0, 6.3, -1.0, color='#2ca02c')
# muls -> +
arr(7.3, 4.25, 9.0, 2.0, color='black')
arr(7.3, -0.75, 9.0, 1.5, color='black')
# + -> c_t
arr(10.0, 1.75, 11.4, 1.75, color='black', lw=2.2)

ax.set_xlim(-0.5, 15)
ax.set_ylim(-3.5, 7)
ax.axis('off')
plt.title('Exercise 10 — Building the new cell state $c_t$',
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# %%
# تمرین ۱۱: نمودار اختصاصی — ساخت دروازه خروجی o و وضعیت پنهان جدید h_t
fig, ax = plt.subplots(figsize=(14, 7))

bbox_x = dict(boxstyle='round,pad=0.5', facecolor='#1f77b4', alpha=0.25)
bbox_h_prev = dict(boxstyle='round,pad=0.5', facecolor='#9467bd', alpha=0.25)
bbox_ct = dict(boxstyle='round,pad=0.5', facecolor='#e377c2', alpha=0.30)
bbox_o = dict(boxstyle='round,pad=0.5', facecolor='#8c564b', alpha=0.25)
bbox_ht = dict(boxstyle='round,pad=0.6', facecolor='#9467bd', alpha=0.45)
bbox_op = dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)

# Sources
ax.text(1, 5.5, '$x_t$', bbox=bbox_x, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1, 3.0, '$h_{t-1}$', bbox=bbox_h_prev, ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1, 0.5, '$c_t$', bbox=bbox_ct, ha='center', va='center', fontsize=14, fontweight='bold')

# o space with addition formula
ax.text(5.5, 4.25, '$o$\n$o = W_{xo}\\, x_t + W_{ho}\\, h_{t-1}$',
        bbox=bbox_o, ha='center', va='center', fontsize=11, fontweight='bold')

# sigmoid(o) and tanh(c_t)
ax.text(8.5, 4.25, r'$\sigma(o)$', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8d4c8'))
ax.text(5.5, 0.5, r'$\tanh(c_t)$', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fbe1ed'))

# Element-wise multiply
ax.text(11, 2.4, r'$\odot$', ha='center', va='center', fontsize=18,
        bbox=bbox_op, fontweight='bold')

# Final h_t
ax.text(13.5, 2.4, '$h_t$\n$h_t = \\sigma(o) \\odot \\tanh(c_t)$',
        bbox=bbox_ht, ha='center', va='center', fontsize=11, fontweight='bold')

def arr(x1, y1, x2, y2, color='black', lw=1.8, label=None, lx=None, ly=None, lcolor=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color))
    if label is not None:
        ax.text(lx, ly, label, fontsize=11, color=lcolor or color,
                fontweight='bold', ha='center')

# x_t -> o   (W_xo)
arr(1.5, 5.5, 4.4, 4.6, color='#1f77b4',
    label=r'$W_{xo}$', lx=2.9, ly=5.3, lcolor='#1f77b4')
# h_{t-1} -> o  (W_ho)
arr(1.7, 3.0, 4.4, 4.0, color='#9467bd',
    label=r'$W_{ho}$', lx=2.9, ly=3.2, lcolor='#9467bd')
# o -> sigmoid(o)
arr(7.2, 4.25, 7.9, 4.25, color='#8c564b')
# sigmoid(o) -> mul
arr(9.1, 4.25, 10.55, 2.7, color='#8c564b')
# c_t -> tanh(c_t)
arr(1.6, 0.5, 4.4, 0.5, color='#e377c2')
# tanh(c_t) -> mul
arr(6.6, 0.5, 10.55, 2.1, color='#e377c2')
# mul -> h_t
arr(11.4, 2.4, 12.5, 2.4, color='black', lw=2.2)

ax.set_xlim(-0.5, 16)
ax.set_ylim(-1, 7)
ax.axis('off')
plt.title('Exercise 11 — Output gate $o_t$ and new hidden state $h_t$',
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# %%
# تمرین ۱۲: نمودار کامل LSTM با گره‌های ⊙ و + به‌صورت صریح
fig, ax = plt.subplots(figsize=(18, 9))

bx = dict(boxstyle='round,pad=0.5', facecolor='#1f77b4', alpha=0.22)
bh = dict(boxstyle='round,pad=0.5', facecolor='#9467bd', alpha=0.22)
bc = dict(boxstyle='round,pad=0.5', facecolor='#e377c2', alpha=0.22)
bz = dict(boxstyle='round,pad=0.5', facecolor='#2ca02c', alpha=0.22)
bf = dict(boxstyle='round,pad=0.5', facecolor='#d62728', alpha=0.22)
bi = dict(boxstyle='round,pad=0.5', facecolor='#ff7f0e', alpha=0.22)
bo = dict(boxstyle='round,pad=0.5', facecolor='#8c564b', alpha=0.22)
bct_new = dict(boxstyle='round,pad=0.6', facecolor='#e377c2', alpha=0.45)
bht_new = dict(boxstyle='round,pad=0.6', facecolor='#9467bd', alpha=0.45)
by_box = dict(boxstyle='round,pad=0.6', facecolor='#bcbd22', alpha=0.30)
bop = dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor='black', linewidth=1.5)

# Inputs
ax.text(0.7, 6.5, '$x_t$', bbox=bx, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.7, 3.5, '$h_{t-1}$', bbox=bh, ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.7, 0.5, '$c_{t-1}$', bbox=bc, ha='center', va='center', fontsize=12, fontweight='bold')

# Pre-activation / gate spaces
ax.text(4.5, 7.5, '$z_t$\n$W_{xz}x_t + W_{hz}h_{t-1}$',
        bbox=bz, ha='center', va='center', fontsize=9.5, fontweight='bold')
ax.text(4.5, 5.5, '$f$\n$W_{xf}x_t + W_{hf}h_{t-1}$',
        bbox=bf, ha='center', va='center', fontsize=9.5, fontweight='bold')
ax.text(4.5, 3.5, '$i$\n$W_{xi}x_t + W_{hi}h_{t-1}$',
        bbox=bi, ha='center', va='center', fontsize=9.5, fontweight='bold')
ax.text(4.5, 1.5, '$o$\n$W_{xo}x_t + W_{ho}h_{t-1}$',
        bbox=bo, ha='center', va='center', fontsize=9.5, fontweight='bold')

# Activations
ax.text(7.5, 7.5, r'$\tanh$', ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4f4d4'))
ax.text(7.5, 5.5, r'$\sigma$', ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax.text(7.5, 3.5, r'$\sigma$', ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5cc'))
ax.text(7.5, 1.5, r'$\sigma$', ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8d4c8'))

# muls for forget and input
ax.text(9.5, 4.5, r'$\odot$', ha='center', va='center', fontsize=15, bbox=bop, fontweight='bold')
ax.text(9.5, 6.5, r'$\odot$', ha='center', va='center', fontsize=15, bbox=bop, fontweight='bold')

# + node
ax.text(11, 5.5, r'$+$', ha='center', va='center', fontsize=17, bbox=bop, fontweight='bold')

# c_t
ax.text(12.8, 5.5, '$c_t$', bbox=bct_new, ha='center', va='center', fontsize=12, fontweight='bold')

# tanh(c_t)
ax.text(14.4, 5.5, r'$\tanh$', ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fbe1ed'))

# mul for output gate
ax.text(15.7, 3.5, r'$\odot$', ha='center', va='center', fontsize=15, bbox=bop, fontweight='bold')

# h_t
ax.text(17.0, 3.5, '$h_t$', bbox=bht_new, ha='center', va='center', fontsize=12, fontweight='bold')

# y_t
ax.text(17.0, 1.5, '$y_t$\n$\\mathrm{softmax}(W_{hy}\\, h_t)$',
        bbox=by_box, ha='center', va='center', fontsize=10, fontweight='bold')

def arr(x1, y1, x2, y2, color='black', lw=1.5, alpha=1.0,
        label=None, lx=None, ly=None, lcolor=None, lsize=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color, alpha=alpha))
    if label is not None:
        ax.text(lx, ly, label, fontsize=lsize, color=lcolor or color,
                fontweight='bold', ha='center')

# x_t -> {z, f, i, o}
for ty, mat in [(7.5, 'W_{xz}'), (5.5, 'W_{xf}'), (3.5, 'W_{xi}'), (1.5, 'W_{xo}')]:
    arr(1.2, 6.5, 3.0, ty, color='#1f77b4', alpha=0.7,
        label=f'${mat}$', lx=2.0, ly=(6.5+ty)/2 + 0.15, lcolor='#1f77b4', lsize=8)
# h_{t-1} -> {z, f, i, o}
for ty, mat in [(7.5, 'W_{hz}'), (5.5, 'W_{hf}'), (3.5, 'W_{hi}'), (1.5, 'W_{ho}')]:
    arr(1.2, 3.5, 3.0, ty, color='#9467bd', alpha=0.7,
        label=f'${mat}$', lx=2.0, ly=(3.5+ty)/2 - 0.15, lcolor='#9467bd', lsize=8)

# z_t -> tanh
arr(6.0, 7.5, 7.0, 7.5, color='#2ca02c')
# f -> sigma
arr(6.0, 5.5, 7.0, 5.5, color='#d62728')
# i -> sigma
arr(6.0, 3.5, 7.0, 3.5, color='#ff7f0e')
# o -> sigma
arr(6.0, 1.5, 7.0, 1.5, color='#8c564b')

# tanh(z_t) -> mul input
arr(8.0, 7.5, 9.2, 6.7, color='#2ca02c')
# c_{t-1} -> mul forget
arr(1.2, 0.5, 9.2, 4.3, color='#e377c2', alpha=0.6)
# sigma(f) -> mul forget
arr(8.0, 5.5, 9.2, 4.7, color='#d62728')
# sigma(i) -> mul input
arr(8.0, 3.5, 9.2, 6.3, color='#ff7f0e', alpha=0.6)

# mul forget -> +
arr(9.8, 4.5, 10.7, 5.3, color='black')
# mul input -> +
arr(9.8, 6.5, 10.7, 5.7, color='black')
# + -> c_t
arr(11.3, 5.5, 12.4, 5.5, color='black', lw=2)
# c_t -> tanh
arr(13.2, 5.5, 14.0, 5.5, color='#e377c2')
# tanh(c_t) -> mul output
arr(14.7, 5.2, 15.4, 3.8, color='#e377c2')
# sigma(o) -> mul output
arr(8.0, 1.5, 15.4, 3.2, color='#8c564b', alpha=0.5)
# mul output -> h_t
arr(16.05, 3.5, 16.6, 3.5, color='black', lw=2)
# h_t -> y_t
arr(17.0, 3.0, 17.0, 2.0, color='#bcbd22', lw=2,
    label=r'$W_{hy}, \mathrm{softmax}$', lx=18.3, ly=2.5, lcolor='#bcbd22', lsize=9)

ax.set_xlim(-0.5, 19.5)
ax.set_ylim(-0.5, 9)
ax.axis('off')
plt.title('Exercise 12 — Complete LSTM with explicit $\\odot$ and $+$ nodes',
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# %%
# تمرین ۱۲: محاسبه عددی کامل LSTM

# تعریف ماتریس‌ها
lstm_W_xz = np.array([[0.4, 0.7], [-0.6, 0.3]])
lstm_W_hz = np.array([[0.8, -0.2], [0.5, 0.6]])
lstm_W_xf = np.array([[0.3, -0.5], [0.9, 0.2]])
lstm_W_hf = np.array([[-0.4, 0.7], [0.1, 0.8]])
lstm_W_xi = np.array([[0.6, 0.1], [-0.3, 0.9]])
lstm_W_hi = np.array([[0.2, -0.6], [0.7, 0.4]])
lstm_W_xo = np.array([[-0.2, 0.8], [0.5, -0.4]])
lstm_W_ho = np.array([[0.3, 0.6], [-0.7, 0.1]])
lstm_W_hy = np.array([[1.0, 0.3], [-0.5, 0.8]])

# تابع sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# بردارهای ورودی (از تمرین ۶)
lstm_inputs = [lstm_x1, lstm_x2, lstm_x3]

# مقادیر اولیه
h_t = lstm_h_prev.copy()
c_t = lstm_c_prev.copy()

print('=' * 60)
print('پردازش مرحله‌به‌مرحله LSTM')
print('=' * 60)
print(f'h_0 = {h_t}')
print(f'c_0 = {c_t}')
print()

for t, x_t in enumerate(lstm_inputs, 1):
    print(f'--- مرحله {t} (x_{t} = {x_t}) ---')
    
    # z_t (feature extraction)
    z = lstm_W_xz @ x_t + lstm_W_hz @ h_t
    print(f'  z = W_xz @ x + W_hz @ h = {z}')
    
    # Forget gate
    f = lstm_W_xf @ x_t + lstm_W_hf @ h_t
    f_gate = sigmoid(f)
    print(f'  f = W_xf @ x + W_hf @ h = {f}')
    print(f'  σ(f) = {f_gate}')
    
    # Input gate
    i = lstm_W_xi @ x_t + lstm_W_hi @ h_t
    i_gate = sigmoid(i)
    print(f'  i = W_xi @ x + W_hi @ h = {i}')
    print(f'  σ(i) = {i_gate}')
    
    # Output gate
    o = lstm_W_xo @ x_t + lstm_W_ho @ h_t
    o_gate = sigmoid(o)
    print(f'  o = W_xo @ x + W_ho @ h = {o}')
    print(f'  σ(o) = {o_gate}')
    
    # New cell state
    c_t = f_gate * c_t + i_gate * np.tanh(z)
    print(f'  c_t = σ(f)⊙c_{t-1} + σ(i)⊙tanh(z) = {c_t}')
    
    # New hidden state
    h_t = o_gate * np.tanh(c_t)
    print(f'  h_t = σ(o)⊙tanh(c_t) = {h_t}')
    print()

# خروجی نهایی y_3
logits = lstm_W_hy @ h_t
y_final = softmax(logits)
print('--- خروجی نهایی ---')
print(f'W_hy @ h_3 = {logits}')
print(f'y_3 = softmax(W_hy @ h_3) = {y_final}')
print(f'مجموع y_3 = {y_final.sum():.6f}')

# %%
# تمرین ۱۲ (ردیابی بصری): تحول حافظه (c_t) و وضعیت پنهان (h_t) در سه گام
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# بازسازی پردازش با ذخیره مقادیر در هر گام
h_track = [lstm_h_prev.copy()]
c_track = [lstm_c_prev.copy()]
y_track = []
all_gates = []

h_t = lstm_h_prev.copy()
c_t = lstm_c_prev.copy()
for x_t in lstm_inputs:
    z = lstm_W_xz @ x_t + lstm_W_hz @ h_t
    f_pre = lstm_W_xf @ x_t + lstm_W_hf @ h_t
    i_pre = lstm_W_xi @ x_t + lstm_W_hi @ h_t
    o_pre = lstm_W_xo @ x_t + lstm_W_ho @ h_t
    f_g = sigmoid(f_pre); i_g = sigmoid(i_pre); o_g = sigmoid(o_pre)
    c_t = f_g * c_t + i_g * np.tanh(z)
    h_t = o_g * np.tanh(c_t)
    h_track.append(h_t.copy()); c_track.append(c_t.copy())
    y_track.append(softmax(lstm_W_hy @ h_t))
    all_gates.append((f_g.copy(), i_g.copy(), o_g.copy()))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: cell state evolution
ax0 = axes[0]
ax0.axhline(0, color='black', lw=0.5, alpha=0.5)
ax0.axvline(0, color='black', lw=0.5, alpha=0.5)
colors_c = ['#888888', '#ff7f0e', '#2ca02c', '#d62728']
labels_c = ['$c_0$', '$c_1$', '$c_2$', '$c_3$']
for v, c, lbl in zip(c_track, colors_c, labels_c):
    ax0.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
               color=c, width=0.012, label=f'{lbl} = [{v[0]:.3f}, {v[1]:.3f}]')
ax0.set_xlim(-1.5, 1.5); ax0.set_ylim(-1.5, 1.5); ax0.set_aspect('equal')
ax0.grid(True, ls='--', alpha=0.5)
ax0.set_title('Cell State $c_t$ Evolution', fontsize=13, fontweight='bold')
ax0.legend(loc='best', fontsize=9)

# Panel 2: hidden state evolution
ax1 = axes[1]
ax1.axhline(0, color='black', lw=0.5, alpha=0.5)
ax1.axvline(0, color='black', lw=0.5, alpha=0.5)
rect = mpatches.Rectangle((-1, -1), 2, 2, linewidth=1.5, edgecolor='red',
                          facecolor='none', linestyle='--',
                          label=r'$\tanh$ Bound $[-1, 1]$')
ax1.add_patch(rect)
colors_h = ['#888888', '#1f77b4', '#9467bd', '#e377c2']
labels_h = ['$h_0$', '$h_1$', '$h_2$', '$h_3$']
for v, c, lbl in zip(h_track, colors_h, labels_h):
    ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
               color=c, width=0.012, label=f'{lbl} = [{v[0]:.3f}, {v[1]:.3f}]')
ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_aspect('equal')
ax1.grid(True, ls='--', alpha=0.5)
ax1.set_title('Hidden State $h_t$ Evolution', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=9)

# Panel 3: gate activations across time-steps
ax2 = axes[2]
gate_names = [r'$\sigma(f)$', r'$\sigma(i)$', r'$\sigma(o)$']
times = ['t=1', 't=2', 't=3']
data = np.zeros((3, 3))
for ti, (fg, ig, og) in enumerate(all_gates):
    data[0, ti] = fg.mean()
    data[1, ti] = ig.mean()
    data[2, ti] = og.mean()
im = ax2.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax2.set_xticks(range(3)); ax2.set_xticklabels(times, fontsize=11)
ax2.set_yticks(range(3)); ax2.set_yticklabels(gate_names, fontsize=12)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                 color='black', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title('Mean Gate Activation per Time-step', fontsize=13, fontweight='bold')

plt.suptitle('Exercise 12 — Numerical Trace Through LSTM',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print a compact summary
print('\n' + '=' * 60)
print('Final Summary')
print('=' * 60)
for t in range(1, 4):
    print(f't={t}:  c_{t} = {c_track[t]},   h_{t} = {h_track[t]}')
    print(f'       y_{t} = {y_track[t-1]},  sum = {y_track[t-1].sum():.6f}')

