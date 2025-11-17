# %matplotlib widget
import matplotlib.gridspec as gridspec
s, n = 0, 1
a = x_test[s: s+n]
b = y_test[s: s+n]
b = torch.from_numpy(b).cuda()
c = torch.from_numpy(a)
with torch.no_grad():
    d = model(c.cuda())
p = d.cpu().numpy()
e = b.cpu().numpy()
radius = 1000
root = np.mean(p, axis=1)
xroot, yroot, zroot = root[0][0], root[0][1], root[0][2]

# mpjpe(d, b).item()
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 7)
axes = []
for i in range(4):
    ax = fig.add_subplot(gs[i, 0])
    ax.set_aspect('equal')
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    gesture = x_test_original[s, :, i * 2: i * 2 + 2] * -1
    root_2d = np.mean(gesture, axis=0)
    r = 300
    ax.set_xlim([-r + root_2d[0], r + root_2d[0]])
    ax.set_ylim([-r + root_2d[1], r + root_2d[1]])
    parents = dataset.skeleton().parents()
    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
    colors_2d = np.full(gesture.shape[0], 'black')
    colors_2d[joints_right_2d] = 'red'
    lines = []
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        if len(parents) == gesture.shape[0]:
            ax.plot([gesture[j, 0], gesture[j_parent, 0]],
                    [gesture[j, 1], gesture[j_parent, 1]], color='pink')
    points = ax.scatter(*gesture.T, 10, color=colors_2d,
                        edgecolors='white', zorder=10)
    axes.append(ax)
for i in range(3):
    ax = fig.add_subplot(gs[1: 3, 1 + i * 2: 3 + i * 2], projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-radius + xroot, radius + xroot])
    ax.set_ylim([-radius + yroot, radius + yroot])
    ax.set_zlim([-radius + zroot, radius + zroot])
    axes.append(ax)
t = p[0]
axes[-3].scatter3D(t[:, 0], t[:, 1], t[:, 2], cmap='Blues', c='b', s=3)  #绘制散点图
for i, parent in enumerate(dataset.skeleton().parents()):
    if parent == -1:
        continue
    axes[-3].plot3D(t[[i, parent], 0], t[[i, parent], 1], t[[i, parent], 2], color='blue')

t = e[0]
axes[-2].scatter3D(t[:, 0], t[:, 1], t[:, 2], cmap='Blues', c='r', s=3)  #绘制散点图
for i, parent in enumerate(dataset.skeleton().parents()):
    if parent == -1:
        continue
    axes[-2].plot3D(t[[i, parent], 0], t[[i, parent], 1], t[[i, parent], 2], color='red') 

t = p[0]
axes[-1].scatter3D(t[:, 0], t[:, 1], t[:, 2], cmap='Blues', s=3, c='b')  #绘制散点图
for i, parent in enumerate(dataset.skeleton().parents()):
    if parent == -1:
        continue
    axes[-1].plot3D(t[[i, parent], 0], t[[i, parent], 1], t[[i, parent], 2], color='blue')
t = e[0]
axes[-1].scatter3D(t[:, 0], t[:, 1], t[:, 2], cmap='Blues', s=3, c='r')  #绘制散点图
for i, parent in enumerate(dataset.skeleton().parents()):
    if parent == -1:
        continue
    axes[-1].plot3D(t[[i, parent], 0], t[[i, parent], 1], t[[i, parent], 2], color='red') 

plt.savefig('Figure1.png', dpi=500)
plt.show()