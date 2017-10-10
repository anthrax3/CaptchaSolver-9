import random
from captcha import *
import model
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def new_captcha():
    # define the callback function for captcha
    return "http://cardinfo.gdufe.edu.cn/getCheckpic.action?rand=%f" % (random.random() * 10000)


# set to false when the files are correctly renamed
download = False
if download:
    build_train_data(100, 10, new_captcha)
    exit(0)

captcha_model = model.Model("train_split", "test_split")
captcha_model.train()

im = Image.open(urllib2.urlopen(new_captcha()))
plt.ylim(30, -0.5)
plt.imshow(im)
plt.axis("off")

im = ImageEnhance.Sharpness(im).enhance(3)
im = im.convert("1")
im = flood_filter(im, 4)
im = neighbor4filter(im, 2)
ch, x_start, x_end = split_image(im, 3, 16)
y_min, y_max = y_range(im)
op_min, op_max = y_range(ch[1])
ax = plt.gca()
c1 = patches.Rectangle((x_start[0] - 0.5, y_min - 0.5), x_end[0] - x_start[0], y_max - y_min + 1, alpha=0.3, fill=True, facecolor="red", edgecolor=None)
c2 = patches.Rectangle((x_start[1] - 0.5, y_min + op_min - 0.5), x_end[1] - x_start[1], op_max - op_min + 1, alpha=0.3, fill=True, facecolor="red", edgecolor=None)
c3 = patches.Rectangle((x_start[2] - 0.5, y_min - 0.5), x_end[2] - x_start[2], y_max - y_min + 1,  alpha=0.3, fill=True, facecolor="red", edgecolor=None)
ax.add_patch(c1)
ax.add_patch(c2)
ax.add_patch(c3)
a = captcha_model.predict([ch[i] for i in [0, 2]])
op = get_operator(ch[1], "train_op")
expr = str(a[0]) + " " + op + " " + str(a[1])
ans = eval(expr)
plt.text((plt.xlim()[1] + 0.5) / 2, 25, expr + " = " + str(ans), size=26, ha="center", va="center", color="blue", weight="bold")
