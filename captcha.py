import urllib2
from PIL import Image, ImageEnhance
import cv2
import numpy
import os


def get_samples(dir_name, n, url_fun):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in range(n):
        img = urllib2.urlopen(url_fun()).read()
        temp_file = open(dir_name + "/%d.jpg" % i, 'wb')
        temp_file.write(img)
        temp_file.close()


def neighbor4filter(img, p):
    pixels = img.load()
    w, h = img.size
    for x in range(w):
        pixels[x, 0] = 255
        pixels[x, h - 1] = 255
    for y in range(h):
        pixels[0, y] = 255
        pixels[w - 1, y] = 255
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not pixels[x, y]:
                count = 1
                if not pixels[x - 1, y]:
                    count = count + 1
                if not pixels[x + 1, y]:
                    count = count + 1
                if not pixels[x, y - 1]:
                    count = count + 1
                if not pixels[x, y + 1]:
                    count = count + 1
                if count <= p:
                    if count == 2:
                        if not pixels[x - 1, y] and not pixels[x - 2, y] and pixels[x - 1, y - 1] and pixels[
                                    x - 1, y + 1]:
                            continue
                        if not pixels[x + 1, y] and not pixels[x + 2, y] and pixels[x + 1, y - 1] and pixels[
                                    x + 1, y + 1]:
                            continue
                        if not pixels[x, y - 1] and not pixels[x, y - 2] and pixels[x - 1, y - 1] and pixels[
                                    x + 1, y - 1]:
                            continue
                        if not pixels[x, y + 1] and not pixels[x, y + 2] and pixels[x - 1, y + 1] and pixels[
                                    x + 1, y + 1]:
                            continue
                    pixels[x, y] = 255
    return img


def neighbor8filter(img, p):
    pixels = img.load()
    w, h = img.size
    # remove edges
    for x in range(w):
        pixels[x, 0] = 255
        pixels[x, h - 1] = 255
    for y in range(h):
        pixels[0, y] = 255
        pixels[w - 1, y] = 255
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not pixels[x, y]:
                count = 0
                for m in range(x - 1, x + 2):
                    for n in range(y - 1, y + 2):
                        if not pixels[m, n]:
                            count = count + 1
                if count <= p:
                    pixels[x, y] = 255
            else:
                count = 0
                for m in range(x - 1, x + 2):
                    for n in range(y - 1, y + 2):
                        if not pixels[m, n]:
                            count = count + 1
                if count >= 7:
                    pixels[x, y] = 0
    return img


def flood_filter(img, p):
    tmp_img = cv2.cvtColor(numpy.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = tmp_img.shape[:2]
    mask = numpy.zeros((h + 2, w + 2), numpy.uint8)
    color = 1
    for x in range(w):
        for y in range(h):
            if (tmp_img[y, x] == 0).all():
                cv2.floodFill(tmp_img, mask, (x, y), (0, 0, color))
                color = color + 1
    color_count = numpy.zeros(255, numpy.uint)
    for x in range(h):
        for y in range(w):
            if (tmp_img[x, y] != 255).any():
                color_count[tmp_img[x, y][2]] += 1
    for x in range(h):
        for y in range(w):
            if (tmp_img[x, y] != 255).any() and color_count[tmp_img[x, y][2]] <= p:
                tmp_img[x, y] = (255, 255, 255)
    for x in range(h):
        for y in range(w):
            if tmp_img[x, y][2] < 255:
                tmp_img[x, y] = (0, 0, 0)
    return Image.fromarray(tmp_img).convert('1')


def split_image(img, p, max_len):
    pixels = img.load()
    w, h = img.size
    x_num = numpy.zeros(w, numpy.int)
    # count the number of black pixels for all x
    for y in range(h):
        for x in range(w):
            if not pixels[x, y]:
                x_num[x] += 1
    x_start = numpy.zeros(3, numpy.int)  # the start of a character
    x_end = numpy.zeros(3, numpy.int)  # the end of a character
    char_num = 0
    ptr = 0
    normal_split = False
    for x in range(w):
        if x_num[x] != 0:
            if x_num[x - 1] == 0:
                if ptr == 0:
                    x_start[char_num] = x
                elif x - ptr >= p or (x - ptr >= 2 and ((char_num == 2 and (
                                numpy.mean(abs(x_num[x:x + 4] - 4)) <= 1.5 or numpy.mean(
                            abs(x_num[ptr - 4:ptr] - 4)) <= 1.5)) or char_num == 1)):
                    if x_start[char_num] != 0:
                        if char_num == 2:
                            normal_split = True
                        x_end[char_num] = ptr + 1
                        char_num = char_num + 1
                    if char_num >= 3:
                        break
                    x_start[char_num] = x
            ptr = x
            if x_start[char_num] != 0 and ptr - x_start[char_num] >= max_len:
                x_end[char_num] = ptr
                char_num = char_num + 1
                if char_num >= 3:
                    break
                if x_num[x + 1] != 0:
                    x_start[char_num] = x + 1
                    ptr = x + 1
                else:
                    ptr = 0

    y_min, y_max = y_range(img)
    if 4 <= x_num[x_end[2] - 1] <= 6:
        t = x_end[2] - 1
        for i in range(x_start[2], t)[::-1]:
            if x_num[i] != 4:
                if 4 < x_num[i] <= 6 and x_num[i - 1] == 4:
                    continue
                else:
                    t = i
                    break
        if x_num[t] > 2:
            t = t + 1
        elif x_num[t - 1] <= 2:
            t = t - 1
            if x_num[t - 1] <= 2:
                t = t - 1
        if normal_split and x_end[2] - t <= 4:
            t = x_end[2]
        x_end[2] = t
    chars = []
    for i in range(3):
        chars.append(img.crop((x_start[i], y_min, x_end[i], y_max + 1)))
    return chars, x_start, x_end


def process(dir_name, num):
    if not os.path.exists(dir_name + "_filter"):
        os.mkdir(dir_name + "_filter")
    if not os.path.exists(dir_name + "_op"):
        os.mkdir(dir_name + "_op")
    if not os.path.exists(dir_name + "_split"):
        os.mkdir(dir_name + "_split")
    for i in range(num):
        im = Image.open(dir_name + "/%d.jpg" % i)
        im = ImageEnhance.Sharpness(im).enhance(3)
        im = im.convert('1')
        im = flood_filter(im, 4)
        im = neighbor4filter(im, 2)
        im.save(dir_name + "_filter/%d.jpg" % i)
        ch, x_start, x_end = split_image(im, 3, 16)
        for j in range(3):
            if j == 1:
                resize_y(ch[j]).save(dir_name + "_op/%d%d.jpg" % (i, j))
            else:
                ch[j].save(dir_name + "_split/%d%d.jpg" % (i, j))


def build_train_data(num_train, num_test, url_fun):
    get_samples("train", num_train, url_fun)
    get_samples("test", num_test, url_fun)
    process("train", num_train)
    process("test", num_test)


def get_operator(img, train_op):
    if not os.path.exists(train_op):
        return None
    files = os.listdir(train_op)
    op_set = {}
    for name in files:
        fullname = os.path.join(train_op, name)
        im = Image.open(fullname)
        im = im.convert("1")
        op_name = name.replace(".jpg", "")
        op_set[op_name] = im
    img = resize_y(img)
    pixels = img.load()
    w, h = img.size
    min_count = w * h
    result = "+"
    for op_name, op_img in op_set.iteritems():
        op_pixels = op_img.load()
        op_w, op_h = op_img.size
        count = 0
        min_width = w if w < op_w else op_w
        min_height = h if h < op_h else op_h
        for y in range(min_height):
            for x in range(min_width):
                if pixels[x, y] != op_pixels[x, y]:
                    count = count + 1
        if count < min_count:
            min_count = count
            result = op_name
    return result


def resize_y(img):
    w, h = img.size
    y_min, y_max = y_range(img)
    return img.crop((0, y_min, w, y_max + 1))


def y_range(img):
    pixels = img.load()
    w, h = img.size
    y_min = 0
    y_max = 0
    for y in range(h):
        for x in range(w):
            if not pixels[x, y]:
                if y_min == 0:
                    y_min = y
                if y > y_max:
                    y_max = y
    return y_min, y_max
