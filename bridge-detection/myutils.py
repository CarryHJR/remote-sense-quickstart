def vis_coco(image_path, boxes, coco, ax=None, figsize=(6, 6), dst=[]):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image,ImageFont, ImageDraw
    import os
    class_names = [v['name'] for k,v in coco.cats.items()]
    class_names.insert(0, 'background')
    hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
    import colorsys
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if not dst:
        dst = class_names
    image = Image.open(image_path)
    # font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
    font_path = '/home/yons/workplace/python/trash-detection/tools/simhei.ttf'
    font = ImageFont.truetype(font=font_path,size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = (np.shape(image)[0] + np.shape(image)[1]) // 416
    
    # boxes
    # xmin ymin xmax ymax class score
    for i in range(len(boxes)):
        cls_id = boxes[i][4]
        if class_names[cls_id] not in dst:
            continue
        # coco xmin ymin width height
        left, top, right, bottom = boxes[i][:4]
        right = left+right
        bottom = top+bottom
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # draw box
        label = '{}'.format(class_names[cls_id])
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            _ = draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[cls_id])
        _ = draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[cls_id])
        _ = draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    _ = ax.imshow(image)


def vis_predict(image_path, result, ax=None, figsize=(10,10), thres=0.3, log=False, class_names=None, dst=[], title=None, cls_path=None):
    from PIL import Image,ImageFont, ImageDraw
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # class_names if for subset trainning
    if class_names is None:
        import codecs
        class_names = codecs.open(cls_path, 'r', 'utf-8').read().strip().split()
        class_names = [c.strip() for c in class_names]
    hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
    import colorsys
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


    if len(dst) < 1:
        dst = class_names
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set(xticks=[], yticks=[], aspect=1)
    
    image = Image.open(image_path)
    font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
    font = ImageFont.truetype(font=font_path,size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = (np.shape(image)[0] + np.shape(image)[1]) // 416

    bbox_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    inds = np.where(bboxes[:, -1] > thres)[0]



    for idx in inds:
        c = labels[idx]
        predicted_class = class_names[c]
        if predicted_class not in dst:
            continue
        if log:
            print(bboxes[idx], predicted_class)
        score = bboxes[idx][-1]
        # top, left, bottom, right = bboxes[idx][:4]
        left, top, right, bottom = bboxes[idx][:4]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[class_names.index(predicted_class)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[class_names.index(predicted_class)])

        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    if title is not None:
        draw = ImageDraw.Draw(image)
        w, h = image.size
        font = ImageFont.truetype(font=font_path,size=np.floor(4e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        draw.text(np.array([w/2, 0.8*h]), title, fill='red', font=font)
        del draw

    _ = ax.imshow(image)

