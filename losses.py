import tensorflow as tf
import keras.backend as K
import cfg


def loss_cls(y_true, y_pred, weight, r=2, e=1e-6):
    """
        pixel loss
        $L _ { p i x e l } = \frac { 1 } { ( 1 + r ) S } W L _ { p i x e l _ { - } C E }$
        Lpixel = [1/(1+r)S] * W * Lpixel−CE
    """
    y_pred = y_pred * (1 - e) + 0.5 * e

    y_true_pos = y_true[:, :, :, 0]
    y_true_neg = y_true[:, :, :, 1]

    y_pred_pos = y_pred[:, :, :, 0]
    y_pred_neg = y_pred[:, :, :, 1]

    loss_ce_pos = y_true_pos * K.log(y_pred_pos) + (1 - y_true_pos) * K.log(1 - y_pred_pos)
    loss_ce_neg = y_true_neg * K.log(y_pred_neg) + (1 - y_true_neg) * K.log(1 - y_pred_neg)
    loss_ce = -K.sum(weight * (loss_ce_pos + loss_ce_neg))

    S = K.sum(y_true[:, :, :, 0])

    alpha = 1 / ((1 + r) * S)
    # return K.mean(alpha * loss_ce)
    return K.mean(-loss_ce_pos)


def loss_link(y_true, y_pred, weight):
    """
    link loss
    :param y_true:
    :param y_pred:
    :param W:
    :return:
    """
    L_link_ce = y_true * K.log(y_pred)

    L_link_pos = weight * L_link_ce[:, :, :, 0:8]
    L_link_neg = weight * L_link_ce[:, :, :, 8:16]

    L_link = L_link_pos / tf.reduce_sum(weight) + L_link_neg / tf.reduce_sum(weight)
    return K.mean(L_link)


def loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    cls_weight = y_true[:, :, :, 1]
    link_weight = y_true[:, :, :, 10:18]
    y_true_cls_pos = y_true[:, :, :, 0:1]
    y_true_cls_neg = 1 - y_true_cls_pos
    y_true_cls = K.concatenate([y_true_cls_pos, y_true_cls_neg], axis=-1)
    y_true_link_pos = y_true[:, :, :, 2:10]
    y_true_link_neg = 1 - y_true_link_pos
    y_true_link = K.concatenate([K.expand_dims(y_true_link_pos,axis=-1),K.expand_dims(y_true_link_neg,axis=-1)], axis=-1)

    y_true_link_pos = tf.cast(y_true_link_pos,dtype=tf.int32)

    return build_loss(y_true_cls_pos, cls_weight, y_true_link_pos, link_weight, y_pred[:, :, :, :2], y_pred[:, :, :, 2:])


def build_loss(pixel_cls_labels, pixel_cls_weights,
               pixel_link_labels, pixel_link_weights,
               pixel_cls_logits, pixel_link_logits,
               ):
    """
    The loss consists of two parts: pixel_cls_loss + link_cls_loss,
        and link_cls_loss is calculated only on positive pixels
    """

    # 模型结果进行softmax,和reshape，cls进行flat，link进行 pos和neg分开
    def logits_to_scores(pixel_cls_logits, pixel_link_logits):
        def flat_pixel_cls_values(values):
            shape = values.shape.as_list()
            values = tf.reshape(values, shape=[cfg.batch_size, -1, shape[-1]])
            return values

        pixel_cls_scores = tf.nn.softmax(pixel_cls_logits)
        pixel_cls_logits_flatten = \
            flat_pixel_cls_values(pixel_cls_logits)
        pixel_cls_scores_flatten = \
            flat_pixel_cls_values(pixel_cls_scores)

        shape = tf.shape(pixel_link_logits)
        pixel_link_logits = tf.reshape(pixel_link_logits, [cfg.batch_size, shape[1], shape[2], cfg.num_neighbours, 2])

        pixel_link_scores = tf.nn.softmax(pixel_link_logits)

        pixel_pos_scores = pixel_cls_scores[:, :, :, 1]
        link_pos_scores = pixel_link_scores[:, :, :, :, 1]

        return pixel_cls_logits_flatten, pixel_cls_scores_flatten,\
               pixel_link_logits, pixel_link_scores, \
               pixel_pos_scores, link_pos_scores

    pixel_cls_logits_flatten, pixel_cls_scores_flatten, \
    pixel_link_logits1, pixel_link_scores, \
    pixel_pos_scores, link_pos_scores = \
        logits_to_scores(pixel_cls_logits, pixel_link_logits)


    def OHNM_single_image(scores, n_pos, neg_mask):
        """Online Hard Negative Mining.
            scores: the scores of being predicted as negative cls
            n_pos: the number of positive samples
            neg_mask: mask of negative samples
            Return:
                the mask of selected negative samples.
                if n_pos == 0, top 10000 negative samples will be selected.
        """

        def has_pos():
            return n_pos * 3

        def no_pos():
            return tf.constant(10000, dtype=tf.int32)

        n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
        max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

        n_neg = tf.minimum(n_neg, max_neg_entries)
        n_neg = tf.cast(n_neg, tf.int32)

        def has_neg():
            neg_conf = tf.boolean_mask(scores, neg_mask)
            vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
            threshold = vals[-1]  # a negtive value
            selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
            return selected_neg_mask

        def no_neg():
            selected_neg_mask = tf.zeros_like(neg_mask)
            return selected_neg_mask

        selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
        return tf.cast(selected_neg_mask, tf.int32)

    def OHNM_batch(neg_conf, pos_mask, neg_mask):
        selected_neg_mask = []
        for image_idx in range(cfg.batch_size):
            image_neg_conf = neg_conf[image_idx, :]
            image_neg_mask = neg_mask[image_idx, :]
            image_pos_mask = pos_mask[image_idx, :]
            n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
            selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

        selected_neg_mask = tf.stack(selected_neg_mask)
        return selected_neg_mask

    # OHNM on pixel classification task
    pixel_cls_labels_flatten = K.reshape(pixel_cls_labels, [cfg.batch_size, -1])
    pos_pixel_weights_flatten = K.reshape(pixel_cls_weights, [cfg.batch_size, -1])

    pos_mask = K.equal(pixel_cls_labels_flatten, cfg.text_label)
    neg_mask = K.equal(pixel_cls_labels_flatten, cfg.background_label)

    # mask 像素数量
    n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))

    def has_pos():
        # 交叉熵
        pos_mask1 = tf.cast(pos_mask,dtype=tf.int32)

        pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pixel_cls_logits_flatten,
            labels=pos_mask1)

        pixel_neg_scores = pixel_cls_scores_flatten[:, :, 0]
        selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

        pixel_cls_weights = pos_pixel_weights_flatten + \
                            tf.cast(selected_neg_pixel_mask,dtype=tf.float32)
        n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
        loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
        return loss

    pixel_cls_loss = has_pos()

    def no_pos():
        return tf.constant(.0), tf.constant(.0)

    def has_pos():
        pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pixel_link_logits1,
            labels=pixel_link_labels)

        def get_loss(label):
            link_mask = tf.equal(pixel_link_labels, label)
            link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
            n_links = tf.reduce_sum(link_weights)
            loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
            return loss

        neg_loss = get_loss(0)
        pos_loss = get_loss(1)
        return neg_loss, pos_loss

    pixel_neg_link_loss, pixel_pos_link_loss = tf.cond(n_pos > 0, has_pos, no_pos)

    pixel_link_loss = pixel_pos_link_loss

    # return 2.0 * pixel_cls_loss + pixel_link_loss
    return pixel_cls_loss
