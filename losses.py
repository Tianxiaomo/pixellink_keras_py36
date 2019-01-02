import tensorflow as tf
import keras.backend as K
import cfg


def  quad_loss(y_true, y_pred):
    # loss for inside_score
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    # balance positive and negative samples in an image
    beta = 1 - tf.reduce_mean(labels)
    # first apply sigmoid activation
    predicts = tf.nn.sigmoid(logits)
    # log +epsilon for stable cal
    inside_score_loss = tf.reduce_mean(
        -1 * (beta * labels * tf.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * tf.log(1 - predicts + cfg.epsilon)))
    inside_score_loss *= cfg.lambda_inside_score_loss

    # loss for side_vertex_code
    vertex_logits = y_pred[:, :, :, 1:3]
    vertex_labels = y_true[:, :, :, 1:3]
    vertex_beta = 1 - (tf.reduce_mean(y_true[:, :, :, 1:2])
                       / (tf.reduce_mean(labels) + cfg.epsilon))
    vertex_predicts = tf.nn.sigmoid(vertex_logits)
    pos = -1 * vertex_beta * vertex_labels * tf.log(vertex_predicts +
                                                    cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * tf.log(
        1 - vertex_predicts + cfg.epsilon)
    positive_weights = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
    side_vertex_code_loss = \
        tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * positive_weights) / (
                tf.reduce_sum(positive_weights) + cfg.epsilon)
    side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, :, :, 3:]
    g_true = y_true[:, :, :, 3:]
    vertex_weights = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
    side_vertex_coord_loss = tf.reduce_sum(pixel_wise_smooth_l1norm) / (
            tf.reduce_sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss
    return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss


def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    pixel_wise_smooth_l1norm = (tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        axis=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    shape = tf.shape(g_true)
    delta_xy_matrix = tf.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(square, axis=-1))
    distance *= 4.0
    distance += cfg.epsilon
    return tf.reshape(distance, shape[:-1])


def loss_cls(y_true,y_pred,weight,r = 2,e = 1e-6):
    """
        pixel loss
        $L _ { p i x e l } = \frac { 1 } { ( 1 + r ) S } W L _ { p i x e l _ { - } C E }$
        Lpixel = [1/(1+r)S] * W * Lpixel−CE
    """
    y_pred = y_pred*(1-e) + 0.5*e

    y_true_pos = y_true[:,:,:,0]
    y_true_neg = y_true[:,:,:,1]

    y_pred_pos = y_pred[:,:,:,0]
    y_pred_neg = y_pred[:,:,:,1]

    loss_ce_pos = y_true_pos*K.log(y_pred_pos) + (1-y_true_pos)*K.log(1-y_pred_pos)
    loss_ce_neg = y_true_neg*K.log(y_pred_neg) + (1-y_true_neg)*K.log(1-y_pred_neg)
    loss_ce = -K.sum(weight * (loss_ce_pos + loss_ce_neg))

    S = K.sum(y_true[:,:,:,0])

    alpha = 1/((1 + r)*S)
    # return K.mean(alpha * loss_ce)
    return K.mean(-loss_ce_pos)


def loss_link(y_true,y_pred,weight):
    """
    link loss
    :param y_true:
    :param y_pred:
    :param W:
    :return:
    """
    L_link_ce = y_true * K.log(y_pred)

    L_link_pos = weight * L_link_ce[:,:,:,0:8]
    L_link_neg = weight * L_link_ce[:,:,:,8:16]

    L_link = L_link_pos / tf.reduce_sum(weight) + L_link_neg / tf.reduce_sum(weight)
    return K.mean(L_link)

def loss(y_true,y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    cls_weight = y_true[:,:,:,1]
    link_weight = y_true[:,:,:,10:18]
    y_true_cls_pos = y_true[:,:,:,0:1]
    y_true_cls_neg = 1 - y_true_cls_pos
    y_true_cls = K.concatenate([y_true_cls_pos,y_true_cls_neg],axis=-1)
    y_true_link_pos = y_true[:,:,:,2:10]
    y_true_link_neg = 1 - y_true_link_pos
    y_true_link = K.concatenate([y_true_link_pos,y_true_link_neg],axis=-1)
    return loss_cls(y_true_cls,y_pred[:,:,:,:2],cls_weight)
           # + loss_link(y_true_link , y_pred[:,:,:,2:18], link_weight)

def build_loss(pixel_cls_labels, pixel_cls_weights,
               pixel_link_labels, pixel_link_weights,
               pixel_cls_logits,
               do_summary=True
               ):
    """
    The loss consists of two parts: pixel_cls_loss + link_cls_loss, 
        and link_cls_loss is calculated only on positive pixels
    """

    def logits_to_scores(pixel_cls_logits,pixel_link_logits):
        def flat_pixel_cls_values(values):
            shape = values.shape.as_list()
            values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
            return values
        pixel_cls_scores = tf.nn.softmax(pixel_cls_logits)
        pixel_cls_logits_flatten = \
            flat_pixel_cls_values(pixel_cls_logits)
        pixel_cls_scores_flatten = \
            flat_pixel_cls_values(pixel_cls_scores)

        shape = tf.shape(pixel_link_logits)
        pixel_link_logits = tf.reshape(pixel_link_logits,[shape[0], shape[1], shape[2], cfg.num_neighbours, 2])

        pixel_link_scores = tf.nn.softmax(pixel_link_logits)

        pixel_pos_scores = pixel_cls_scores[:, :, :, 1]
        link_pos_scores = pixel_link_scores[:, :, :, :, 1]

        return pixel_cls_logits_flatten,pixel_cls_scores_flatten,pixel_link_logits,pixel_link_scores,pixel_pos_scores,link_pos_scores

    pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_link_logits, pixel_link_scores, pixel_pos_scores, link_pos_scores = logits_to_scores(pixel_cls_logits)


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
            return n_pos * cfg.max_neg_pos_ratio

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
    pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [cfg.batch_size, -1])
    pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [cfg.batch_size, -1])

    pos_mask = tf.equal(pixel_cls_labels_flatten, cfg.text_label)
    neg_mask = tf.equal(pixel_cls_labels_flatten, cfg.background_label)

    n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))

    def no_pos():
        return tf.constant(.0);

    def has_pos():
        pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits= pixel_cls_logits_flatten,
            labels=tf.cast(pos_mask, dtype=tf.int32))

        pixel_neg_scores = pixel_cls_scores_flatten[:, :, 0]
        selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

        pixel_cls_weights = pos_pixel_weights_flatten + \
                            tf.cast(selected_neg_pixel_mask, tf.float32)
        n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
        loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
        return loss

    pixel_cls_loss = has_pos()

    def no_pos():
        return tf.constant(.0), tf.constant(.0)

    def has_pos():
        pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pixel_link_logits,
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

    pixel_link_loss = pixel_pos_link_loss + pixel_neg_link_loss * cfg.pixel_link_neg_loss_weight_lambda
