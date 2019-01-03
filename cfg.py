import os

text_label = 1
background_label = 0
ignore_label = -1

num_neighbours = 8

bbox_border_width = 1

score_map_shape = [1280,720]
img_scall = 4

train_task_id = '1T640'
initial_epoch = 0
epoch_num = 24
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
load_weights = False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

num_train_img = 1000
num_val_img = 500

validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
# if max_train_img_size == 256:
#     batch_size = 8
# elif max_train_img_size == 384:
#     batch_size = 4
# elif max_train_img_size == 512:
#     batch_size = 2
# else:
batch_size = 1
steps_per_epoch = num_train_img// batch_size
validation_steps = num_val_img // batch_size

data_dir = '../Data'

origin_image_dir_name = 'ICDAR2015/Challenge4/ch4_training_images'
origin_txt_dir_name = 'ICDAR2015/Challenge4/ch4_training_localization_transcription_gt'

train_image_dir_name = 'ICDAR2015/Challenge4/ch4_training_images'
train_label_dir_name = 'PixelLink_py36/label_train'

show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id

val_image_dir_name = 'ICDAR2015/Challenge4/ch4_training_images'
val_label_dir_name = 'PixelLink_py36/label_train'

gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(4, 0, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_models/east_model_weights_%s.h5'\
                                % train_task_id

pixel_threshold = 0.7
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True

gpu = 1

pixel_cls_border_weight_lambda = 1.0