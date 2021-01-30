from enum import Flag
import tensorflow as tf
from object_detection.utils import dataset_util
import albumentations as albu
import numpy as np
import json
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import random
import os
import glob

#this code is adapted for LabelMe (Luppa Core)

''' Usage
python generate_tfrecord.py \
       --anotations dir=<path of anotations-from-labelme> \
       [--desired_labels=<comma-separated-labels>] \
       [--test_split=<split_percentage>] \
       [--albumentations_transforms=<albumentations-exported-transformations>] \
       --output_dir=<out_dir>

Required:
  output_dir
  labelme_json

Optional:
  desired_labels: defaults to every label.
  test_split: defaults to 0.3
  albumentations_transforms: defaults to no augmentation.
                             if used, each image will have the original plus the augmented, then the dataset will be twice bigger.
'''

flags = tf.compat.v1.flags
flags.DEFINE_string('output_dir', '', 'Directory to output TFRecords')
flags.DEFINE_string('anotations_dir', '', 'Json exported from LabelMe')
flags.DEFINE_string('desired_labels', '', 'Comma-separated labels to extract from LabelMe')
flags.DEFINE_string('test_split', '0.2', 'Percentage to split the dataset into test [0-1]')
flags.DEFINE_string('albumentations_transforms', '', "Exported Albumentations' transforms to Data Augmentation")
flags.DEFINE_string('base_path', '', "Image Base Path Dir")
FLAGS = flags.FLAGS

def create_tf_example(example):
  height = example['img_height'] # Image height
  width = example['img_width'] # Image width
  filename = example['filename'] # Filename of the image. Empty if image is not from file
  encoded_image_data = example['encoded_img'] # Encoded image bytes
  image_format = example['img_format'] # b'jpeg' or b'png'

  xmins = example['xmins'] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example['xmaxs'] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = example['ymins'] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example['ymaxs'] # List of normalized bottom y coordinates in bounding box (1 per box)

  classes_text = example['classes_text'] # List of string class name of bounding box (1 per box)
  classes = example['classes'] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def create_examples(anotations_dir, desired_labels, test_split, albumentations_transforms, base_path):
  print()
  print("Settings and Creating Augmented Examples...")
  if albumentations_transforms is not None:
    albumentations_transforms = albu.load(albumentations_transforms)
  
  desired_labels = list(set(desired_labels))
  examples = []

  USE_ALL_LABELS = False
  if not desired_labels:
    USE_ALL_LABELS = True

  def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)

  print()
  print('Creating examples')
  base_path = FLAGS.base_path
  files = glob.glob(os.path.join(anotations_dir, "*json"))
  for classification in tqdm(files):
    classification = json.load(open(classification))
    img_path = classification['imagePath']
    img_binary = os.path.join(base_path, img_path)
    img = Image.open(img_binary, "r")
    #width, height = img.size
    width = classification["imageWidth"]
    height = classification["imageHeight"]
    width, height = float(width), float(height)
    filename = classification['imagePath']
    temp_img_name = f'temp_img.{img.format.lower()}'
    img.save(temp_img_name)
    with tf.io.gfile.GFile(temp_img_name, 'rb') as gfile:
      encoded_img = gfile.read()
    os.remove(temp_img_name)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    labels = classification['shapes']
    #talvez um loop para percorrer as lista de objetos
    objetos = labels
    for itens in objetos:
      label = itens["label"]
      if USE_ALL_LABELS:
        if label not in desired_labels:
          desired_labels.append(label)
      if label in desired_labels:
          coords = itens["points"]
          x1, y1, x2, y2 = bounding_box(coords)
          x1 = x1/width #Normalize BBox
          x2 = x2/width #Normalize BBox
          y1 = y1/height #Normalize BBox
          y2 = y2/height #Normalize BBox
          xmins.append(float(x1))
          xmaxs.append(float(x2))
          ymins.append(float(y1))
          ymaxs.append(float(y2))
          class_index = desired_labels.index(label) + 1
          classes_text.append(str.encode(label))
          classes.append(class_index)
          example = dict()
          example['img_height'] = int(height)
          example['img_width'] = int(width)
          example['filename'] = str.encode(filename)
          example['encoded_img'] = encoded_img
          example['img_format'] = str.encode(img.format.lower())
          example['xmins'] = xmins
          example['xmaxs'] = xmaxs
          example['ymins'] = ymins
          example['ymaxs'] = ymaxs
          example['classes_text'] = classes_text
          example['classes'] = classes
          examples.append(example)
    if albumentations_transforms is not None:
      np_img = np.asarray(img)
      # Prepare bbox in Albumentations format: [x_min, y_min, x_max, y_max]
      bboxes = []
      for bbox in list(zip(example['xmins'], example['ymins'], example['xmaxs'], example['ymaxs'])):

        bboxes.append(bbox)
      
      annotations = {'image': np_img, 'bboxes': bboxes, 'classes_text': example['classes_text']}
      augmented_annotations = albumentations_transforms(**annotations)

      # Create new Example
      augmented_img = augmented_annotations['image']
      img = Image.fromarray(np.uint8(augmented_img))
      width, height = img.size
      width, height = float(width), float(height)
      
      splitted_filename = filename.split('.')
      splitted_filename[-2] = splitted_filename[-2] + '-augmented'
      filename = '.'.join(splitted_filename)

      img.save(temp_img_name)
      img = Image.open(temp_img_name)
      with tf.io.gfile.GFile(temp_img_name, 'rb') as gfile:
        encoded_img = gfile.read()
      os.remove(temp_img_name)

      xmins = []
      xmaxs = []
      ymins = []
      ymaxs = []
      for bbox in augmented_annotations['bboxes']:
        # Albumentations format: [x_min, y_min, x_max, y_max]
        xs = [bbox[0], bbox[2]]
        ys = [bbox[1], bbox[3]]
        xmins.append(min(xs))
        xmaxs.append(max(xs))
        ymins.append(min(ys))
        ymaxs.append(max(ys))
        
      classes_text = augmented_annotations['classes_text']
      classes = [desired_labels.index(label.decode('utf-8')) + 1 for label in classes_text]

      example = dict()
      example['img_height'] = int(height)
      example['img_width'] = int(width)
      example['filename'] = str.encode(filename)
      example['encoded_img'] = encoded_img
      example['img_format'] = str.encode(img.format.lower())
      example['xmins'] = xmins
      example['xmaxs'] = xmaxs
      example['ymins'] = ymins
      example['ymaxs'] = ymaxs
      example['classes_text'] = classes_text
      example['classes'] = classes
      examples.append(example)

  random.shuffle(examples)
  split = int(len(examples) * (1 - test_split))

  train_examples = examples[:split]
  test_examples  = examples[split:]

  random.shuffle(train_examples)
  random.shuffle(test_examples)

  class_id_set = set()
  for example in examples:
    for class_id in zip(example['classes'], example['classes_text']):
      class_id_set.add(class_id)

  class_id_set = sorted(class_id_set)

  category_index = {k:v.decode("utf-8") for k, v in class_id_set}
  category_index = json.dumps(category_index)
  with open('category_index.json', 'w') as f:
    f.write(category_index)

  print(),
  print(f'TOTAL EXAMPLES : {len(examples)}')
  print(f'TRAIN EXAMPLES : {len(train_examples)}')
  print(f'TEST EXAMPLES  : {len(test_examples)}')
  print(f'TOTAL CLASSES  : {len(class_id_set)}')
  for class_id in class_id_set:
    print(f'    - {class_id[1].decode("utf-8")} ({class_id[0]})')

  return train_examples, test_examples

def main(_):
  if not FLAGS.output_dir:
    print('Please, set the output_dir parameter')
    exit(-1)
  
  if not FLAGS.anotations_dir:
    print('Please, set the labelme_json parameter')
    exit(-1)
  base_path = FLAGS.base_path
  anotations_dir = FLAGS.anotations_dir
  desired_labels = FLAGS.desired_labels.strip()
  desired_labels = desired_labels.split(',') if desired_labels else []
  test_split = float(FLAGS.test_split)
  albumentations_transforms = FLAGS.albumentations_transforms.strip() \
                              if FLAGS.albumentations_transforms.strip() != '' \
                              else None

  train_examples, test_examples = create_examples(anotations_dir, desired_labels, test_split, albumentations_transforms, base_path)

  train_path = os.path.join(FLAGS.output_dir, 'train.record')
  train_writer = tf.io.TFRecordWriter(train_path)

  test_path = os.path.join(FLAGS.output_dir, 'test.record')
  test_writer = tf.io.TFRecordWriter(test_path)

  print()
  print('Creating TFRecord (TRAINING SET)')
  for example in tqdm(train_examples):
    tf_example = create_tf_example(example)
    train_writer.write(tf_example.SerializeToString())

  print()
  print('Creating TFRecord (TEST SET)')
  for example in tqdm(test_examples):
    tf_example = create_tf_example(example)
    test_writer.write(tf_example.SerializeToString())

  train_writer.close()
  test_writer.close()

if __name__ == '__main__':
  tf.compat.v1.app.run()



def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1:
            raise ValueError(
                "Expected {name} for bbox {bbox} "
                "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))


def check_bboxes(bboxes):
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)