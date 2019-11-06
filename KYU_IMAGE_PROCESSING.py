import os
import cv2
import glob
import shutil
from joblib import Parallel, delayed

base_path  = './dir'
image_path = os.path.join(base_path, 'dir')
arg_path = os.path.join(base_path, 'dir')

def make_subdir(path):
  if os.path.exists(path):
      shutil.rmtree(path)

def get_red_free_image(pImg):
  redFreeImage  = pImg[:, :, 1]
  return redFreeImage

def save_image(pImg, path, srcFName):
  fileName = os.path.join(path, srcFName)
  cv2.imwrite(fileName, pImg)

def do_get_red_free_image(pFileName):
  _, fileName = os.path.split(pFileName)
  prefix, ext = fileName.split('.')
  imgOrg = cv2.imread(pFileName)

  newFileName = prefix + '_C.' + ext
  imgNew = imgOrg
  save_image(imgNew, arg_path, newFileName)

  newFileName = prefix + '_G.' + ext
  imgNew = get_red_free_image(imgOrg)
  save_image(imgNew, arg_path, newFileName)

def run_main():
  query    = os.path.join(image_path, '*.JPG')
  fileList = glob.glob(query) 
  fileList.sort()

  make_subdir(arg_path)

  Parallel(n_jobs=-1)(delayed(do_get_red_free_image)
                      (fileName) for fileName in fileList)

if __name__ == '__main__':
  run_main()
