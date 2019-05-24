from glob import glob
import xml.etree.ElementTree as ET
import skimage
from IPython.core.debugger import set_trace
from numpy.random import randint
from itertools import product
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from skimage.color import rgb2gray
from skimage.segmentation import flood
from skimage.measure import find_contours
from numpy.fft import fft




def parse_file(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text))-int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymax').text))-int(float(bbox.find('ymin').text))]
        objects.append(obj_struct)
    return objects


def load_no_bug_annotations(data_type, n_per_image=2000, n_bboxes=50, maxx=800, maxy=800, bbsize=(25, 25), mindist=100):
    annotations = []
    i = 0
    for file in glob(f'../project-data/annotations/{data_type}/*.xml'):
        img_filename = file.replace('.xml', '.jpg').replace('annotations', 'images')
        image = skimage.io.imread(img_filename)
        parsed = parse_file(file)
        locs = np.matrix([ v['bbox'][0:2] for v in parsed ])

        for _ in range(randint(1, n_per_image)):              
            x, y = randint(0, maxx), randint(0, maxy)
            norms, dists = None, None
            
            if len(locs) > 1:
                norms = locs - np.array([x, y])
                dists = np.linalg.norm(norms)
            
            if len(locs) <= 1 or np.min(dists) > mindist:
                annotation = {
                    'filename': img_filename,
                    'xml': file,
                    'bbox': [ x, y, bbsize[0], bbsize[1] ],
                    'bbox_img': image[x:x+bbsize[0], y:y+bbsize[1]]
                }
                if np.sum(image[x:x+bbsize[0], y:y+bbsize[1]]) > 0:
                    annotations.append(annotation)
                
            i += 1
            if i > n_bboxes:
                return annotations
    return annotations
    

def load_bug_annotations(data_type, n_bboxes=50, name='*'):
    annotations = []
    i = 0
    for file in glob(f'../project-data/annotations/{data_type}/{name}.xml'):
        img_filename = file.replace('.xml', '.jpg').replace('annotations', 'images')
        image = skimage.io.imread(img_filename)
        
        parsed = parse_file(file)
        for vaorra in parsed:
            x, y, w, h = vaorra['bbox']
            annotation = {
                'filename': file.replace('.xml', '.jpg').replace('annotations', 'images'),
                'xml': file,
                'bbox': vaorra['bbox'],
                'bbox_img': image[y:y+h, x:x+w]
            }
            annotations.append(annotation)
            
            i += 1
            if i > n_bboxes:
                return annotations
    return annotations


def sliding_window(image, size, overlap=50):
    stepSize = int((size[0] * overlap) / 100)
    for y in range(0, image.shape[0] - size[0], stepSize):
        for x in range(0, image.shape[1] - size[1], stepSize):
            yield (x, y, image[y:y + size[1], x:x + size[0]])

            
def image_from_annotation(annotation):
    image = skimage.io.imread(annotation['filename'])
    x, y, w, h = annotation['bbox']
    return image[y:y+h, x:x+w]
            
    
def pca_normalize_size(gray, desired):
    zeros = np.zeros((desired - len(gray)))
    return np.concatenate((gray, zeros))
    
    
def pca_extract(X_train, n_features=2, sample_length=2700, pca=None, grayscale=True):
    print('PCA based extraction has been started...')
    imgs = np.zeros((len(X_train), sample_length))

    if not grayscale:
        sample_length *= 3
    
    accum = 0
    for i, x in enumerate(X_train):
        img = x['bbox_img']
        flatten_img = (rgb2gray(img) if grayscale else img).flatten()
        norm_img = pca_normalize_size(flatten_img, sample_length)
        imgs[i, :] = np.matrix(norm_img)
        #if i % 1000 == 0:
        #    print(f'{i} out of {len(X_train)}')

    # Apply PCA
    if pca is None:
        pca = PCA(n_components=n_features)
        pca.fit(imgs)
        
    X_train_pca = pca.transform(imgs)
    
    print('PCA based extraction is done...')
    return X_train_pca, pca
    
    
def fd_extract(X_train, n_features=3, tolerance=0.2):
    print('Fourier Descriptor based extraction has been started...')
    
    descriptors = []
    for x in X_train:
        img = rgb2gray(x['bbox_img'])

        w_h, h_h = img.shape[0] // 2, img.shape[1] // 2
        
        res = flood(img, (w_h, h_h), tolerance=tolerance)
        cont = find_contours(res, 0)
        
        if len(cont) > 0:
            sing_cont = [ np.complex(n[0], n[1]) for n in cont[0] ]
            fd = fft(sing_cont)
            
            if len(fd) >= n_features + 2:
                fd_ref = abs(fd[1])
                fd_inv = []
                for j in range(n_features):
                    fd_inv.append(abs(fd[j + 2]) / fd_ref)
                descriptors.append(fd_inv)
            else:
                descriptors.append([32.1] * n_features)
        else:
            descriptors.append([32.1] * n_features)
    
    print('Fourier Descriptor based extraction is done')
    return np.matrix(descriptors)
    
    
def sift_extract(X_train, n_features=3):
    print('SIFT based extraction has been started...')
    X_desc = []

    # Extract descriptors for training images
    surf = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    for x in X_train:
        kp, des = surf.detectAndCompute(x['bbox_img'], None)
        descriptors.append(des)

    # Create BOW dictionary
    BOW = cv2.BOWKMeansTrainer(n_features)
    for dsc in descriptors:
        if dsc is not None:
            BOW.add(dsc)
    dictionary = BOW.cluster()
    bowDictionary = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
    bowDictionary.setVocabulary(dictionary)
    
    # Determine features for each image
    for x in X_train:
        features = surf.detect(x['bbox_img'])
        res = [0.987] * n_features
        if features is not None:
            r = bowDictionary.compute(x['bbox_img'], features)
            if r is not None:
                res = list(r[0])
        X_desc.append(res)
    
    print('SIFT based extraction is done...')
    return np.matrix(X_desc)


def combined_extract(X, pca_model=None):
    X_comb = []
    X_pca, pca_model = pca_extract(X, pca=pca_model)
    X_sift = sift_extract(X)
    X_fd = fd_extract(X)
    
    print('Combining results...')
    for x_pca, x_sift, x_fd in zip(X_pca, X_sift, X_fd):
        # comb = x_fd[0, :].A1
        comb = np.concatenate((x_pca, x_sift[0, :].A1, x_fd[0, :].A1))
        X_comb.append(comb)
    return X_comb, pca_model

    
def visualise_bboxes(bboxes_est, bboxes_ref):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    for x, y, w, h in bboxes_est:
        rect = patches.Rectangle((x, y), w, h,
                            linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
    for x, y, w, h in bboxes_ref:
        rect = patches.Rectangle((x, y), w, h,
                            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        
    return fig, ax


def segment(img, clf, pca_model=None, overlap=50, window=(25, 25), epsilon=0.1, radius=40):
    # Create bounding boxes
    print('Loading windows...')
    X_single = []
    subimg_iter = sliding_window(img, window, overlap=overlap)
    for x, y, subimg in subimg_iter:
        X_single.append({ 'bbox_img': subimg })
        
    # Run classifier
    print('Running classifier...')
    X_comb, _ = combined_extract(X_single, pca_model=pca_model)
    y_pred = clf.predict(X_comb)
    y_pred_prob = clf.predict_proba(X_comb)
    
    # Classify
    print('Classifing bounding boxes...')
    bbox_descs = []
    subimg_iter = sliding_window(img, window, overlap=overlap)
    for i, (x, y, subimg) in enumerate(subimg_iter):
        if y_pred[i] == 1:
            if y_pred_prob[i][0] < epsilon:
                bbox_descs.append({
                    'bbox': np.array([ x, y, window[0], window[1] ]),
                    'prob': y_pred_prob[i][0]
                })
                
    # Filter bounding boxes
    print('Filtering bounding boxes...')
    bboxes = []
    for bbox_desc in bbox_descs:
        best = True
        for bbox_desc_neigh in bbox_descs:
            if np.linalg.norm(bbox_desc_neigh['bbox'][0:2] - bbox_desc['bbox'][0:2]) < radius and\
                bbox_desc_neigh['prob'] < bbox_desc['prob']:
                best = False
                break
                
        if best:
            bboxes.append(bbox_desc['bbox'])
                
    return bboxes
    
    
def create_dataset(dtype, n_bboxes=100):
    X = []
    y = []
    
    no_bug_annotations = load_no_bug_annotations(dtype, n_per_image=50, n_bboxes=n_bboxes * 3)
    bug_annotations = load_bug_annotations(dtype, n_bboxes=n_bboxes)
    for annot in no_bug_annotations:
        X.append(annot)
        y.append(0)
    for annot in bug_annotations:
        X.append(annot)
        y.append(1)
    return X, y
    
    
def area(r):
    """ Calculate area size of the rectangle
    
    Parameters
    ----------
    r: patches.Rectangle
        Rectangle which area will be calculated
        
    Returns
    -------
    : float
        Rectangle area size [px], otherwise None
    """
    r_xmax = max(r.get_x(), r.get_x() + r.get_width())
    r_xmin = min(r.get_x(), r.get_x() + r.get_width())
    r_ymax = max(r.get_y(), r.get_y() + r.get_height())
    r_ymin = min(r.get_y(), r.get_y() + r.get_height()) 
    
    dx = r_xmax - r_xmin
    dy = r_ymax - r_ymin
    
    if dx>=0 and dy>=0:
        return dx*dy


def area_intersection(r1, r2):  # returns None if rectangles don't intersect
    """ Size of an intersection area of two rectangles
    
    Parameters
    ----------
    r1: patches.Rectangle
        First rectangle
    r2: patches.Rectangle
        Second rectangle
        
    Returns
    -------
    : float
        Rectangle area size [px], otherwise None
    """
    a_xmax = max(r1.get_x(), r1.get_x() + r1.get_width())
    b_xmax = max(r2.get_x(), r2.get_x() + r2.get_width())
    a_xmin = min(r1.get_x(), r1.get_x() + r1.get_width())
    b_xmin = min(r2.get_x(), r2.get_x() + r2.get_width())
    a_ymax = max(r1.get_y(), r1.get_y() + r1.get_height())
    b_ymax = max(r2.get_y(), r2.get_y() + r2.get_height())
    a_ymin = min(r1.get_y(), r1.get_y() + r1.get_height())
    b_ymin = min(r2.get_y(), r2.get_y() + r2.get_height()) 
    
    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    
    if dx>=0 and dy>=0:
        return dx*dy
    
def area_union(r1, r2):
    """ Size of an union area of two rectangles
    
    Parameters
    ----------
    r1: patches.Rectangle
        First rectangle
    r2: patches.Rectangle
        Second rectangle
        
    Returns
    -------
    : float
        Rectangle area size [px], otherwise None 
    """
    return area(r1) + area(r2) - area_intersection(r1, r2)


def get_performance_scores(im, annotations, props, treshold, debug=False):
    """ Calculates performance scores
    
    This method calculates performance scores: Precision, Recall, F1-scores
    
    Parameters
    ----------
    im: numpy.ndarray
        Default image with varroas on it (not labeled, nor processed anyhow)
    annotations: List of dictionaries
        This is list of annotations read from XML file provided for the exercise
    props: propertieslist of RegionProperties (skimage.measure.regionprops)
        Properties of our labeled image
    treshold: float
        Treshold for IoU in range [0, 1]
    debug: bool
        By default is False, but can be True in order to plot every iteration where intersection of
        true value and predicted one is found.
    
    Returns
    -------
    precision: float, None
        Precision value is calculated with tp / (tp+fp)
    recall: float, None
        Recall value is calculated with tp / (tp+fn)
    f1_score: float, None
        F1-score is calculated with 2*precision*recall / (precision + recall)
    
    Attributes
    ----------
    tp: int
        True Positives
    fp: int
        False Positives
    fn: int
        False Negative
    i_plot: int
        Counts axes for degugging plot
    """
    tp = 0
    fp = 0
    fn = 0
    
    i_plot = 0
    for i, a in enumerate(annotations):
        rect_real = patches.Rectangle((a['bbox'][0], a['bbox'][1]), a['bbox'][2], a['bbox'][3])

        for j, p in enumerate(props):
            rect_segm = patches.Rectangle((p[0], p[1]), p[2], p[3])   
            
            intersection = area_intersection(rect_real, rect_segm)
            
            if intersection:
                union = area_union(rect_real, rect_segm)
                IoU = intersection/union
                
                if debug:
                    print(f"Image : \n\tIntersection = {intersection}\n\tUnion = {union}\n\tIoU = {IoU:.2f}")
                
                    _ax_num = (i_plot//5, i_plot%5 if i_plot >= 5 else i_plot)
                    _ax_num = _ax_num[1] if rows == 1 else _ax_num
                
                    ax[_ax_num].imshow(im)
                    ax[_ax_num].add_patch(patches.Rectangle((a['bbox'][0], a['bbox'][1]), a['bbox'][2], a['bbox'][3], 
                        linewidth=2, edgecolor='r', facecolor='none'))
                    ax[_ax_num].add_patch(patches.Rectangle((p[0], p[1]), p[2], p[3], 
                        linewidth=2, edgecolor='b', facecolor='none'))
                    ax[_ax_num].set_title(f"Intersection: {intersection} px")
                
                    i_plot += 1
            
                if intersection:
                    if IoU >= treshold:
                        tp += 1
                    else:
                        fn += 1
    
    fp = len(props) - (tp + fn)
    
    print(f"TP FN FP = {tp} {fn} {fp}")
    
    if tp+fp > 0:
        precision = tp / (tp+fp)
        print(f"Precision = {precision:.2f}")
    else:
        precision = None
        print("Precision could not be calculated since number TP + FP is zero!")
    
    if tp+fn > 0:
        recall = tp / (tp+fn)
        print(f"Recall = {recall:.2f}")
    else:
        recall = None
        print("Recall could not be calculated since number TP + FN is zero!")
    
    if precision and recall and precision+recall > 0:
        f1_score = 2*precision*recall / (precision + recall)
        print(f"F1-score = {f1_score:.2f}")
    else:
        f1_score = None
        print("F1-score could not be calculated since number TP is zero!")
    
    return precision, recall, f1_score
