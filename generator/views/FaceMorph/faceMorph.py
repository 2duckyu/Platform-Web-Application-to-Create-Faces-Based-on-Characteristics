import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
from imutils import face_utils
from . import align_images
import random
import shutil

# Read points from text file
# 그러나 우린 바로 매칭해줄거라 안씀!!!!!!
def readPoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):
    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            list4.append((dictionary1[pt1], dictionary1[pt2], dictionary1[pt3]))

    dictionary1 = {}
    return list4


def make_delaunay(f_w, f_h, theList, img1, img2):
    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary.
    theList = theList.tolist()
    points = [(int(x[0]), int(x[1])) for x in theList]
    dictionary = {x[0]: x[1] for x in list(zip(points, range(76)))}

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)

    # Return the list.
    return list4

def random_select(gender, features_list):
    """
    :param gender: 성별 "F"이나 "M"
    :param features_list: 특징 리스트
    :return: 리턴 없어
    """
    for index, feature in enumerate(features_list):
        from_ = "generator/views/FaceMorph/sample/"+gender+"/" + feature + "/" + random.choice(
            os.listdir("generator/views/FaceMorph/sample/"+gender+"/" + feature + "/"))
        to_ = "generator/views/FaceMorph/images"
        shutil.copy(from_, to_)

def main(result_gender, result_feature):
    random_select(result_gender, result_feature)
    align_images.abc_mart("~/projects/myproject/generator/views/FaceMorph/images/", "~/projects/myproject/generator/views/FaceMorph/morph_images/")    # 첫 번째 파라미터는 정렬을 할 이미지 경로, 두 번째 파라미터는 정렬을 마친 후 합성을 할 이미지 경로

    '''filename = input()
    result_list = []
    filename_list = filename.split(' ')
    points_box = []
    print(filename_list)'''

    filename_list = []
    for morph_img in os.listdir("generator/views/FaceMorph/morph_images/"):
        filename_list.append(morph_img)
        #print(morph_img)

    #print(filename_list)

    points_box = []
    result_list = []

    for index in range(1, len(filename_list)):
        if len(result_list) < 1:
            filename1 = filename_list[0]
        else:
            filename1 = result_list[index - 2]
        filename2 = filename_list[index]
        alpha = 1 - float(index / (index + 1))
        # 각 이미지의 점 좌표를 저장하기 위한 리스트 선언
        # points_box = []

        # Read images
        img1 = cv2.imread("generator/views/FaceMorph/morph_images/" + filename1);
        img2 = cv2.imread("generator/views/FaceMorph/morph_images/" + filename2);

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # print(generate_face_correspondences(img1, img2)[5])

        # opencv에서 ESC 키입력 상수
        ESC_KEY = 27

        '''
        RGB > BGR or BGR > RGB 변환 
        dlib는 RGB 형태로 이미지를 사용하고
        openCV는 BGR 형태이므로 B와 R을 바꿔주는 함수가 필요하다.
        '''

        def swapRGB2BGR(rgb):
            r, g, b = cv2.split(img)
            bgr = cv2.merge([b, g, r])
            return bgr

        '''
        매개변수가 3개여야 한다.
        '''
        # if len(sys.argv) != 1:
        #     print(
        #         "Give the path to the trained shape predictor model as the first "
        #         "argument and then the directory containing the facial images.\n"
        #         "For example, if you are in the python_examples folder then "
        #         "execute this program by running:\n"
        #         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        #         "You can download a trained facial shape predictor from:\n"
        #         "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        #     exit()

        # 랜드마크 파일 경로
        predictor_path = "generator/views/FaceMorph/shape_predictor_68_face_landmarks.dat"
        # 이미지 경로
        faces_folder_path = "generator/views/FaceMorph/morph_images/"

        # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
        detector = dlib.get_frontal_face_detector()
        # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성
        predictor = dlib.shape_predictor(predictor_path)

        # 이미지를 화면에 표시하기 위한 openCV 윈도 생성
        #cv2.namedWindow('Face')

        # 두번째 매개변수로 지정한 폴더를 싹다 뒤져서 jpg파일을 찾는다.
        for j, f in enumerate(glob.glob(os.path.join(faces_folder_path, "*.jpg"))):
            points_box.append([])
            if ((index > 1) & (j < len(filename_list))):
                continue
            # points_box.append([])

            print("Processing file: {}".format(f))

            # 파일에서 이미지 불러오기
            img = dlib.load_rgb_image(f)

            # 불러온 이미지 데이터를 R과 B를 바꿔준다.
            cvImg = swapRGB2BGR(img)

            # 이미지를 두배로 키운다.
            #cvImg = cv2.resize(cvImg, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

            # 얼굴 인식 두번째 변수 1은 업샘플링을 한번 하겠다는 얘기인데
            # 업샘플링을하면 더 많이 인식할 수 있다고 한다.
            # 다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
            # 그냥 1이면 될 듯.
            dets = detector(img, 1)

            # 인식된 얼굴 개수 출력
            #print("Number of faces detected: {}".format(len(dets)))

            # 이제부터 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시할 것이다.
            for k, d in enumerate(dets):
                # k 얼굴 인덱스
                # d 얼굴 좌표
                #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #    k, d.left(), d.top(), d.right(), d.bottom()))

                # 인식된 좌표에서 랜드마크 추출
                shape = predictor(img, d)
                #print(shape.num_parts)
                # num_parts(랜드마크 구조체)를 하나씩 루프를 돌린다.
                # points_box.append([])
                for i in range(0, shape.num_parts):
                    # 해당 X,Y 좌표를 두배로 하지말고 키워 좌표를 얻고
                    x = shape.part(i).x
                    y = shape.part(i).y

                    # 좌표값 출력
                    #print(str(x) + " " + str(y))
                    # points_box.append([])
                    points_box[j].append((int(x), int(y)))

                    # 이미지 랜드마크 좌표 지점에 인덱스(랜드마크번호, 여기선 i)를 putText로 표시해준다.
                    cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                    # 랜드마크가 표시된 이미지를 openCV 윈도에 표시
                #배경
                points_box[j].append((0, 0))
                points_box[j].append((512, 0))
                points_box[j].append((1023, 0))
                points_box[j].append((0, 536))
                points_box[j].append((1023, 536))
                points_box[j].append((0, 1023))
                points_box[j].append((512, 1023))
                points_box[j].append((1023, 1023))
                #cv2.imshow('Face', cvImg)

            #print(points_box[j])
            # # 무한 대기를 타고 있다가 ESC 키가 눌리면 빠져나와 다음 이미지를 검색한다.
            # while True:
            #     if cv2.waitKey(0) == ESC_KEY:
            #         break;

        #cv2.destroyWindow('Face')

        # Read array of corresponding points
        if len(result_list) < 1:
            points1 = points_box[0]
        else:
            points1 = points_box[len(filename_list) + index - 2]
        points2 = points_box[index]
        points = [];

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))


        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        # Read triangles from tri.txt
        with open("generator/views/FaceMorph/final tri.txt") as file :
            for line in file :
                x, y, z = line.split()

                x = int(x)
                y = int(y)
                z = int(z)

                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [ points[x], points[y], points[z] ]

                # Morph one triangle at a time.
                morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


        # Display Result

        cv2.imwrite('generator/views/FaceMorph/morph_images/testresult{}.jpg'.format(index), np.uint8(imgMorph))
        result_list.append('testresult{}.jpg'.format(index))
        cv2.waitKey(0)
    cv2.imwrite('generator/static\img/final_result.jpg', np.uint8(imgMorph))

    #결과를 제외한 모든 합성에 이용한 사진 제거
    file_list1 = os.listdir("generator/views/FaceMorph/morph_images/")
    for file1 in file_list1:
        file_path1 = os.path.join("generator/views/FaceMorph/morph_images/", file1)
        os.remove(file_path1)

    # 결과를 제외한 모든 합성에 이용한 사진 제거
    file_list2 = os.listdir("generator/views/FaceMorph/images/")
    for file2 in file_list2:
        file_path2 = os.path.join("generator/views/FaceMorph/images/", file2)
        os.remove(file_path2)

    return None
